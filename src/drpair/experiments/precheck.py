"""Budget safeguard on inner-selection predictions, not a final hypothesis test."""

import numpy as np

from drpair.data.manifest import write_csv
from drpair.evaluation.metrics import expert_boundary_diagnostics
from drpair.runtime import run_directory, save_json
from drpair.teachers.targets import counterfactual_teacher, discrete_oracle
from drpair.training.predictions import load_pair


class BoundaryReviewRequired(RuntimeError):
    pass


def boundary_opportunities(s, f, threshold=0.5):
    single = expert_boundary_diagnostics(s["y"], s["q"], threshold, s["patient"])
    paired = expert_boundary_diagnostics(s["y"], f["q"], threshold, s["patient"])
    local = counterfactual_teacher(s["q"], f["q"], s["y"], threshold)
    oracle = discrete_oracle(s["q"], f["q"], s["y"], threshold)
    rows = []
    for b in range(4):
        rows.append(
            {
                "boundary": b,
                "true_positive": single[b]["positive"],
                "positive_patients": single[b]["positive_patients"],
                "single_predicted_positive": single[b]["predicted_positive"],
                "paired_predicted_positive": paired[b]["predicted_positive"],
                "single_recall": single[b]["recall"],
                "paired_recall": paired[b]["recall"],
                "single_auc": single[b]["auc"],
                "paired_auc": paired[b]["auc"],
                "single_average_precision": single[b]["average_precision"],
                "paired_average_precision": paired[b]["average_precision"],
                "local_positive_utility": int((local["utility"][:, b] > 0).sum()),
                "local_negative_utility": int((local["utility"][:, b] < 0).sum()),
                "oracle_selected_boundary": int((oracle["mask"][:, b] > 0).sum()),
                "oracle_boundary_in_beneficial_joint_mask": int(
                    ((oracle["mask"][:, b] > 0) & (oracle["utility"] > 0)).sum()
                ),
            }
        )
    review = single[3]["silent_at_threshold"] and paired[3]["silent_at_threshold"]
    return rows, bool(review), local, oracle


def check_initial_experts(cfg):
    run = run_directory(cfg)
    s, f = load_pair(
        run / "predictions" / "final_single_selection_holdout.npz",
        run / "predictions" / "final_paired_selection_holdout.npz",
    )
    rows, silent, local, oracle = boundary_opportunities(s, f, cfg["evaluation"]["threshold"])
    out = run / "diagnostics"
    write_csv(out / "initial_boundary_opportunities.csv", rows)
    np.savez_compressed(
        out / "initial_boundary_opportunities.npz",
        patient=s["patient"],
        side=s["side"],
        local_utility=local["utility"],
        oracle_mask=oracle["mask"],
        oracle_joint_utility=oracle["utility"],
    )
    stop = silent and cfg.get("diagnostics", {}).get("stop_if_both_b3_silent", True)
    save_json(
        out / "initial_expert_check.json",
        {
            "status": "needs_review" if stop else "continue",
            "source": "inner_validation_used_for_checkpoint_selection_not_unbiased_oof",
            "both_b3_silent_at_threshold": silent,
            "final_evidence": False,
            "additional_training_fits": 0,
            "boundary_results": rows,
            "interpretation": "Counts alone neither establish nor reject boundary-wise benefit",
            "unchanged_loss_and_threshold": True,
        },
    )
    print(f"Initial expert boundary check: {'NEEDS REVIEW' if stop else 'continue'}", flush=True)
    if stop:
        raise BoundaryReviewRequired(
            "Both selected experts predict zero B3-positive cases despite positive labels. "
            "Review diagnostics before spending the remaining refit/OOF budget. "
            "No loss weights or thresholds were changed automatically."
        )
