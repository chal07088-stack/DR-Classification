import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from drpair.data.manifest import write_csv
from drpair.experiments.arms import ARMS, PROPOSED
from drpair.runtime import run_directory, save_json
from drpair.teachers.targets import counterfactual_teacher, discrete_oracle, joint_effect
from drpair.training.predictions import load_pair

from .metrics import alpha_summary, boundary_metrics, subgroup_metrics
from .ordinal import decode, violations
from .statistics import locality_analysis, paired_bootstrap_delta


def make_report(cfg):
    run = run_directory(cfg)
    if not (run / "gate_protocol.json").exists():
        raise FileNotFoundError("Complete the gate stage first")
    threshold = cfg["evaluation"]["threshold"]
    metrics, ci, alpha_rows, joint_rows, boundary_rows = [], [], [], [], []
    for split in ("val_crossfit", "test"):
        source = "val" if split == "val_crossfit" else "test"
        s, f = load_pair(
            run / "predictions" / f"final_single_{source}.npz",
            run / "predictions" / f"final_paired_{source}.npz",
        )
        local = counterfactual_teacher(s["q"], f["q"], s["y"], threshold)
        oracle = discrete_oracle(s["q"], f["q"], s["y"], threshold)
        np.savez_compressed(
            run / "predictions" / f"discrete_oracle_diagnostic_{split}.npz", **oracle
        )
        groups = {
            "overall": np.ones(len(s["y"]), bool),
            "gap_ge1": abs(s["y"] - s["fellow_y"]) >= 1,
            "gap_ge2": abs(s["y"] - s["fellow_y"]) >= 2,
        }
        for arm in ARMS:
            with np.load(run / "predictions" / f"{arm}_{split}.npz", allow_pickle=False) as p:
                q, alpha = p["q"], p["alpha"]
                metrics.extend(
                    {"arm": arm, "split": split, **r}
                    for r in subgroup_metrics(s["y"], q, s["fellow_y"], s["patient"], threshold)
                )
                alpha_rows.extend({"arm": arm, "split": split, **r} for r in alpha_summary(alpha))
                effect = joint_effect(s["q"], f["q"], s["y"], alpha, threshold)
                local_active = (local["utility"] > 0) & (alpha >= 0.5)
                joint_rows.append(
                    {
                        "arm": arm,
                        "split": split,
                        "beneficial_joint": int((effect["utility"] > 0).sum()),
                        "harmful_joint": int((effect["utility"] < 0).sum()),
                        "neutral_joint": int((effect["utility"] == 0).sum()),
                        "local_positive_but_joint_nonpositive": int(
                            (local_active.any(1) & (effect["utility"] <= 0)).sum()
                        ),
                        "no_positive_local_but_joint_positive": int(
                            ((local["utility"] <= 0).all(1) & (effect["utility"] > 0)).sum()
                        ),
                        "pre_projection_violations": int(violations(p["raw"]).sum()),
                        "mean_squared_error_recovery": float(effect["utility"].mean()),
                    }
                )
                np.savez_compressed(
                    run / "predictions" / f"local_vs_joint_{arm}_{split}.npz",
                    local_utility=local["utility"],
                    joint_utility=effect["utility"],
                    patient=s["patient"],
                    side=s["side"],
                    alpha=alpha,
                )
                if split == "test" and arm in (
                    "A_single",
                    "D_global_learned",
                    "F_teacher_a",
                    PROPOSED,
                ):
                    for group, mask in groups.items():
                        result = paired_bootstrap_delta(
                            s["y"][mask],
                            decode(f["q"][mask], threshold),
                            decode(q[mask], threshold),
                            s["patient"][mask],
                            cfg["evaluation"]["bootstrap_repeats"],
                            cfg["evaluation"]["bootstrap_seed"],
                        )
                        ci.append({"arm_minus_paired": arm, "group": group, **result})
        boundary_rows.extend(
            {"split": split, **r} for r in boundary_metrics(s["y"], s["q"], f["q"], threshold)
        )
        slopes, interactions = locality_analysis(
            s["y"], s["fellow_y"], s["q"], f["q"], s["patient"], threshold
        )
        write_csv(run / "metrics" / f"locality_slopes_{split}.csv", slopes)
        write_csv(run / "metrics" / f"locality_interactions_{split}.csv", interactions)
    for name, rows in [
        ("arm_metrics", metrics),
        ("paired_bootstrap_ci", ci),
        ("alpha_summary", alpha_rows),
        ("local_vs_joint", joint_rows),
        ("boundary_harm", boundary_rows),
    ]:
        write_csv(run / "metrics" / f"{name}.csv", rows)
    figures = run / "figures"
    figures.mkdir(exist_ok=True)
    selected = ["A_single", "B_paired", "D_global_learned", "F_teacher_a", PROPOSED]
    groups = ["overall", "gap_ge1", "gap_ge2"]
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    colors = ["#3975B4", "#D77734", "#787878", "#6A63A8", "#279469"]
    for i, (arm, color) in enumerate(zip(selected, colors)):
        values = [
            next(
                r["qwk"]
                for r in metrics
                if r["arm"] == arm and r["split"] == "test" and r["group"] == g
            )
            for g in groups
        ]
        ax.bar(
            np.arange(3) + (i - 2) * 0.15,
            [np.nan if x is None else x for x in values],
            0.15,
            label=arm,
            color=color,
        )
    ax.set_xticks(np.arange(3), ["Overall", "Gap >= 1", "Gap >= 2"])
    ax.set_ylabel("Test QWK")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    fig.savefig(figures / "qwk_by_gap.png", dpi=200)
    plt.close(fig)
    lines = [
        "# PHASE 1 result report",
        "",
        "ConvNeXt-S, seed 42, patient-level five-fold OOF.",
        "",
        "Proposed arm fixed before test: G_teacher_b (one-boundary Teacher B).",
        "",
        "| Arm | Overall QWK | Gap >= 1 QWK | Gap >= 2 QWK |",
        "| --- | --- | --- | --- |",
    ]
    for arm in ARMS:
        values = [
            next(
                r["qwk"]
                for r in metrics
                if r["arm"] == arm and r["split"] == "test" and r["group"] == g
            )
            for g in groups
        ]
        lines.append(
            "| "
            + arm
            + " | "
            + " | ".join("NA" if v is None else f"{v:.4f}" for v in values)
            + " |"
        )
    lines += [
        "",
        "![QWK by gap](figures/qwk_by_gap.png)",
        "",
        "Patient bootstrap intervals quantify sampling uncertainty, not variation across training seeds.",
        "Teacher B labels are local U_b; joint deployed utility is reported separately.",
        "The 16-mask oracle is a diagnostic discrete policy class, not a learned method or a continuous-gate upper bound.",
        "Locality slopes are descriptive and prevalence-adjusted; they do not prove a causal mechanism.",
        "No external comparator paper has been reproduced by the generic reliability baseline.",
        "PHASE 1 ends here. PHASE 2/3 require separate decisions.",
    ]
    (run / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json(
        run / "completion.json",
        {
            "phase": 1,
            "status": "complete",
            "proposed": PROPOSED,
            "seeds": [42],
            "automatic_next_phase": False,
        },
    )
    return run / "REPORT.md"
