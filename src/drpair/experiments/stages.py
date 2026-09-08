"""Explicit review boundary between final-expert diagnostics and OOF/gate studies."""

import json

from drpair.runtime import experiment_fingerprint, file_hash, run_directory


def require_review(cfg, reviewed):
    if not reviewed:
        raise ValueError(
            "PHASE 1B requires explicit --reviewed-1a after reviewing the PHASE 1A report"
        )
    run = run_directory(cfg)
    path = run / "diagnostics" / "phase1a" / "summary.json"
    if not path.is_file():
        raise ValueError("Complete PHASE 1A before continuing")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not summary.get("complete") or summary.get("fingerprint") != experiment_fingerprint(cfg):
        raise ValueError("PHASE 1A report does not match the current experiment")
    for kind in ("single", "paired"):
        prediction = run / "predictions" / f"final_{kind}_val.npz"
        if summary["prediction_sha256"][kind] != file_hash(prediction):
            raise ValueError("Validation predictions changed since PHASE 1A review")
    return summary


def execute_stage(cfg, stage, reviewed=False, update=lambda *a, **kw: None):
    if stage not in ("1a", "1b"):
        raise ValueError("Choose stage 1a or 1b")
    if stage == "1b":
        require_review(cfg, reviewed)
    from drpair.experiments.experts import expert_jobs, run_experts

    if stage == "1a":
        from drpair.evaluation.feasibility import make_feasibility_report

        update("final_experts")
        run_experts(cfg, ["final_single", "final_paired"], evaluation_splits=("val",))
        update("validation_diagnostics")
        report = make_feasibility_report(cfg)
        update("review_boundary", "awaiting_review", report=str(report))
        return report

    from drpair.evaluation.report import make_report
    from drpair.experiments.experts import predict_final_experts
    from drpair.experiments.gate_study import run_gate_study

    update("expert_oof", review_confirmed=True)
    run_experts(cfg, [j for j in expert_jobs() if j.startswith("oof")])
    update("final_test_predictions")
    predict_final_experts(cfg, splits=("test",))
    update("teachers_and_gates")
    run_gate_study(cfg)
    update("report")
    report = make_report(cfg)
    update("finished", "complete", report=str(report))
    return report
