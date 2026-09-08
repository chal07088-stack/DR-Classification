"""Validation-only PHASE 1A diagnostics with an unconditional human review stop."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from drpair.data.manifest import read_fold_assignment, write_csv
from drpair.gates.features import fellow_predictions, gate_features
from drpair.runtime import experiment_fingerprint, file_hash, run_directory, save_json
from drpair.teachers.targets import counterfactual_teacher, discrete_oracle
from drpair.training.predictions import load_pair

from .calibration import crossfit_calibration
from .headroom import rollback_grid
from .metrics import boundary_metrics, grade_metrics
from .ordinal import blend, decode
from .statistics import locality_analysis, paired_bootstrap_delta


def groups(y, fellow):
    gap = abs(np.asarray(y) - np.asarray(fellow))
    return {
        "overall": np.ones(len(gap), bool),
        "gap0": gap == 0,
        "gap1": gap == 1,
        "gap_ge1": gap >= 1,
        "gap_ge2": gap >= 2,
    }


def clinical_counts(target, predicted):
    target, predicted = np.asarray(target, bool), np.asarray(predicted, bool)
    tp, tn = int((target & predicted).sum()), int((~target & ~predicted).sum())
    positives, negatives = int(target.sum()), int((~target).sum())
    return {
        "positive": positives,
        "negative": negatives,
        "tp": tp,
        "fn": positives - tp,
        "tn": tn,
        "fp": negatives - tn,
        "sensitivity": tp / positives if positives else None,
        "specificity": tn / negatives if negatives else None,
        "miss_rate": 1 - tp / positives if positives else None,
    }


def clustered_rate_delta(target, reference, candidate, patient, repeats, seed=42):
    """Candidate minus reference sensitivity, retaining both eyes of sampled patients."""
    ids, inverse = np.unique(patient, return_inverse=True)
    target = np.asarray(target, bool)
    if len(ids) < 2 or not target.any():
        return {"delta": None, "ci_low": None, "ci_high": None, "valid_replicates": 0}
    den = np.bincount(inverse, weights=target, minlength=len(ids))
    num = np.bincount(
        inverse, weights=target * (np.asarray(candidate, float) - reference), minlength=len(ids)
    )
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        idx = rng.integers(0, len(ids), len(ids))
        d = den[idx].sum()
        if d:
            values.append(num[idx].sum() / d)
    return {
        "delta": float(num.sum() / den.sum()),
        "ci_low": float(np.quantile(values, 0.025)) if values else None,
        "ci_high": float(np.quantile(values, 0.975)) if values else None,
        "valid_replicates": len(values),
    }


def shrinkage_design(y, fellow):
    """Signed fellow-minus-target difference, adjusted by target-grade indicators."""
    y, fellow = np.asarray(y), np.asarray(fellow)
    return np.column_stack(
        [np.ones(len(y)), fellow - y] + [(y == g).astype(float) for g in range(1, 5)]
    )


def shrinkage_permutation(y, fellow, qs, qf, repeats=1000, seed=42):
    """Restricted null: fellow grades shuffled inside each target grade.

    Severity boundaries are not exchangeable, so an unrestricted shuffle would not
    be a defensible null. Holding every eye inside its own target grade asks the
    narrower question the slope is meant to answer: does the fusion shift track the
    fellow beyond what the target grade already explains?

    Descriptive, with two limits worth stating in any write-up. It permutes eyes
    rather than patients, so within-patient dependence is only partly represented,
    and a small p-value rules out this null, not confounding in general.
    """
    y, fellow = np.asarray(y), np.asarray(fellow)
    response = qf.sum(1) - qs.sum(1)
    x = shrinkage_design(y, fellow)
    definition = "fellow grade shuffled within target grade; descriptive, not causal"
    if int(np.linalg.matrix_rank(x)) != x.shape[1] or repeats < 1:
        return {
            "observed_slope": None,
            "p_permutation": None,
            "repeats": 0,
            "null_definition": definition,
        }

    def slope(design):
        return float((np.linalg.pinv(design.T @ design) @ design.T @ response)[1])

    observed = slope(x)
    strata = [np.flatnonzero(y == grade) for grade in np.unique(y)]
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats)
    for r in range(repeats):
        shuffled = fellow.copy()
        for index in strata:
            shuffled[index] = shuffled[rng.permutation(index)]
        draws[r] = slope(shrinkage_design(y, shuffled))
    extreme = int(np.sum(np.abs(draws) >= abs(observed)))
    return {
        "observed_slope": observed,
        "null_mean": float(draws.mean()),
        "null_sd": float(draws.std()),
        "p_permutation": float((extreme + 1) / (repeats + 1)),
        "repeats": int(repeats),
        "null_definition": definition,
    }


def shrinkage_association(y, fellow, qs, qf, patient):
    """Target-grade adjusted association, explicitly not a causal shrinkage estimate."""
    y, fellow = np.asarray(y), np.asarray(fellow)
    x = shrinkage_design(y, fellow)
    response = qf.sum(1) - qs.sum(1)
    rank = int(np.linalg.matrix_rank(x))
    if rank != x.shape[1]:
        return {
            "slope": None,
            "ci_low": None,
            "ci_high": None,
            "design_rank": rank,
            "interpretation": "Insufficient support for grade-adjusted slope",
        }
    bread = np.linalg.pinv(x.T @ x)
    beta = bread @ x.T @ response
    residual = response - x @ beta
    ids, inverse = np.unique(patient, return_inverse=True)
    scores = np.zeros((len(ids), x.shape[1]))
    np.add.at(scores, inverse, x * residual[:, None])
    correction = len(ids) / max(1, len(ids) - 1) * (len(y) - 1) / max(1, len(y) - x.shape[1])
    covariance = bread @ scores.T @ scores @ bread * correction
    se = float(np.sqrt(max(covariance[1, 1], 0)))
    return {
        "slope": float(beta[1]),
        "cluster_se": se,
        "ci_low": float(beta[1] - 1.96 * se),
        "ci_high": float(beta[1] + 1.96 * se),
        "design_rank": rank,
        "response": "sum(q_fusion)-sum(q_single)",
        "predictor": "fellow_grade-target_grade",
        "interpretation": "Descriptive association adjusted for target grade; not causal evidence",
    }


def predictability_probe(s, f, folds):
    """Cheap raw-feature diagnostic; not the proposed Teacher B gate or a test result."""
    fellow = fellow_predictions(s["q"], s["patient"], s["side"])
    features, names = gate_features(s["q"], f["q"], fellow)
    utility = counterfactual_teacher(s["q"], f["q"], s["y"])["utility"]
    targets = utility > 0
    probabilities = np.zeros_like(utility)
    models = []
    for fold in np.unique(folds):
        train, held = folds != fold, folds == fold
        scaler = StandardScaler().fit(features[train])
        tx, hx = scaler.transform(features[train]), scaler.transform(features[held])
        for b in range(4):
            y = targets[train, b]
            record = {
                "fold": int(fold),
                "boundary": b,
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
                "positive": int(y.sum()),
            }
            if len(np.unique(y)) == 2:
                model = LogisticRegression(C=1, max_iter=1000, random_state=42).fit(tx, y)
                probabilities[held, b] = model.predict_proba(hx)[:, 1]
                record.update(
                    coefficient=model.coef_[0].tolist(), intercept=float(model.intercept_[0])
                )
            else:
                probabilities[held, b] = float(y.mean())
                record["constant"] = float(y.mean())
            models.append(record)
    rows = []
    for b in range(4):
        target = targets[:, b]
        rows.append(
            {
                "boundary": b,
                "positive": int(target.sum()),
                "prevalence": float(target.mean()),
                "average_precision": float(average_precision_score(target, probabilities[:, b]))
                if target.any()
                else None,
                "auc": float(roc_auc_score(target, probabilities[:, b]))
                if len(np.unique(target)) == 2
                else None,
            }
        )
    return (
        rows,
        {
            "x": features,
            "utility": utility,
            "probability": probabilities,
            "feature_names": np.array(names),
        },
        models,
    )


def make_feasibility_report(cfg):
    run = run_directory(cfg)
    directory = run / "diagnostics" / "phase1a"
    directory.mkdir(parents=True, exist_ok=True)
    single_path, fusion_path = [
        run / "predictions" / f"final_{kind}_val.npz" for kind in ("single", "paired")
    ]
    s, f = load_pair(single_path, fusion_path)
    if not np.all(s["split"] == "val") or not np.all(f["split"] == "val"):
        raise ValueError("PHASE 1A diagnostics accept official validation only")
    identity = [{"patient": str(p), "side": str(side)} for p, side in zip(s["patient"], s["side"])]
    folds = read_fold_assignment(cfg["data"]["fold_manifest"], identity, "val")
    y, patient = s["y"], s["patient"]
    masks = groups(y, s["fellow_y"])
    settings = cfg["diagnostics"]
    methods, specificities = settings["calibration_methods"], settings["specificity_targets"]
    threshold = cfg["evaluation"]["threshold"]
    repeats = settings["bootstrap_repeats"]
    metrics, intervals, clinical, clinical_intervals, events = [], [], [], [], []
    headroom_rows, calibration_records, mechanism = [], {}, {}
    for method in methods:
        print(f"PHASE 1A diagnostics: {method} calibration and rollback grids", flush=True)
        qs, ts, rs = crossfit_calibration(s["q"], y, patient, folds, method, specificities)
        qf, tf, rf = crossfit_calibration(f["q"], y, patient, folds, method, specificities)
        calibration_records[method] = {"single": rs, "paired": rf}
        binary = discrete_oracle(qs, qf, y, threshold)
        grid = rollback_grid(qs, qf, y, settings["alpha_grid"], threshold)
        fixed, _ = blend(qs, qf, np.full_like(qs, 0.5))
        predictions = {
            "single": qs,
            "paired": qf,
            "fixed50": fixed,
            "binary_diagnostic": binary["q"],
            "grid_diagnostic": grid["q"],
        }
        np.savez_compressed(
            directory / f"{method}_diagnostics.npz",
            patient=patient,
            side=s["side"],
            y=y,
            fellow_y=s["fellow_y"],
            fold=folds,
            qs=qs,
            qf=qf,
            single_thresholds=ts,
            paired_thresholds=tf,
            binary_alpha=binary["mask"],
            binary_q=binary["q"],
            binary_utility=binary["utility"],
            grid_alpha=grid["alpha"],
            grid_q=grid["q"],
            grid_utility=grid["utility"],
        )
        local = counterfactual_teacher(qs, qf, y, threshold)
        for row in boundary_metrics(y, qs, qf, threshold):
            b = row["boundary"]
            events.append(
                {
                    "calibration": method,
                    **row,
                    "single_positive": int((qs[:, b] >= threshold).sum()),
                    "paired_positive": int((qf[:, b] >= threshold).sum()),
                    "positive_patients": len(np.unique(patient[y > b])),
                    "local_benefit_eyes": int((local["utility"][:, b] > 0).sum()),
                    "local_benefit_patients": len(np.unique(patient[local["utility"][:, b] > 0])),
                    "binary_selected": int((binary["mask"][:, b] > 0).sum()),
                    "grid_selected": int((grid["alpha"][:, b] > 0).sum()),
                }
            )
        mechanism[method] = {
            **shrinkage_association(y, s["fellow_y"], qs, qf, patient),
            "permutation": shrinkage_permutation(
                y, s["fellow_y"], qs, qf, repeats, cfg["evaluation"]["bootstrap_seed"]
            ),
        }
        for group, mask in masks.items():
            if not mask.any():
                continue
            for arm, q in predictions.items():
                metrics.append(
                    {
                        "calibration": method,
                        "group": group,
                        "arm": arm,
                        "n_patients": len(np.unique(patient[mask])),
                        **grade_metrics(y[mask], q[mask], threshold),
                    }
                )
            for name, result in [("binary", binary), ("grid", grid)]:
                u = result["utility"][mask]
                headroom_rows.append(
                    {
                        "calibration": method,
                        "group": group,
                        "policy": name,
                        "benefited_eyes": int((u > 0).sum()),
                        "benefited_patients": len(np.unique(patient[mask][u > 0])),
                        "mean_squared_error_recovery": float(u.mean()),
                    }
                )
            if group in ("overall", "gap_ge1", "gap_ge2"):
                for reference, candidate in [
                    ("single", "paired"),
                    ("paired", "binary_diagnostic"),
                    ("paired", "grid_diagnostic"),
                ]:
                    delta = paired_bootstrap_delta(
                        y[mask],
                        decode(predictions[reference][mask], threshold),
                        decode(predictions[candidate][mask], threshold),
                        patient[mask],
                        repeats,
                    )
                    intervals.append(
                        {
                            "calibration": method,
                            "group": group,
                            "reference": reference,
                            "candidate": candidate,
                            **delta,
                        }
                    )
            for j, grade in enumerate((2, 3)):
                for k, specificity in enumerate(specificities):
                    valid = mask & np.isfinite(ts[:, j, k]) & np.isfinite(tf[:, j, k])
                    if not valid.any():
                        continue
                    target = y[valid] >= grade
                    ps = qs[valid, grade - 1] >= ts[valid, j, k]
                    pf = qf[valid, grade - 1] >= tf[valid, j, k]
                    for arm, pred in [("single", ps), ("paired", pf)]:
                        clinical.append(
                            {
                                "calibration": method,
                                "group": group,
                                "arm": arm,
                                "grade": grade,
                                "target_training_specificity": specificity,
                                "n_eyes": int(valid.sum()),
                                "n_patients": len(np.unique(patient[valid])),
                                **clinical_counts(target, pred),
                            }
                        )
                    if group in ("overall", "gap_ge1", "gap_ge2"):
                        delta = clustered_rate_delta(target, ps, pf, patient[valid], repeats)
                        clinical_intervals.append(
                            {
                                "calibration": method,
                                "group": group,
                                "grade": grade,
                                "target_training_specificity": specificity,
                                "reference": "single",
                                "candidate": "paired",
                                **delta,
                            }
                        )
    locality, interactions = locality_analysis(y, s["fellow_y"], s["q"], f["q"], patient, threshold)
    probe_rows, probe, probe_models = predictability_probe(s, f, folds)
    np.savez_compressed(
        directory / "raw_predictability_probe.npz",
        patient=patient,
        side=s["side"],
        fold=folds,
        **probe,
    )
    save_json(directory / "calibration_models.json", calibration_records)
    save_json(directory / "predictability_models.json", probe_models)
    save_json(directory / "shrinkage_association.json", mechanism)
    for name, rows in [
        ("grade_metrics", metrics),
        ("qwk_deltas", intervals),
        ("boundary_events", events),
        ("clinical_operating_points", clinical),
        ("clinical_sensitivity_deltas", clinical_intervals),
        ("headroom", headroom_rows),
        ("predictability_probe", probe_rows),
        ("locality", locality),
        ("locality_interactions", interactions),
    ]:
        if rows:
            write_csv(directory / f"{name}.csv", rows)
    _figures(directory, metrics, events)
    report = directory / "REPORT.md"
    lines = [
        "# PHASE 1A: validation review",
        "",
        "Status: awaiting human review. OOF and proposed gates have not started.",
        "",
        "EyePACS official validation; ConvNeXt-S; seed 42. Expert epochs were selected inside official train.",
        "",
        "Calibration and clinical operating points are fit on four patient folds and evaluated on the heldout fifth.",
        "Realized specificity is reported; it need not equal the requested training specificity on heldout data.",
        "",
        "## Raw predictions",
        "",
        "| Group | Method | Eyes | QWK | Severe under-rate |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in metrics:
        if row["calibration"] == "raw":
            lines.append(
                f"| {row['group']} | {row['arm']} | {row['n_eyes']} | {_number(row['qwk'])} | {_number(row['severe_under_rate'])} |"
            )
    lines += [
        "",
        "![Grade metrics](grade_metrics.png)",
        "",
        "![Calibration](calibration_brier.png)",
        "",
        "## Review these files",
        "",
        "- qwk_deltas.csv: paired patient-bootstrap differences (candidate minus reference).",
        "- headroom.csv: squared-error recovery and benefited eyes/patients for label-aware policies.",
        "- boundary_events.csv: boundary support, harmful/beneficial flips and local Teacher B opportunities.",
        "- clinical_operating_points.csv: grade >=2 and severe-or-worse grade >=3 sensitivity, specificity and misses.",
        "- clinical_sensitivity_deltas.csv: clustered sensitivity differences at training-selected operating points.",
        "- predictability_probe.csv: patient-cross-fitted raw-feature logistic probes; compare AP with prevalence.",
        "- shrinkage_association.json and locality.csv: descriptive, grade-adjusted associations.",
        "",
        "## Interpretation limits",
        "",
        "Binary and finite-grid policies use labels and minimize per-eye squared grade error; neither is a QWK upper bound or a learned result.",
        "Bootstrap intervals are conditional on these trained models and fitted cross-validation predictions; they exclude training/calibrator seed uncertainty.",
        "A zero 0.5-threshold event count does not establish zero ranking/calibration signal.",
        "The predictability probe is exploratory, not the Teacher B gate and not a causal explanation.",
        "Grade >=3 is severe-or-worse DR, not a complete VTDR/DME endpoint.",
        "No automatic +0.005 cutoff or automatic go/no-go decision is used. Review effect sizes, support and uncertainty.",
        "Test predictions are not loaded by this report. Clinical efficacy, novelty and external generalization are not established.",
        "",
        "## Before PHASE 1B",
        "",
        "Review the report and freeze endpoints, comparator protocol and calibration choices before test access.",
        "Explicit continuation: uv run --locked python scripts/run_phase1.py --stage 1b --reviewed-1a",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    save_json(
        directory / "summary.json",
        {
            "complete": True,
            "status": "awaiting_review",
            "stage": "1a",
            "automatic_next_stage": False,
            "test_evaluated": False,
            "n_eyes": len(y),
            "n_patients": len(np.unique(patient)),
            "calibration_methods": methods,
            "alpha_grid": settings["alpha_grid"],
            "grid_policies": len(settings["alpha_grid"]) ** 4,
            "fingerprint": experiment_fingerprint(cfg),
            "prediction_sha256": {
                "single": file_hash(single_path),
                "paired": file_hash(fusion_path),
            },
            "report": str(report),
        },
    )
    return report


def _number(value):
    return "NA" if value is None else f"{value:.4f}"


def _figures(directory, metrics, events):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["overall", "gap0", "gap1", "gap_ge2"]
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    x = np.arange(len(names))
    for j, (arm, color) in enumerate(
        [("single", "#e07632"), ("paired", "#2689ba"), ("grid_diagnostic", "#777777")]
    ):
        values = [
            next(
                (
                    r["qwk"]
                    for r in metrics
                    if r["calibration"] == "raw" and r["arm"] == arm and r["group"] == g
                ),
                None,
            )
            for g in names
        ]
        ax.bar(
            x + (j - 1) * 0.24,
            [np.nan if v is None else v for v in values],
            0.24,
            label=arm,
            color=color,
        )
    ax.set(
        xticks=x,
        xticklabels=names,
        ylabel="QWK",
        title="Validation: model predictions and label-aware grid diagnostic",
    )
    ax.legend(frameon=False)
    ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(directory / "grade_metrics.png", dpi=220)
    fig.savefig(directory / "grade_metrics.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    methods = ["raw", "platt", "isotonic"]
    for arm, color in [("single", "#e07632"), ("fusion", "#2689ba")]:
        values = [
            np.mean([r[f"brier_{arm}"] for r in events if r["calibration"] == m]) for m in methods
        ]
        ax.plot(methods, values, marker="o", label=arm, color=color)
    ax.set(ylabel="Mean boundary Brier loss", title="Patient-cross-fitted calibration")
    ax.legend(frameon=False)
    ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(directory / "calibration_brier.png", dpi=220)
    fig.savefig(directory / "calibration_brier.pdf")
    plt.close(fig)
