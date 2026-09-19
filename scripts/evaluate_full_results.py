from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader

from drpair.full_baselines.metrics import metric_dict
from drpair.full_baselines.runtime import (
    DirectionalDataset,
    build_model,
    image_root,
    infer_batch,
    method_transform,
    rows_for_run,
)

SEEDS = [42, 123, 128]
METHODS = ["huang_single", "huang_pff", "drgcnn_single", "drgcnn_bffn", "d_odr", "aor_dr"]
PAIR = {"huang_pff": "huang_single", "drgcnn_bffn": "drgcnn_single"}
DISPLAY = {
    "huang_single": "Huang Single",
    "huang_pff": "Huang PFF",
    "drgcnn_single": "DRGCNN Single",
    "drgcnn_bffn": "DRGCNN BFFN",
    "d_odr": "D-ODR",
    "aor_dr": "AOR-DR",
}
TYPE = {
    "huang_single": "Single",
    "huang_pff": "Bilateral",
    "drgcnn_single": "Single",
    "drgcnn_bffn": "Bilateral",
    "d_odr": "Single / Ordinal",
    "aor_dr": "Single / Ordinal",
}


def prediction_path(repo, dataset, method, seed):
    return repo / f"outputs/predictions/{dataset}/{method}/seed{seed}_test.csv"


def load(repo, dataset, method, seed):
    frame = pd.read_csv(prediction_path(repo, dataset, method, seed))
    frame["sample_id"] = frame.sample_id.astype(str)
    if frame.sample_id.duplicated().any():
        raise RuntimeError(f"duplicate sample IDs {dataset}/{method}/{seed}")
    return frame.sort_values("sample_id").reset_index(drop=True)


def values(frame):
    return metric_dict(frame.y_true.to_numpy(int), frame.y_pred.to_numpy(float))


def change(single, bilateral):
    merged = single.merge(bilateral, on="sample_id", suffixes=("_s", "_b"), validate="one_to_one")
    if len(merged) != len(single) or not np.array_equal(merged.y_true_s, merged.y_true_b):
        raise RuntimeError("prediction alignment failure")
    es = np.abs(merged.y_pred_s - merged.y_true_s)
    eb = np.abs(merged.y_pred_b - merged.y_true_s)
    correct = es == 0
    wrong = ~correct
    return {
        "N": len(merged),
        "Benefit": 100 * np.mean(eb < es),
        "Neutral": 100 * np.mean(eb == es),
        "Harm": 100 * np.mean(eb > es),
        "Preservation": 100 * np.mean(eb[correct] == 0) if correct.any() else np.nan,
        "Recovery": 100 * np.mean(eb[wrong] == 0) if wrong.any() else np.nan,
        "Delta_QWK": values(bilateral)["qwk"] - values(single)["qwk"],
        "Delta_MAE": values(bilateral)["mae"] - values(single)["mae"],
        "Prediction_Change": 100 * np.mean(merged.y_pred_s != merged.y_pred_b),
    }


def table41(repo, dataset):
    detail = []
    for method in METHODS:
        for seed in SEEDS:
            detail.append(
                {
                    "Method": DISPLAY[method],
                    "method_id": method,
                    "Type": TYPE[method],
                    "Input": "Target+Fellow" if method in PAIR else "Target",
                    "seed": seed,
                    **values(load(repo, dataset, method, seed)),
                }
            )
    pd.DataFrame(detail).to_csv(
        repo / f"outputs/metrics/{dataset}_per_seed_metrics.csv", index=False
    )
    rows = []
    for method in METHODS:
        subset = [r for r in detail if r["method_id"] == method]
        row = {
            "Method": DISPLAY[method],
            "Type": TYPE[method],
            "Input": "Target+Fellow" if method in PAIR else "Target",
        }
        for key, label in (
            ("qwk", "QWK"),
            ("mae", "MAE"),
            ("accuracy", "Accuracy"),
            ("macro_f1", "MacroF1"),
        ):
            x = np.asarray([r[key] for r in subset])
            row[f"{label}_mean"] = x.mean()
            row[f"{label}_std"] = x.std(ddof=1)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(repo / f"outputs/tables/{dataset}_table4_1.csv", index=False)
    return out, detail


def table42(repo, dataset):
    rows = []
    for paired, single in PAIR.items():
        for seed in SEEDS:
            rows.append(
                {
                    "Comparison": f"{DISPLAY[paired]} vs {DISPLAY[single]}",
                    "seed": seed,
                    "scope": "per_seed",
                    **change(load(repo, dataset, single, seed), load(repo, dataset, paired, seed)),
                }
            )
        subset = rows[-len(SEEDS) :]
        summary = {
            "Comparison": f"{DISPLAY[paired]} vs {DISPLAY[single]}",
            "seed": "mean±std",
            "scope": "summary",
        }
        for key in [
            "N",
            "Benefit",
            "Neutral",
            "Harm",
            "Preservation",
            "Recovery",
            "Delta_QWK",
            "Delta_MAE",
            "Prediction_Change",
        ]:
            x = np.asarray([r[key] for r in subset], float)
            summary[f"{key}_mean"] = np.nanmean(x)
            summary[f"{key}_std"] = np.nanstd(x, ddof=1)
        rows.append(summary)
    out = pd.DataFrame(rows)
    out.to_csv(repo / f"outputs/tables/{dataset}_table4_2.csv", index=False)
    return out


def gap_analysis(repo, dataset):
    rows = []
    for method in METHODS:
        for seed in SEEDS:
            frame = load(repo, dataset, method, seed)
            single = load(repo, dataset, PAIR[method], seed) if method in PAIR else None
            for group, predicate in (
                ("gap0", lambda x: x == 0),
                ("gap1", lambda x: x == 1),
                ("gap2plus", lambda x: x >= 2),
            ):
                part = frame[predicate(frame.inter_eye_gap)].copy()
                row = {
                    "dataset": dataset,
                    "method": DISPLAY[method],
                    "seed": seed,
                    "gap_group": group,
                    "N_patients": part.patient_id.nunique(),
                    "N_target_eyes": len(part),
                    **values(part),
                }
                if method in PAIR:
                    row.update(change(single[single.sample_id.isin(part.sample_id)], part))
                row["warning"] = "LOW_N" if group == "gap2plus" and row["N_patients"] < 30 else ""
                rows.append(row)
    pd.DataFrame(rows).to_csv(repo / f"outputs/analysis/{dataset}_gap_analysis.csv", index=False)


def aggregate_confusions(repo, dataset):
    for method in METHODS:
        matrices = []
        for seed in SEEDS:
            path = repo / f"outputs/metrics/{dataset}/{method}/seed{seed}_test_confusion.csv"
            matrices.append(np.loadtxt(path, delimiter=",", dtype=int))
        aggregate = np.sum(matrices, axis=0)
        np.savetxt(
            repo / f"outputs/metrics/{dataset}/{method}/aggregate_confusion.csv",
            aggregate,
            delimiter=",",
            fmt="%d",
        )
        support = aggregate.sum(1)
        predicted = aggregate.sum(0)
        true_positive = np.diag(aggregate)
        precision = np.divide(
            true_positive, predicted, out=np.zeros(5, float), where=predicted != 0
        )
        recall = np.divide(true_positive, support, out=np.zeros(5, float), where=support != 0)
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros(5, float),
            where=(precision + recall) != 0,
        )
        pd.DataFrame(
            {
                "grade": range(5),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        ).to_csv(repo / f"outputs/metrics/{dataset}/{method}/aggregate_classwise.csv", index=False)


def random_map(repo, dataset):
    path = repo / f"outputs/random_fellow/{dataset}_random_fellow_map_seed4242_k10.csv"
    if path.is_file():
        return pd.read_csv(path)
    test = [r for r in rows_for_run(repo, dataset) if r["run_split"] == "test"]
    rows = []
    for target in test:
        candidates = [
            r
            for r in test
            if r["patient_id"] != target["patient_id"] and r["fellow_eye"] == target["fellow_eye"]
        ]
        rng = random.Random(4242 + int.from_bytes(target["sample_id"].encode(), "little") % (2**31))
        chosen = rng.sample(candidates, 10)
        for k, c in enumerate(chosen):
            rows.append(
                {
                    "dataset": dataset,
                    "sample_id": target["sample_id"],
                    "replicate": k,
                    "random_patient_id": c["patient_id"],
                    "random_fellow_eye": c["fellow_eye"],
                    "random_fellow_image_path": c["fellow_image_path"],
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return pd.DataFrame(rows)


def random_control(repo, dataset):
    mapping = random_map(repo, dataset)
    device = torch.device("cuda:0")
    stats = (
        json.loads((repo / "outputs/data_audit/messidor_rgb_stats.json").read_text())
        if dataset == "messidor2"
        else None
    )
    mean = (
        [0.425753653049469, 0.29737451672554016, 0.21293757855892181]
        if stats is None
        else stats["mean"]
    )
    std = (
        [0.27670302987098694, 0.20240527391433716, 0.1686241775751114]
        if stats is None
        else stats["std"]
    )
    test = [r for r in rows_for_run(repo, dataset) if r["run_split"] == "test"]
    output = []
    for method, single_method in PAIR.items():
        method_mean, method_std = mean, std
        if dataset == "eyepacs" and method == "drgcnn_bffn":
            method_mean = [0.4844570755958557, 0.2249917984008789, 0.0770205482840538]
            method_std = [0.29392191767692566, 0.14738348126411438, 0.06418044865131378]
        transform = method_transform(method, False, method_mean, method_std)
        for seed in SEEDS:
            parent = repo / f"outputs/checkpoints/{dataset}/{single_method}/seed{seed}/best.pt"
            paired = repo / f"outputs/checkpoints/{dataset}/{method}/seed{seed}/best.pt"
            model = build_model(repo, method, False, parent).to(device)
            model.load_state_dict(
                torch.load(paired, map_location=device, weights_only=False)["model_state"]
            )
            model.eval()
            single = load(repo, dataset, single_method, seed)
            true = load(repo, dataset, method, seed)
            base = change(single, true)
            output.append(
                {
                    "dataset": dataset,
                    "method": DISPLAY[method],
                    "seed": seed,
                    "condition": "true",
                    "replicate": -1,
                    "QWK": values(true)["qwk"],
                    "Harm": base["Harm"],
                    "Prediction_Change": base["Prediction_Change"],
                }
            )
            for rep in range(10):
                mp = mapping[mapping.replicate == rep].set_index("sample_id")
                changed = []
                for row in test:
                    r = dict(row)
                    r["fellow_image_path"] = mp.loc[r["sample_id"], "random_fellow_image_path"]
                    changed.append(r)
                loader = DataLoader(
                    DirectionalDataset(changed, image_root(repo, dataset), transform),
                    batch_size=64,
                    shuffle=False,
                    num_workers=8,
                    pin_memory=True,
                )
                pred = []
                with torch.inference_mode():
                    for target, fellow, _, _ in loader:
                        score, _ = infer_batch(model, method, target.to(device), fellow.to(device))
                        pred.extend(np.clip(np.rint(score.cpu().numpy()), 0, 4).astype(int))
                predicted = pd.DataFrame(
                    {"sample_id": [r["sample_id"] for r in changed], "random_pred": pred}
                )
                random_frame = (
                    true.drop(columns=["y_pred"])
                    .merge(predicted, on="sample_id", validate="one_to_one")
                    .rename(columns={"random_pred": "y_pred"})
                )
                delta = change(single, random_frame)
                output.append(
                    {
                        "dataset": dataset,
                        "method": DISPLAY[method],
                        "seed": seed,
                        "condition": "random",
                        "replicate": rep,
                        "QWK": values(random_frame)["qwk"],
                        "Harm": delta["Harm"],
                        "Prediction_Change": delta["Prediction_Change"],
                    }
                )
    frame = pd.DataFrame(output)
    summary = []
    for method in [DISPLAY[x] for x in PAIR]:
        for seed in SEEDS:
            x = frame[(frame.method == method) & (frame.seed == seed)]
            t = x[x.condition == "true"].iloc[0]
            r = x[x.condition == "random"]
            summary.append(
                {
                    "method": method,
                    "seed": seed,
                    "True_Fellow_QWK": t.QWK,
                    "Random_Fellow_QWK_mean": r.QWK.mean(),
                    "Random_Fellow_QWK_std": r.QWK.std(ddof=1),
                    "Delta_QWK": t.QWK - r.QWK.mean(),
                    "True_Fellow_Harm": t.Harm,
                    "Random_Fellow_Harm_mean": r.Harm.mean(),
                    "True_Fellow_Prediction_Change": t.Prediction_Change,
                    "Random_Fellow_Prediction_Change_mean": r.Prediction_Change.mean(),
                }
            )
    frame.to_csv(repo / f"outputs/analysis/{dataset}_random_fellow_replicates.csv", index=False)
    pd.DataFrame(summary).to_csv(
        repo / f"outputs/tables/{dataset}_true_vs_random_fellow.csv", index=False
    )


def bootstrap(repo, dataset, replicates=2000):
    output = []
    rng = np.random.default_rng(2026)
    for paired, single_method in PAIR.items():
        for seed in SEEDS:
            s = load(repo, dataset, single_method, seed)
            b = load(repo, dataset, paired, seed)
            m = s.merge(b, on="sample_id", suffixes=("_s", "_b")).sort_values(
                ["patient_id_s", "target_eye_s"]
            )
            patients = m.patient_id_s.unique()
            draws = defaultdict(list)
            grouped = [
                m[m.patient_id_s == p][["y_true_s", "y_pred_s", "y_pred_b"]].to_numpy()
                for p in patients
            ]
            if any(len(g) != 2 for g in grouped):
                raise RuntimeError("bootstrap requires exactly two target eyes per patient")
            cube = np.stack(grouped, axis=0)
            for _ in range(replicates):
                z = cube[rng.integers(0, len(cube), size=len(cube))].reshape(-1, 3)
                ys = z[:, 0]
                ps = z[:, 1]
                pb = z[:, 2]
                es = np.abs(ps - ys)
                eb = np.abs(pb - ys)
                c = es == 0
                w = ~c
                draws["QWK_difference"].append(
                    cohen_kappa_score(ys, pb, weights="quadratic")
                    - cohen_kappa_score(ys, ps, weights="quadratic")
                )
                draws["MAE_difference"].append(np.mean(eb) - np.mean(es))
                draws["Harm_difference"].append(100 * np.mean(eb > es))
                draws["Preservation_difference"].append(
                    (100 * np.mean(eb[c] == 0) - 100) if c.any() else np.nan
                )
                draws["Recovery_difference"].append(
                    100 * np.mean(eb[w] == 0) if w.any() else np.nan
                )
            for metric, x in draws.items():
                output.append(
                    {
                        "dataset": dataset,
                        "comparison": f"{DISPLAY[paired]} vs {DISPLAY[single_method]}",
                        "seed": seed,
                        "metric": metric,
                        "estimate": np.nanmean(x),
                        "ci_low": np.nanpercentile(x, 2.5),
                        "ci_high": np.nanpercentile(x, 97.5),
                        "bootstrap_unit": "patient",
                        "bootstrap_seed": 2026,
                        "replicates": replicates,
                    }
                )
    pd.DataFrame(output).to_csv(
        repo / f"outputs/analysis/{dataset}_patient_bootstrap_ci.csv", index=False
    )


def final(repo, tables, table42):
    status = list((repo / "outputs/status").glob("*.done"))
    completed = sum(1 for p in status if p.name.startswith(("eyepacs_", "messidor2_")))
    failed = [p.name for p in (repo / "outputs/status").glob("*.failed")]
    audit = json.loads((repo / "outputs/data_audit/data_audit_summary.json").read_text())
    environment = json.loads((repo / "outputs/final/environment.json").read_text())
    summary = {
        "study": "four_paper_full_seed2026",
        "completed_run_markers": completed,
        "expected_runs": 60,
        "failed": failed,
        "seeds": SEEDS,
        "datasets": list(tables),
        "methods": METHODS,
        "excluded": ["ours", "clip_dr", "retfound_d_odr", "diffgeo_aor", "ensemble"],
        "data_audit": audit,
        "environment": environment,
        "table4_1": {d: t.to_dict(orient="records") for d, t in tables.items()},
        "table4_2": {
            d: t.where(pd.notna(t), None).to_dict(orient="records") for d, t in table42.items()
        },
        "artifacts": {
            "gap_analysis": {d: f"outputs/analysis/{d}_gap_analysis.csv" for d in tables},
            "random_fellow": {d: f"outputs/tables/{d}_true_vs_random_fellow.csv" for d in tables},
            "bootstrap_ci": {d: f"outputs/analysis/{d}_patient_bootstrap_ci.csv" for d in tables},
            "predictions": "outputs/predictions/<dataset>/<method>/seed<seed>_test.csv",
            "confusions": "outputs/metrics/<dataset>/<method>/",
        },
    }
    final_dir = repo / "outputs/final"
    final_dir.mkdir(parents=True, exist_ok=True)
    (final_dir / "all_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Four-paper full replication report",
        "",
        f"Completed runs: {completed}/60",
        f"Failed runs: {', '.join(failed) if failed else 'none'}",
        "",
        f"GPU: {environment['gpu']}",
        f"PyTorch: {environment['torch']} (CUDA {environment['cuda_runtime']})",
        f"Source git commit: {environment['git_commit']}",
        "",
        f"EyePACS: {audit['datasets']['eyepacs']['patients']} patients / {audit['datasets']['eyepacs']['eyes']} eyes",
        f"Messidor-2: {audit['datasets']['messidor2']['patients']} patients / {audit['datasets']['messidor2']['eyes']} eyes",
        "",
        "Authoritative patient-level splits were audited before training. EyePACS and Messidor-2 were trained independently. Test sets were used only after checkpoint lock.",
        "",
        "## Configurations",
        "",
        "- Huang: ResNet-50 Single and same-seed frozen-encoder PFF.",
        "- DRGCNN: CAM-EfficientNetV2-M Single and same-seed frozen-encoder BFFN.",
        "- D-ODR: single-eye ViT-B/16, physical batch 128.",
        "- AOR-DR: MICCAI 2025 single-eye ViT-B/16 cross-attention conference configuration.",
        "",
        "## Result artifacts",
        "",
        "- `outputs/tables/`: Table 4-1, Table 4-2, and true-vs-random-fellow controls.",
        "- `outputs/analysis/`: gap analysis, random replicates, and patient-bootstrap confidence intervals.",
        "- `outputs/predictions/`: per-sample raw predictions for every dataset/method/seed.",
        "- `outputs/metrics/`: per-seed and aggregate confusion/class-wise results.",
        "",
        "See `configs/reproduction_notes/` for source-locked settings and disclosed upstream ambiguities.",
    ]
    (final_dir / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    a = p.parse_args()
    repo = a.repo.resolve()
    for directory in [
        "outputs/metrics",
        "outputs/tables",
        "outputs/analysis",
        "outputs/random_fellow",
        "outputs/final",
    ]:
        (repo / directory).mkdir(parents=True, exist_ok=True)
    tables = {}
    t42 = {}
    for dataset in ["eyepacs", "messidor2"]:
        tables[dataset], _ = table41(repo, dataset)
        t42[dataset] = table42(repo, dataset)
        gap_analysis(repo, dataset)
        aggregate_confusions(repo, dataset)
        random_control(repo, dataset)
        bootstrap(repo, dataset)
    with pd.ExcelWriter(repo / "outputs/tables/table4_1_combined.xlsx") as writer:
        for d, t in tables.items():
            t.to_excel(writer, sheet_name=d, index=False)
    with pd.ExcelWriter(repo / "outputs/tables/table4_2_combined.xlsx") as writer:
        for d, t in t42.items():
            t.to_excel(writer, sheet_name=d, index=False)
    final(repo, tables, t42)
    print("FINISHED evaluation analysis tables report", flush=True)


if __name__ == "__main__":
    main()
