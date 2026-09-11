import numpy as np
import torch

from drpair.data.manifest import read_fold_assignment, write_csv
from drpair.evaluation.ordinal import blend
from drpair.gates.features import entropy, fellow_predictions, gate_features
from drpair.gates.fit import fit_gate, predict_gate
from drpair.runtime import run_directory, save_json
from drpair.teachers import counterfactual_teacher, loss_guided_teacher
from drpair.training.predictions import load_pair

from .arms import ARMS, PROPOSED
from .precheck import boundary_opportunities

LEARNED = ["D_global_learned", "E_boundary_task", "F_teacher_a", "G_teacher_b", "global_asymmetry"]


def fit_arm(arm, features, s, f, y, teachers, cfg, seed):
    global_gate = arm in ("D_global_learned", "global_asymmetry")
    x = features[:, -1:] if arm == "global_asymmetry" else features
    task = arm in ("D_global_learned", "E_boundary_task", "global_asymmetry")
    target = teachers["a" if arm == "F_teacher_a" else "b"]["alpha"]
    return fit_gate(
        x, s, f, y, target, cfg, seed, mode="task" if task else "teacher", global_gate=global_gate
    )


def infer_arm(arm, model, features):
    return predict_gate(model, features[:, -1:] if arm == "global_asymmetry" else features)


def generate_teachers(s, f, cfg):
    return {
        "a": loss_guided_teacher(s["q"], f["q"], s["y"], cfg["gate"]["teacher_temperature"]),
        "b": counterfactual_teacher(
            s["q"],
            f["q"],
            s["y"],
            cfg["evaluation"]["threshold"],
            cfg["gate"]["teacher_temperature"],
        ),
    }


def compute_features(s, f, cfg):
    fellow = fellow_predictions(s["q"], s["patient"], s["side"])
    return gate_features(s["q"], f["q"], fellow, cfg["evaluation"]["threshold"])


def run_gate_study(cfg, include_expert_oof=True):
    torch.set_num_threads(4)
    run = run_directory(cfg)
    # All supervised gate fitting uses untouched official validation predictions.
    # Train expert-OOF dumps are preserved for independent teacher diagnostics.
    prepared = {}
    subsets = ("oof", "val") if include_expert_oof else ("val",)
    for subset in subsets:
        s_path = (
            run / "predictions" / ("oof_single.npz" if subset == "oof" else "final_single_val.npz")
        )
        f_path = (
            run / "predictions" / ("oof_paired.npz" if subset == "oof" else "final_paired_val.npz")
        )
        s, f = load_pair(s_path, f_path)
        teachers = generate_teachers(s, f, cfg)
        features, names = compute_features(s, f, cfg)
        for directory in ("teachers", "features", "gates", "metrics"):
            (run / directory).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            run / "teachers" / f"{subset}.npz",
            **{f"{kind}_{k}": v for kind, obj in teachers.items() for k, v in obj.items()},
            patient=s["patient"],
            side=s["side"],
            y=s["y"],
            fold=s["fold"],
        )
        np.savez_compressed(
            run / "features" / f"{subset}.npz",
            x=features,
            patient=s["patient"],
            side=s["side"],
            feature_names=np.array(names),
        )
        comparison = []
        for b in range(4):
            a = teachers["a"]["alpha"][:, b] > 0
            c = teachers["b"]["alpha"][:, b] > 0
            comparison.append(
                {
                    "boundary": b,
                    "a_active": int(a.sum()),
                    "b_active": int(c.sum()),
                    "disagreement_rate": float(np.mean(a != c)),
                    "local_cf_benefit": int((teachers["b"]["utility"][:, b] > 0).sum()),
                }
            )
        write_csv(run / "metrics" / f"teacher_a_vs_b_{subset}.csv", comparison)
        if subset == "oof":
            opportunity_rows, _, _, _ = boundary_opportunities(s, f, cfg["evaluation"]["threshold"])
            write_csv(run / "metrics" / "oof_boundary_opportunities.csv", opportunity_rows)
        prepared[subset] = (s, f, teachers, features, names)
    # Gate fitting reads the official validation dump by name. Inheriting whatever
    # the loop happened to leave behind would silently refit gates on train OOF.
    s, f, teachers, features, names = prepared["val"]
    if not np.all(s["split"] == "val"):
        raise ValueError("Gate fitting source is not the official validation split")
    validation_rows = [
        {"patient": str(p), "side": str(side)} for p, side in zip(s["patient"], s["side"])
    ]
    folds = read_fold_assignment(
        cfg["data"]["fold_manifest"],
        validation_rows,
        "val",
        cfg["experiment"]["seed"],
        cfg["gate"]["crossfit_folds"],
    )
    alpha_oof = {}
    crossfit_identity = np.column_stack([s["patient"], s["side"], folds.astype(str)])
    np.savez_compressed(run / "features" / "gate_crossfit_identity.npz", rows=crossfit_identity)
    for arm in LEARNED:
        alpha = np.full_like(s["q"], np.nan)
        for fold in range(cfg["gate"]["crossfit_folds"]):
            train, held = folds != fold, folds == fold
            if set(s["patient"][train]) & set(s["patient"][held]):
                raise ValueError("Gate cross-fitting patient leakage")
            targets = {
                kind: {key: value[train] for key, value in obj.items()}
                for kind, obj in teachers.items()
            }
            model, history = fit_arm(
                arm,
                features[train],
                s["q"][train],
                f["q"][train],
                s["y"][train],
                targets,
                cfg["gate"],
                cfg["experiment"]["seed"],
            )
            alpha[held] = infer_arm(arm, model, features[held])
            torch.save(model.state_dict(), run / "gates" / f"{arm}_crossfit{fold}.pt")
            write_csv(run / "logs" / f"{arm}_crossfit{fold}.csv", history)
        alpha_oof[arm] = alpha
        model, history = fit_arm(
            arm,
            features,
            s["q"],
            f["q"],
            s["y"],
            teachers,
            cfg["gate"],
            cfg["experiment"]["seed"],
        )
        torch.save(model.state_dict(), run / "gates" / f"{arm}_final.pt")
        write_csv(run / "logs" / f"{arm}_final.csv", history)
        # Test is touched only after all model choices for this arm are fixed.
        ts, tf = load_pair(
            run / "predictions" / "final_single_test.npz",
            run / "predictions" / "final_paired_test.npz",
        )
        tx, _ = compute_features(ts, tf, cfg)
        test_alpha = infer_arm(arm, model, tx)
        save_arm(run, arm, "test", ts, tf, test_alpha)
        save_arm(run, arm, "val_crossfit", s, f, alpha, folds)
    for split, s_name, f_name in [("val_crossfit", "val", "val"), ("test", "test", "test")]:
        ps, pf = load_pair(
            run / "predictions" / f"final_single_{s_name}.npz",
            run / "predictions" / f"final_paired_{f_name}.npz",
        )
        alpha = entropy(pf["q"]) / (entropy(ps["q"]) + entropy(pf["q"]) + 1e-8)
        for arm, a in [
            ("A_single", np.ones_like(alpha)),
            ("B_paired", np.zeros_like(alpha)),
            ("C_fixed50", np.full_like(alpha, 0.5)),
            ("entropy_reliability", alpha),
        ]:
            save_arm(run, arm, split, ps, pf, a, folds if split == "val_crossfit" else None)
    save_json(
        run / "gate_protocol.json",
        {
            "proposed": PROPOSED,
            "arms": ARMS,
            "fitting_source": "official_validation_with_patient_5fold_crossfitting",
            "expert_oof_source": (
                "official_train_with_nested_patient_selection_and_refit"
                if include_expert_oof
                else "not_repeated_in_phase2_replication"
            ),
            "test_labels_used_for_fitting": False,
            "threshold": cfg["evaluation"]["threshold"],
            "feature_names": names,
            "complete": True,
        },
    )


def save_arm(run, arm, split, s, f, alpha, folds=None):
    q, raw = blend(s["q"], f["q"], alpha)
    np.savez_compressed(
        run / "predictions" / f"{arm}_{split}.npz",
        q=q,
        raw=raw,
        alpha=alpha,
        **{key: s[key] for key in ("patient", "side", "y", "fellow_y", "seed")},
        fold=s["fold"] if folds is None else folds,
        split=np.array([split] * len(q)),
    )
