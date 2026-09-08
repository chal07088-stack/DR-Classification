import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_training import TinyData, TinyEncoder
from torch.utils.data import DataLoader

from drpair.data.manifest import patient_folds
from drpair.evaluation.metrics import expert_boundary_diagnostics
from drpair.experiments.precheck import boundary_opportunities
from drpair.models import SingleOrdinal
from drpair.runtime import load_config, run_directory
from drpair.training.engine import fit


def test_rare_stratum_fallback_is_visible():
    rows = [{"patient": str(i), "label": 4 if i == 0 else 0} for i in range(6)]
    with pytest.warns(RuntimeWarning) as caught:
        folds = patient_folds(rows, folds=3)
    assert set(folds) == {0, 1, 2}
    assert any("unstratified" in str(w.message) for w in caught)


def test_silent_boundary_does_not_mean_no_ranking_signal():
    y = np.array([0, 0, 4, 4])
    q = np.array(
        [
            [0.4, 0.3, 0.2, 0.1],
            [0.4, 0.3, 0.2, 0.1],
            [0.49, 0.48, 0.47, 0.45],
            [0.49, 0.48, 0.47, 0.45],
        ]
    )
    d = expert_boundary_diagnostics(y, q)
    assert d[3]["positive"] == 2 and d[3]["predicted_positive"] == 0
    assert d[3]["recall"] == 0 and d[3]["auc"] == 1
    assert d[3]["silent_at_threshold"]
    s = {"y": y, "q": q, "patient": np.array([1, 1, 2, 2])}
    rows, review, _, _ = boundary_opportunities(s, {"q": q})
    assert review and len(rows) == 4
    f = {"q": q.copy()}
    f["q"][2:] = [0.9, 0.8, 0.7, 0.6]
    _, review, _, _ = boundary_opportunities(s, f)
    assert not review  # A single active expert still offers rollback/mixing opportunities.


def test_run_record_rejects_changed_config(tmp_path):
    cfg = load_config(Path(__file__).parents[1] / "configs" / "phase1.toml")
    cfg["data"]["results"] = str(tmp_path)
    run_directory(cfg)
    cfg["training"]["backbone_lr"] *= 2
    with pytest.raises(ValueError, match="Config changed"):
        run_directory(cfg)


def test_checkpoint_rejects_changed_config_or_unmarked_legacy(tmp_path):
    cfg = {
        "training": {
            "epochs": 1,
            "effective_batch": 4,
            "backbone_lr": 0.001,
            "head_lr": 0.001,
            "weight_decay": 0.0001,
            "ema_decay": 0.9,
            "grad_clip": 1.0,
        }
    }
    path = tmp_path / "model.pt"
    loader = DataLoader(TinyData(), batch_size=2)

    def train(config):
        return fit(
            SingleOrdinal(TinyEncoder()),
            loader,
            config,
            torch.device("cpu"),
            None,
            False,
            path,
            tmp_path / "log.jsonl",
        )

    train(cfg)
    train(cfg)  # A completed identical checkpoint can be reused.
    changed = copy.deepcopy(cfg)
    changed["training"]["head_lr"] *= 2
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        train(changed)
    saved = torch.load(path, weights_only=True)
    del saved["fingerprint"]
    torch.save(saved, path)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        train(cfg)
    log = json.loads((tmp_path / "log.jsonl").read_text().splitlines()[-1])
    assert log["data_wait_seconds"] >= 0 and log["target_eyes_per_second"] > 0


def test_each_selection_epoch_records_boundary_events(tmp_path):
    class WithRows(TinyData):
        def __init__(self):
            self.rows = [{"patient": str(i), "label": i} for i in range(5)]

    cfg = {
        "training": {
            "epochs": 1,
            "effective_batch": 4,
            "backbone_lr": 0.001,
            "head_lr": 0.001,
            "weight_decay": 0.0001,
            "ema_decay": 0.9,
            "grad_clip": 1.0,
        }
    }
    loader = DataLoader(WithRows(), batch_size=2)
    fit(
        SingleOrdinal(TinyEncoder()),
        loader,
        cfg,
        torch.device("cpu"),
        None,
        False,
        tmp_path / "selection.pt",
        tmp_path / "selection.jsonl",
        validation_loader=loader,
    )
    report = json.loads((tmp_path / "selection.boundaries.json").read_text())
    assert report["epoch"] == 1 and report["boundaries"][3]["positive"] == 1
    assert "predicted_positive" in report["boundaries"][3]


@pytest.mark.parametrize("silent", [False, True])
def test_initial_pair_selections_precede_refits_without_extra_fits(tmp_path, monkeypatch, silent):
    from types import SimpleNamespace

    from drpair.data.manifest import write_csv
    from drpair.experiments import experts

    cfg = load_config(Path(__file__).parents[1] / "configs" / "phase1.toml")
    cfg["training"]["epochs"] = 1
    cfg["diagnostics"]["stop_if_both_b3_silent"] = True
    cfg["data"] = {
        "root": str(tmp_path),
        "manifest": str(tmp_path / "manifest.csv"),
        "fold_manifest": str(tmp_path / "folds.csv"),
        "results": str(tmp_path / "results"),
    }
    rows = []
    for p in range(60):
        for side, other in [("left", "right"), ("right", "left")]:
            rows.append(
                {
                    "patient": str(p),
                    "side": side,
                    "label": p % 5,
                    "fellow_label": p % 5,
                    "split": "train",
                    "path": f"{p}_{side}.jpeg",
                    "fellow_path": f"{p}_{other}.jpeg",
                }
            )
    write_csv(cfg["data"]["manifest"], rows)
    fold_ids = patient_folds(rows)
    write_csv(
        cfg["data"]["fold_manifest"],
        [
            {
                "patient": r["patient"],
                "side": r["side"],
                "split": "train",
                "seed": 42,
                "fold": int(f),
            }
            for r, f in zip(rows, fold_ids)
        ],
    )
    calls, cached = [], set()

    def fake_fit(model, loader, config, device, dtype, paired, checkpoint, log, **kwargs):
        if checkpoint.name not in cached:
            calls.append(checkpoint.stem)
            cached.add(checkpoint.name)
        return model, 1

    def fake_loader(records, *args, **kwargs):
        return SimpleNamespace(dataset=SimpleNamespace(rows=records))

    def fake_predict(model, loader, *args):
        y = np.array([r["label"] for r in loader.dataset.rows])
        if silent:
            return np.full((len(y), 4), 0.25)
        return 0.05 + 0.9 * (y[:, None] > np.arange(4))

    monkeypatch.setattr(experts, "setup_device", lambda cfg: (torch.device("cpu"), None))
    monkeypatch.setattr(experts, "validate_manifest", lambda *a, **kw: None)
    monkeypatch.setattr(experts, "build_expert", lambda *a: TinyEncoder())
    monkeypatch.setattr(experts, "make_loader", fake_loader)
    monkeypatch.setattr(experts, "fit", fake_fit)
    monkeypatch.setattr(experts, "predict", fake_predict)
    if silent:
        from drpair.experiments.precheck import BoundaryReviewRequired

        with pytest.raises(BoundaryReviewRequired):
            experts.run_experts(cfg, ["final_single", "final_paired"])
    else:
        experts.run_experts(cfg, ["final_single", "final_paired"])
    expected = ["final_single_selection", "final_paired_selection"]
    assert calls == expected + ([] if silent else ["final_single", "final_paired"])
    report = json.loads(
        (run_directory(cfg) / "diagnostics" / "initial_expert_check.json").read_text()
    )
    assert report["status"] == ("needs_review" if silent else "continue")
    assert report["additional_training_fits"] == 0
