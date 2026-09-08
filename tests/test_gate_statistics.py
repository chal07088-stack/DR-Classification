import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score

from drpair.evaluation.ordinal import project
from drpair.evaluation.statistics import adjust_pvalues, paired_bootstrap_delta, qwk_from_confusion
from drpair.gates.features import gate_features
from drpair.gates.fit import fit_gate, predict_gate
from drpair.teachers import counterfactual_teacher


def test_qwk_bootstrap_and_multiple_testing():
    y = np.tile(np.arange(5), 10)
    pred = np.clip(y + np.tile([-1, 0, 1, 0, 0], 10), 0, 4)
    cm = np.zeros((5, 5))
    np.add.at(cm, (y, pred), 1)
    np.testing.assert_allclose(
        qwk_from_confusion(cm), cohen_kappa_score(y, pred, weights="quadratic")
    )
    result = paired_bootstrap_delta(y, pred, pred, np.repeat(np.arange(25), 2), 20)
    assert result["delta"] == result["ci_low"] == result["ci_high"] == 0
    np.testing.assert_allclose(adjust_pvalues([0.01, 0.04, 0.03]), [0.03, 0.04, 0.04])
    np.testing.assert_allclose(adjust_pvalues([0.01, 0.04, 0.03], "holm"), [0.03, 0.06, 0.06])


def test_gate_training_inference_and_true_global_alpha():
    torch.set_num_threads(2)
    rng = np.random.default_rng(42)
    s, f = project(rng.random((30, 4))), project(rng.random((30, 4)))
    y = rng.integers(0, 5, 30)
    x, _ = gate_features(s, f, s[::-1])
    teacher = counterfactual_teacher(s, f, y)
    cfg = {
        "hidden": 4,
        "epochs": 2,
        "batch": 8,
        "lr": 0.001,
        "weight_decay": 0.001,
        "intervention_penalty": 0.01,
        "rare_boundary_prior": 2,
    }
    for mode, global_gate in [("teacher", False), ("task", False), ("task", True)]:
        model, history = fit_gate(x, s, f, y, teacher["alpha"], cfg, 42, mode, global_gate)
        alpha = predict_gate(model, x)
        assert np.isfinite(alpha).all() and ((alpha >= 0) & (alpha <= 1)).all()
        assert len(history) == 2
        if global_gate:
            np.testing.assert_allclose(alpha, np.repeat(alpha[:, :1], 4, axis=1))
