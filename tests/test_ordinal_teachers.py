import numpy as np
import torch
from scipy.optimize import isotonic_regression

from drpair.evaluation.ordinal import blend, decode, project, violations
from drpair.gates.network import project_torch
from drpair.teachers.targets import counterfactual_teacher, discrete_oracle, joint_effect


def test_projection_matches_independent_scipy_solver():
    raw = np.random.default_rng(5).random((50, 4))
    expected = np.array([isotonic_regression(r, increasing=False).x for r in raw])
    np.testing.assert_allclose(project(raw), expected, atol=1e-7)
    x = torch.tensor(raw, requires_grad=True)
    actual = project_torch(x)
    np.testing.assert_allclose(actual.detach(), expected, atol=1e-7)
    actual.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_ordered_experts_can_produce_disordered_boundary_blend():
    s = np.array([[0.95, 0.9, 0.8, 0.7]])
    f = np.array([[0.6, 0.5, 0.4, 0.3]])
    q, raw = blend(s, f, np.array([[0, 1, 0, 1]]))
    assert violations(raw).all()
    assert not violations(q).any()
    np.testing.assert_allclose(blend(s, f, np.zeros((1, 4)))[0], f)
    np.testing.assert_allclose(blend(s, f, np.ones((1, 4)))[0], s)


def test_teacher_b_is_exact_one_boundary_intervention():
    rng = np.random.default_rng(42)
    s, f = project(rng.random((90, 4))), project(rng.random((90, 4)))
    y = rng.integers(0, 5, 90)
    teacher = counterfactual_teacher(s, f, y)
    base = (decode(f) - y) ** 2
    for b in range(4):
        candidate = f.copy()
        candidate[:, b] = s[:, b]
        np.testing.assert_array_equal(teacher["utility"][:, b], base - (decode(candidate) - y) ** 2)
    assert set(teacher) == {"utility", "alpha"}
    assert (teacher["alpha"][teacher["utility"] <= 0] == 0).all()


def test_joint_effect_is_evaluation_and_can_differ_from_local_sum():
    rng = np.random.default_rng(7)
    s, f = project(rng.random((100, 4))), project(rng.random((100, 4)))
    y = rng.integers(0, 5, 100)
    teacher = counterfactual_teacher(s, f, y)
    before = teacher["utility"].copy()
    joint = joint_effect(s, f, y, np.ones_like(s))
    assert np.any(before.sum(1) != joint["utility"])
    oracle = discrete_oracle(s, f, y)
    assert (oracle["utility"] >= 0).all()
    np.testing.assert_array_equal(teacher["utility"], before)
