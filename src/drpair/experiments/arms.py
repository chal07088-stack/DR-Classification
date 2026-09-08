ARMS = {
    "A_single": "Target-eye single ordinal expert",
    "B_paired": "End-to-end paired ordinal expert",
    "C_fixed50": "Fixed 50:50 single/paired probability blend",
    "D_global_learned": "One learned rollback weight per target eye",
    "E_boundary_task": "Boundary gate optimized by final ordinal loss; experts fixed",
    "F_teacher_a": "Boundary gate with Loss-Guided supervision",
    "G_teacher_b": "Boundary gate with one-boundary Counterfactual Ordinal supervision (proposed)",
    "global_asymmetry": "Predicted-asymmetry-only learned scalar gate (internal baseline)",
    "entropy_reliability": "Entropy-weighted probability blend (generic internal baseline)",
}
PROPOSED = "G_teacher_b"
