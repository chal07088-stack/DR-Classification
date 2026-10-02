"""Standalone Huang preflight; safe to run in the paper-specific Python environment."""

from pathlib import Path

import torch

from drpair.benchmark.quick_models import HuangPaired, HuangSingle
from drpair.full_baselines.runtime import image_root, method_transform


def main() -> None:
    paired = HuangPaired(HuangSingle(pretrained=False).encoder)
    assert not any(parameter.requires_grad for parameter in paired.encoder.parameters())
    paired.train()
    assert paired.encoder.training
    assert any(
        module.training
        for module in paired.encoder.modules()
        if isinstance(module, torch.nn.BatchNorm2d)
    )
    paired.eval()
    assert not paired.encoder.training

    repo = Path("/example/repo")
    expected = repo / "data/cache/drgcnn/eyepacs"
    for method in ("huang_single", "huang_pff", "drgcnn_single"):
        assert image_root(repo, "eyepacs", method) == expected
    transform = method_transform("huang_single", False, [0.5] * 3, [0.25] * 3)
    assert not any(step.__class__.__name__ == "CropBlack" for step in transform.transforms)
    print("HUANG_ALIGNMENT_PASS")


if __name__ == "__main__":
    main()
