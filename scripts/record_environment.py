"""Record installed versions and capability checks without running a model."""

import importlib.metadata
import platform
from pathlib import Path

import torch

from drpair.runtime import save_json


def main():
    repo = Path(__file__).resolve().parents[1]
    names = ["torch", "torchvision", "timm", "numpy", "scipy", "scikit-learn", "pillow", "dvc"]
    save_json(
        repo / "experimental_results" / "repository_validation" / "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in names},
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "bf16_supported": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            "experiment_started": False,
            "verification": "Imports and capability queries only; no image-model execution",
        },
    )


if __name__ == "__main__":
    main()
