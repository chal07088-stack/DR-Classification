from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import numpy
import sklearn
import timm
import torch
import torchvision


def command(*args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.SubprocessError) as error:
        return f"unavailable: {error}"


root = Path(__file__).resolve().parents[1]
git_commit = command("git", "-C", str(root), "rev-parse", "HEAD")
if git_commit.startswith("unavailable:") and (root / ".source_git_commit").is_file():
    git_commit = (root / ".source_git_commit").read_text(encoding="utf-8").strip()
value = {
    "platform": platform.platform(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "timm": timm.__version__,
    "numpy": numpy.__version__,
    "scikit_learn": sklearn.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_runtime": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "nvidia_smi": command(
        "nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"
    ),
    "git_commit": git_commit,
}
path = root / "outputs/final/environment.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(value, indent=2), encoding="utf-8")
print(json.dumps(value, indent=2))
