import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_torch(cfg: Dict[str, Any]) -> None:
    runtime = cfg.get("runtime", {})
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(runtime.get("tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(runtime.get("tf32", True))
        torch.backends.cudnn.benchmark = bool(runtime.get("cudnn_benchmark", True))


def device_and_amp(precision: str) -> tuple[torch.device, torch.dtype | None]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        return device, None
    if precision == "bf16" and torch.cuda.is_bf16_supported():
        return device, torch.bfloat16
    if precision in {"bf16", "fp16", "float16"}:
        return device, torch.float16
    return device, None


def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + os.linesep)
