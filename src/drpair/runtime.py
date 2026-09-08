"""Portable configuration, provenance, and deterministic runtime helpers."""

import hashlib
import json
import os
import random
import subprocess
import tomllib
from pathlib import Path

import numpy as np
import torch


def load_config(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        cfg = tomllib.load(stream)
    root = path.parent.parent
    for key in ("root", "manifest", "fold_manifest", "results"):
        override = os.environ.get("DRPAIR_" + key.upper())
        p = Path(override or cfg["data"][key])
        cfg["data"][key] = str(p if p.is_absolute() else root / p)
    if cfg["experiment"]["phase"] != 1 or cfg["experiment"]["seed"] != 42:
        raise ValueError("This runner is restricted to PHASE 1 / seed 42.")
    if cfg["experiment"]["folds"] != 5 or cfg["gate"]["crossfit_folds"] != 5:
        raise ValueError("PHASE 1 requires five patient folds.")
    for key in ("single_batch", "paired_batch"):
        if cfg["training"]["effective_batch"] % cfg["training"][key]:
            raise ValueError("Physical batch must divide effective_batch exactly.")
    if cfg["model"]["head"] != "ordered_coral":
        raise ValueError("PHASE 1 head is ordered_coral.")
    if cfg["runtime"]["precision"] != "bf16":
        raise ValueError("PHASE 1 uses BF16 first, with automatic FP16 fallback.")
    return cfg


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def seed_everything(seed, deterministic=True):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)


def seed_worker(_worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def setup_device(cfg, cpu=False):
    runtime = cfg["runtime"]
    seed_everything(cfg["experiment"]["seed"], runtime["deterministic"])
    if not cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. CPU is allowed only for explicit verification.")
    device = torch.device("cpu" if cpu else "cuda")
    torch.backends.cudnn.benchmark = runtime["cudnn_benchmark"]
    torch.backends.cudnn.allow_tf32 = runtime["tf32"]
    torch.backends.cuda.matmul.allow_tf32 = runtime["tf32"]
    dtype = None
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return device, dtype


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_directory(cfg):
    path = Path(cfg["data"]["results"]) / cfg["experiment"]["run_name"]
    portable_cfg = {k: v for k, v in cfg.items() if k != "data"}
    signature = hashlib.sha256(json.dumps(portable_cfg, sort_keys=True).encode()).hexdigest()
    source_root = Path(__file__).parent
    code_hash = hashlib.sha256()
    for source in sorted(source_root.rglob("*.py")):
        code_hash.update(source.relative_to(source_root).as_posix().encode())
        code_hash.update(source.read_text(encoding="utf-8").encode())
    provenance = {
        "source_sha256": code_hash.hexdigest(),
        "manifest_sha256": file_hash(cfg["data"]["manifest"]),
        "fold_manifest_sha256": file_hash(cfg["data"]["fold_manifest"]),
    }
    record = path / "run.json"
    if record.exists():
        old = json.loads(record.read_text(encoding="utf-8"))
        if old["config_sha256"] != signature or any(old.get(k) != v for k, v in provenance.items()):
            raise ValueError("Config changed. Use a new run_name; old artifacts cannot be reused.")
    else:
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
        save_json(
            record,
            {
                "config_sha256": signature,
                "config": cfg,
                "git_commit": git.stdout.strip(),
                "torch": torch.__version__,
                **provenance,
            },
        )
    return path
