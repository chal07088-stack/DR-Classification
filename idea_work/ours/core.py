from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageChops
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ConvNeXt_Small_Weights, convnext_small
from torchvision.transforms import InterpolationMode

BOUNDARY_NAMES = ("01", "12", "23", "34")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str] | None = None) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in materialized for key in row))
    if not fields:
        raise ValueError(f"Cannot write fieldless CSV: {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)
    tmp.replace(path)


def metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    pred = np.asarray(pred, dtype=int)
    return {
        "qwk": float(cohen_kappa_score(y_true, pred, weights="quadratic")),
        "mae": float(mean_absolute_error(y_true, pred)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(
            f1_score(y_true, pred, labels=list(range(5)), average="macro", zero_division=0)
        ),
        "n": int(len(y_true)),
    }


def metric_key(value: dict[str, float]) -> tuple[float, float, float]:
    return (value["qwk"], value["macro_f1"], -value["mae"])


def crop_black(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    bbox = ImageChops.difference(image, Image.new("RGB", image.size, (0, 0, 0))).getbbox()
    return image.crop(bbox) if bbox else image


class CropBlack:
    def __call__(self, image: Image.Image) -> Image.Image:
        return crop_black(image)


def image_transform(train: bool, size: int = 512):
    steps: list[object] = [CropBlack()]
    if train:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomResizedCrop(
                    size,
                    scale=(0.87, 1.15),
                    ratio=(0.7, 1.3),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomRotation((-180, 180), fill=0),
                transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), fill=0),
                transforms.Resize((size, size), interpolation=InterpolationMode.BILINEAR),
            ]
        )
    else:
        steps.append(transforms.Resize((size, size), interpolation=InterpolationMode.BILINEAR))
    steps.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(steps)


def load_rows(config: dict, dataset: str) -> list[dict[str, str]]:
    baseline = Path(config["baseline_root"])
    manifest = baseline / "outputs/data_audit" / (
        "eyepacs_directional_manifest.csv"
        if dataset == "eyepacs"
        else "messidor2_directional_manifest.csv"
    )
    rows = read_csv(manifest)
    if not rows:
        raise RuntimeError(f"Empty manifest: {manifest}")
    if dataset == "eyepacs":
        for row in rows:
            row["run_split"] = row["split"]
    else:
        patients = sorted({row["patient_id"] for row in rows if row["split"] == "train"})
        ranked = sorted(
            patients,
            key=lambda patient: hashlib.sha256(f"2026:{patient}".encode()).hexdigest(),
        )
        val_ids = set(ranked[: round(len(ranked) * 0.2)])
        for row in rows:
            if row["split"] == "train":
                row["run_split"] = "inner_val" if row["patient_id"] in val_ids else "train"
            else:
                row["run_split"] = row["split"]
    validate_splits(rows, dataset)
    return rows


def validate_splits(rows: list[dict[str, str]], dataset: str) -> None:
    split_ids: dict[str, set[str]] = {}
    for row in rows:
        split_ids.setdefault(row["run_split"], set()).add(row["patient_id"])
        if row["target_eye"] == row["fellow_eye"]:
            raise AssertionError(f"Same target/fellow eye: {row['sample_id']}")
        if int(row["y_true"]) not in range(5) or int(row["fellow_y_true"]) not in range(5):
            raise AssertionError(f"Invalid grade: {row['sample_id']}")
    keys = list(split_ids)
    for left_index, left in enumerate(keys):
        for right in keys[left_index + 1 :]:
            overlap = split_ids[left] & split_ids[right]
            if overlap:
                raise AssertionError(f"{dataset} patient leakage {left}/{right}: {len(overlap)}")
    expected = (
        {"train": 12293, "val": 2635, "test": 2635}
        if dataset == "eyepacs"
        else {"train": 557, "inner_val": 139, "test": 174}
    )
    actual = {key: len(value) for key, value in split_ids.items()}
    if actual != expected:
        raise AssertionError(f"{dataset} split counts {actual} != {expected}")
    counts = Counter(row["sample_id"] for row in rows)
    if any(value != 1 for value in counts.values()):
        raise AssertionError("sample_id is not unique")


def image_root(config: dict, dataset: str) -> Path:
    root = Path(config["baseline_root"])
    return root / "data" / ("eyepacs" if dataset == "eyepacs" else "messidor2/IMAGES")


class DirectionalImageDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        root: Path,
        transform,
        include_fellow: bool = False,
    ):
        self.rows = rows
        self.root = root
        self.transform = transform
        self.include_fellow = include_fellow

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(self.root / row["target_image_path"]) as image:
            target = self.transform(image.convert("RGB"))
        output: dict[str, object] = {
            "target": target,
            "label": torch.tensor(int(row["y_true"]), dtype=torch.long),
            "index": index,
        }
        if self.include_fellow:
            with Image.open(self.root / row["fellow_image_path"]) as image:
                output["fellow"] = self.transform(image.convert("RGB"))
        return output


class StageAModel(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        base = convnext_small(weights=weights)
        self.features = base.features
        self.avgpool = base.avgpool
        self.norm = base.classifier[0]
        self.classifier = nn.Linear(768, 5)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.features(images)
        values = self.avgpool(values)
        features = self.norm(values).flatten(1)
        return features, self.classifier(features)


def boundaries(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != 5:
        raise ValueError(f"Expected [B,5] logits, got {tuple(logits.shape)}")
    return logits[:, 1:] - logits[:, :-1]


def reconstruct_logits(state: torch.Tensor) -> torch.Tensor:
    if state.ndim != 2 or state.shape[1] != 4:
        raise ValueError(f"Expected [B,4] boundaries, got {tuple(state.shape)}")
    return torch.cat([torch.zeros_like(state[:, :1]), torch.cumsum(state, dim=1)], dim=1)


class StageBGate(nn.Module):
    def __init__(self, mode: str, adapter_dim: int = 256, hidden: int = 128):
        super().__init__()
        if mode not in {"bounded", "global", "unbounded"}:
            raise ValueError(mode)
        self.mode = mode
        self.adapter = nn.Sequential(nn.Linear(768, adapter_dim), nn.GELU(), nn.LayerNorm(adapter_dim))
        output = 1 if mode == "global" else 4
        self.gate = nn.Sequential(nn.Linear(adapter_dim * 2 + 10, hidden), nn.GELU(), nn.Linear(hidden, output))

    def coefficients(
        self,
        h_target: torch.Tensor,
        h_fellow: torch.Tensor,
        p_target: torch.Tensor,
        p_fellow: torch.Tensor,
    ) -> torch.Tensor:
        target = self.adapter(h_target)
        fellow = self.adapter(h_fellow)
        relation = torch.cat([target - fellow, target * fellow, p_target, p_fellow], dim=1)
        raw = self.gate(relation)
        if self.mode != "unbounded":
            raw = torch.sigmoid(raw)
        if self.mode == "global":
            raw = raw.repeat(1, 4)
        return raw

    def forward(
        self,
        h_target: torch.Tensor,
        h_fellow: torch.Tensor,
        p_target: torch.Tensor,
        p_fellow: torch.Tensor,
        a_target: torch.Tensor,
        a_fellow: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coefficient = self.coefficients(h_target, h_fellow, p_target, p_fellow)
        corrected = a_target + coefficient * (a_fellow - a_target)
        return reconstruct_logits(corrected), coefficient


def rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
