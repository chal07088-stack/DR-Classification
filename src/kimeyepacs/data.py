import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


EYE_RE = re.compile(r"^(?P<patient>\d+)_(?P<side>left|right)\.jpeg$", re.I)
BOUNDARIES = np.arange(1, 5)


@dataclass(frozen=True)
class EyeItem:
    patient: str
    side: str
    path: str
    label: int
    fellow_path: str
    fellow_label: int
    split: str


def ordinal_target(label: int) -> np.ndarray:
    return (int(label) >= BOUNDARIES).astype(np.float32)


def scan_split(data_root: str | Path, split: str) -> List[EyeItem]:
    root = Path(data_root) / split
    by_patient: dict[str, dict[str, tuple[str, int]]] = {}
    for grade_dir in sorted(root.iterdir()):
        if not grade_dir.is_dir():
            continue
        grade = int(grade_dir.name)
        for path in grade_dir.glob("*.jpeg"):
            m = EYE_RE.match(path.name)
            if not m:
                continue
            by_patient.setdefault(m.group("patient"), {})[m.group("side").lower()] = (
                str(path),
                grade,
            )

    out: list[EyeItem] = []
    for patient, eyes in sorted(by_patient.items(), key=lambda kv: int(kv[0])):
        if "left" not in eyes or "right" not in eyes:
            continue
        left_path, left_label = eyes["left"]
        right_path, right_label = eyes["right"]
        out.append(EyeItem(patient, "left", left_path, left_label, right_path, right_label, split))
        out.append(EyeItem(patient, "right", right_path, right_label, left_path, left_label, split))
    return out


def scan_all(data_root: str | Path) -> dict[str, List[EyeItem]]:
    return {split: scan_split(data_root, split) for split in ("train", "val", "test")}


def limit_patients(items: List[EyeItem], max_patients: int | None, seed: int) -> List[EyeItem]:
    if not max_patients:
        return items
    patients = sorted({x.patient for x in items}, key=int)
    rng = np.random.default_rng(seed)
    chosen = set(rng.choice(patients, size=min(max_patients, len(patients)), replace=False))
    return [x for x in items if x.patient in chosen]


def class_counts(items: Iterable[EyeItem]) -> dict[int, int]:
    counts = {i: 0 for i in range(5)}
    for item in items:
        counts[item.label] += 1
    return counts


def boundary_pos_weight(items: List[EyeItem], max_weight: float = 20.0) -> torch.Tensor:
    labels = np.asarray([x.label for x in items])
    y = (labels[:, None] >= BOUNDARIES[None, :]).astype(np.float32)
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    w = neg / np.maximum(pos, 1.0)
    return torch.tensor(np.clip(w, 1.0, max_weight), dtype=torch.float32)


def _load_image(path: str, image_size: int, train: bool, hflip: bool, color_mode: str) -> torch.Tensor:
    with Image.open(path) as im:
        if color_mode == "grayscale_reference":
            im = im.convert("L").resize((image_size, image_size), Image.Resampling.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0
            if train and hflip and np.random.rand() < 0.5:
                arr = arr[:, ::-1].copy()
            x = torch.from_numpy(arr)[None].repeat(3, 1, 1)
            return (x - 0.5) / 0.25

        im = im.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        if train and hflip and np.random.rand() < 0.5:
            arr = arr[:, ::-1].copy()
        x = torch.from_numpy(arr).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        return (x - mean) / std


class SingleEyeDataset(Dataset):
    def __init__(self, items: List[EyeItem], image_size: int, train: bool, hflip: bool, color_mode: str):
        self.items = items
        self.image_size = image_size
        self.train = train
        self.hflip = hflip
        self.color_mode = color_mode

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        x = _load_image(item.path, self.image_size, self.train, self.hflip, self.color_mode)
        y = torch.from_numpy(ordinal_target(item.label))
        return x, y, torch.tensor(item.label, dtype=torch.long), torch.tensor(idx, dtype=torch.long)


class PairedEyeDataset(Dataset):
    def __init__(self, items: List[EyeItem], image_size: int, train: bool, hflip: bool, color_mode: str):
        self.items = items
        self.image_size = image_size
        self.train = train
        self.hflip = hflip
        self.color_mode = color_mode

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        x = _load_image(item.path, self.image_size, self.train, self.hflip, self.color_mode)
        fellow = _load_image(item.fellow_path, self.image_size, self.train, self.hflip, self.color_mode)
        y = torch.from_numpy(ordinal_target(item.label))
        return x, fellow, y, torch.tensor(item.label, dtype=torch.long), torch.tensor(idx, dtype=torch.long)
