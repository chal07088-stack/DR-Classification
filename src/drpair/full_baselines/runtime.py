from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageChops
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

METHODS = ("huang_single", "huang_pff", "drgcnn_single", "drgcnn_bffn", "d_odr", "aor_dr")
PAIRED = {"huang_pff", "drgcnn_bffn"}
HIGH_RES = {"huang_single", "huang_pff", "drgcnn_single", "drgcnn_bffn"}


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_seed(worker_id: int) -> None:
    value = torch.initial_seed() % (2**32)
    np.random.seed(value)
    random.seed(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def crop_black(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    background = Image.new("RGB", image.size, (0, 0, 0))
    bbox = ImageChops.difference(image, background).getbbox()
    return image.crop(bbox) if bbox else image


class CropBlack:
    def __call__(self, image: Image.Image) -> Image.Image:
        return crop_black(image)


def method_transform(method: str, train: bool, mean: list[float], std: list[float]):
    size = 512 if method in HIGH_RES else 224
    items: list[object] = [CropBlack()]
    if train and method in HIGH_RES:
        crop = transforms.RandomResizedCrop(
            size,
            scale=(0.87, 1.15),
            ratio=(0.7, 1.3),
            interpolation=InterpolationMode.BILINEAR,
        )
        colour = transforms.ColorJitter(brightness=0.2, contrast=0.2)
        items.extend([transforms.RandomHorizontalFlip(0.5), transforms.RandomVerticalFlip(0.5)])
        # The released fusion scripts place crop before colour jitter, while
        # the released single-eye config lists colour jitter before crop.
        items.extend([crop, colour] if method in PAIRED else [colour, crop])
        items.extend(
            [
                transforms.RandomRotation(degrees=(-180, 180), fill=0),
                transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), fill=0),
                transforms.Resize((size, size), interpolation=InterpolationMode.BILINEAR),
            ]
        )
    else:
        items.append(transforms.Resize((size, size), interpolation=InterpolationMode.BILINEAR))
        if train:
            items.append(transforms.RandomHorizontalFlip(0.5))
    items.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(items)


def load_manifest(repo: Path, dataset: str) -> list[dict[str, str]]:
    name = (
        "eyepacs_directional_manifest.csv"
        if dataset == "eyepacs"
        else "messidor2_directional_manifest.csv"
    )
    path = repo / "outputs" / "data_audit" / name
    rows = read_rows(path)
    if not rows:
        raise RuntimeError(f"empty directional manifest: {path}")
    return rows


def deterministic_messidor_inner(
    rows: list[dict[str, str]], val_fraction: float = 0.2
) -> list[dict[str, str]]:
    train_patients = sorted({r["patient_id"] for r in rows if r["split"] == "train"})
    ranked = sorted(train_patients, key=lambda p: hashlib.sha256(f"2026:{p}".encode()).hexdigest())
    val_ids = set(ranked[: round(len(ranked) * val_fraction)])
    copied = []
    for row in rows:
        row = dict(row)
        if row["split"] == "train":
            row["run_split"] = "inner_val" if row["patient_id"] in val_ids else "train"
        else:
            row["run_split"] = row["split"]
        copied.append(row)
    return copied


def rows_for_run(repo: Path, dataset: str) -> list[dict[str, str]]:
    rows = load_manifest(repo, dataset)
    if dataset == "eyepacs":
        for row in rows:
            row["run_split"] = row["split"]
    else:
        rows = deterministic_messidor_inner(rows)
    return rows


class DirectionalDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], image_root: Path, transform):
        self.rows, self.image_root, self.transform = rows, image_root, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        target = self.transform(
            Image.open(self.image_root / row["target_image_path"]).convert("RGB")
        )
        fellow = self.transform(
            Image.open(self.image_root / row["fellow_image_path"]).convert("RGB")
        )
        return target, fellow, torch.tensor(int(row["y_true"]), dtype=torch.long), index


def image_root(repo: Path, dataset: str) -> Path:
    return repo / "data" / ("eyepacs" if dataset == "eyepacs" else "messidor2/IMAGES")


def build_model(
    repo: Path, method: str, pretrained: bool = True, parent_checkpoint: Path | None = None
):
    from drpair.benchmark.quick_models import (
        DRGCNNPaired,
        DRGCNNSingle,
        HuangPaired,
        HuangSingle,
        load_drgcnn_encoder,
        load_huang_encoder,
    )

    if method == "huang_single":
        return HuangSingle(pretrained)
    if method == "drgcnn_single":
        return DRGCNNSingle(pretrained)
    if method in PAIRED:
        if parent_checkpoint is None:
            raise ValueError(f"{method} requires same-seed single checkpoint")
        checkpoint = torch.load(parent_checkpoint, map_location="cpu", weights_only=False)
        return (
            HuangPaired(load_huang_encoder(checkpoint))
            if method == "huang_pff"
            else DRGCNNPaired(load_drgcnn_encoder(checkpoint))
        )
    if method == "d_odr":
        source = repo / "third_party" / "D-ODR"
        sys.path.insert(0, str(source))
        try:
            from model.model import DRGradingModelWrapper

            return DRGradingModelWrapper(
                backbone="vit_base_patch16_224",
                use_pretrained=pretrained,
                img_size=224,
                lambda_odr=1.0,
                kernel_k=25,
            )
        finally:
            sys.path.pop(0)
    if method == "aor_dr":
        from drpair.benchmark.aor_dr import AORDR

        return AORDR(pretrained=pretrained, diffusion_batch_mul=4, sampling_steps="100")
    raise ValueError(method)


def optimizer_for(model: nn.Module, method: str):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if method == "huang_single":
        return torch.optim.SGD(parameters, lr=0.001, momentum=0.9, nesterov=True, weight_decay=5e-4)
    if method == "drgcnn_single":
        return torch.optim.SGD(
            parameters, lr=0.0005, momentum=0.9, nesterov=True, weight_decay=5e-4
        )
    if method in PAIRED:
        return torch.optim.SGD(parameters, lr=0.02, momentum=0.9, nesterov=True, weight_decay=5e-4)
    if method == "d_odr":
        return torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-4)
    return torch.optim.AdamW(parameters, lr=1e-4)


def schedule_for(optimizer, method: str):
    epochs = epochs_for(method)
    if method in {"huang_single", "drgcnn_single"}:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return None


def epochs_for(method: str) -> int:
    return {
        "huang_single": 25,
        "huang_pff": 50,
        "drgcnn_single": 25,
        "drgcnn_bffn": 2,
        "d_odr": 100,
        "aor_dr": 300,
    }[method]


def batch_for(method: str) -> int:
    return {
        "huang_single": 16,
        "huang_pff": 64,
        "drgcnn_single": 16,
        "drgcnn_bffn": 64,
        "d_odr": 128,
        "aor_dr": 32,
    }[method]


def forward_loss(
    model: nn.Module, method: str, target: torch.Tensor, fellow: torch.Tensor, labels: torch.Tensor
):
    if method == "d_odr":
        output = model(pixel_values=target, labels=labels)
        return output["loss"], output["logits"].flatten()
    if method == "aor_dr":
        return model.training_loss(target, labels), None
    score = model(target, fellow if method in PAIRED else None)
    return nn.functional.mse_loss(score.float(), labels.float()), score


@torch.inference_mode()
def infer_batch(model: nn.Module, method: str, target: torch.Tensor, fellow: torch.Tensor):
    if method == "d_odr":
        score = model(pixel_values=target)["logits"].flatten()
        prob = torch.softmax(-((score[:, None] - torch.arange(5, device=score.device)) ** 2), dim=1)
        return score, prob
    if method == "aor_dr":
        boundary = model.sample(target)
        sampled_class = boundary.round().clamp(0, 1).sum(1).long()
        prob = torch.nn.functional.one_hot(sampled_class, num_classes=5).float()
        return sampled_class.float(), prob
    score = model(target, fellow if method in PAIRED else None)
    prob = torch.softmax(-((score[:, None] - torch.arange(5, device=score.device)) ** 2), dim=1)
    return score, prob
