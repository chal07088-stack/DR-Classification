import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset

from drpair.runtime import seed_worker

MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


class EyeDataset(Dataset):
    def __init__(self, rows, root, size, paired, augmentation=None):
        self.rows, self.root, self.size = rows, Path(root), size
        self.paired, self.augmentation = paired, augmentation

    def __len__(self):
        return len(self.rows)

    def _parameters(self):
        if self.augmentation is None:
            return False, 0.0, 1.0, 1.0
        a = self.augmentation
        return (
            random.random() < a["horizontal_flip"],
            random.uniform(-a["rotation_degrees"], a["rotation_degrees"]),
            random.uniform(1 - a["brightness"], 1 + a["brightness"]),
            random.uniform(1 - a["contrast"], 1 + a["contrast"]),
        )

    def _load(self, path, parameters):
        flip, angle, brightness, contrast = parameters
        with Image.open(self.root / path) as source:
            im = source.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
        if flip:
            im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if angle:
            im = im.rotate(angle, resample=Image.Resampling.BILINEAR)
        if brightness != 1.0:
            im = ImageEnhance.Brightness(im).enhance(brightness)
        if contrast != 1.0:
            im = ImageEnhance.Contrast(im).enhance(contrast)
        tensor = torch.from_numpy(np.asarray(im, dtype=np.float32).copy()).permute(2, 0, 1) / 255
        return (tensor - MEAN) / STD

    def __getitem__(self, index):
        row = self.rows[index]
        parameters = self._parameters()
        result = {
            "target": self._load(row["path"], parameters),
            "label": row["label"],
            "index": index,
        }
        if self.paired:
            result["fellow"] = self._load(row["fellow_path"], parameters)
        return result


def make_loader(rows, cfg, paired, train, seed, batch=None):
    dataset = EyeDataset(
        rows,
        cfg["data"]["root"],
        cfg["model"]["image_size"],
        paired,
        cfg["augmentation"] if train else None,
    )
    runtime = cfg["runtime"]
    size = batch or cfg["training"]["paired_batch" if paired else "single_batch"]
    kwargs = {
        "batch_size": size,
        "shuffle": train,
        "drop_last": False,
        "num_workers": runtime["workers"],
        "pin_memory": runtime["pin_memory"],
        "worker_init_fn": seed_worker,
        "generator": torch.Generator().manual_seed(seed),
    }
    if runtime["workers"]:
        kwargs.update(
            persistent_workers=runtime["persistent_workers"],
            prefetch_factor=runtime["prefetch_factor"],
        )
    return DataLoader(dataset, **kwargs)
