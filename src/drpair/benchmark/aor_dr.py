"""Adapter around the official MICCAI 2025 AOR-DR diffusion modules."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault(
    "HF_HOME", str(Path(__file__).resolve().parents[3] / "data" / "cache" / "huggingface")
)

import timm
import torch
from torch import nn
from torch.nn import functional as F


def _official_diff_loss():
    source = Path(__file__).resolve().parents[3] / "third_party/AOR-DR"
    if not (source / "diffloss.py").is_file():
        raise RuntimeError("Run scripts/fetch_official_sources.py before B4")
    sys.path.insert(0, str(source))
    try:
        from diffloss import DiffLoss
    finally:
        sys.path.pop(0)
    return DiffLoss


class AORFusion(nn.Module):
    """Device-safe form of the official cross-sample autoregressive fusion."""

    def __init__(self, feature_dim: int = 768, embedding_dim: int = 768):
        super().__init__()
        self.position_embedding = nn.Embedding(4, embedding_dim)
        self.proj_feature = nn.Linear(feature_dim, embedding_dim)
        self.proj_context = nn.Embedding(5, embedding_dim)

    def forward(self, features: torch.Tensor, context: torch.Tensor, step: int):
        features = self.proj_feature(features)
        context = self.proj_context(context.long()).squeeze(1)
        position = torch.full(
            (len(context),), step, dtype=torch.long, device=context.device
        )
        context = context + self.position_embedding(position)
        attention = F.softmax(features @ context.T, dim=1)
        return attention @ context + features


class AORDR(nn.Module):
    """ViT-B plus four autoregressive conditional diffusion boundary steps."""

    output_type = "autoregressive_diffusion_boundaries"

    def __init__(
        self,
        pretrained: bool = True,
        diffusion_batch_mul: int = 4,
        sampling_steps: str = "100",
    ):
        super().__init__()
        DiffLoss = _official_diff_loss()
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=pretrained)
        self.diffusion = DiffLoss(
            target_channels=1, z_channels=768, width=1024, depth=3,
            num_sampling_steps=sampling_steps, grad_checkpointing=False,
        )
        self.fusion = AORFusion()
        self.diffusion_batch_mul = diffusion_batch_mul

    def _features(self, image: torch.Tensor):
        features = self.vit.forward_features(image.float())
        return features[:, 0] if features.ndim == 3 else features

    def training_loss(self, image: torch.Tensor, labels: torch.Tensor):
        targets = (labels[:, None] > torch.arange(4, device=labels.device)).T.float()
        total, context, features = 0.0, None, None
        for step in range(4):
            target = targets[step].unsqueeze(1).repeat(self.diffusion_batch_mul, 1)
            if step == 0:
                features = self._features(image).repeat(self.diffusion_batch_mul, 1)
            else:
                context = context.round().clamp(0, 1)
                features = self.fusion(features, context, step)
            loss, context = self.diffusion(target=target, z=features)
            total = total + loss
        return total

    @torch.inference_mode()
    def sample(self, image: torch.Tensor):
        features = self._features(image)
        boundaries = []
        for step in range(4):
            context = self.diffusion.sample(features, temperature=1.0, cfg=1.0)
            boundaries.append(context.squeeze(1))
            features = self.fusion(features, context, step)
        return torch.stack(boundaries, dim=1)
