"""Models used by the capped EyePACS existing-method benchmark."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    ConvNeXt_Small_Weights,
    ResNet50_Weights,
    convnext_small,
    resnet50,
)


class ConvNeXtSingle(nn.Module):
    """B0: ImageNet initialized ConvNeXt-S with a native five-class head."""

    output_type = "five_class_logits"

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = convnext_small(weights=weights)
        features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(features, 5)

    def forward(self, target: torch.Tensor, fellow: torch.Tensor | None = None):
        del fellow
        return self.model(target)


class HuangSingle(nn.Module):
    """B1-S: Huang et al. ResNet-50 scalar-regression eye model."""

    output_type = "scalar_grade_regression"

    def __init__(self, pretrained: bool = True):
        super().__init__()
        # torchvision's legacy pretrained=True used V1 in the published code era.
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder = resnet50(weights=weights)
        self.encoder.fc = nn.Identity()
        self.regressor = nn.Linear(2048, 1)

    def forward(self, target: torch.Tensor, fellow: torch.Tensor | None = None):
        del fellow
        return self.regressor(self.encoder(target)).squeeze(1)


class HuangPaired(nn.Module):
    """B1-P: official frozen ResNet-50 plus paired feature fusion head."""

    output_type = "scalar_grade_regression"

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.encoder.fc = nn.Identity()
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.fusion = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.MaxPool1d(2, 2),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.MaxPool1d(2, 2),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1, bias=False),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, target: torch.Tensor, fellow: torch.Tensor | None = None):
        if fellow is None:
            raise ValueError("HuangPaired requires a fellow-eye tensor")
        with torch.no_grad():
            target_features = self.encoder(target)
            fellow_features = self.encoder(fellow)
        joined = torch.cat((target_features, fellow_features), dim=1).unsqueeze(1)
        return self.fusion(joined).flatten()


def load_huang_encoder(single_checkpoint: dict) -> nn.Module:
    """Recover only the selected B1-S encoder; the regression head is discarded."""
    encoder = resnet50(weights=None)
    encoder.fc = nn.Identity()
    prefix = "encoder."
    state = {
        key[len(prefix) :]: value
        for key, value in single_checkpoint["model_state"].items()
        if key.startswith(prefix)
    }
    encoder.load_state_dict(state, strict=True)
    return encoder
