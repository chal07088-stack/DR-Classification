"""Models used by the capped EyePACS existing-method benchmark."""

from __future__ import annotations

import os

import torch
from torch import nn
from torchvision.models import (
    ConvNeXt_Small_Weights,
    EfficientNet_V2_M_Weights,
    ResNet50_Weights,
    convnext_small,
    efficientnet_v2_m,
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


class CategoryAttention(nn.Module):
    """DRGCNN category attention module, inserted before global pooling."""

    def __init__(self, dim: int = 1280, classes: int = 5, channels_per_class: int = 3):
        super().__init__()
        channels = classes * channels_per_class
        self.classes, self.channels_per_class = classes, channels_per_class
        self.conv = nn.Conv2d(dim, channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor):
        batch, _, height, width = inputs.shape
        category = self.relu(self.bn(self.conv(inputs)))
        scores = nn.functional.adaptive_max_pool2d(category, 1)
        scores = scores.view(batch, 1, 1, self.classes, self.channels_per_class).mean(-1)
        # Preserve the released DRGCNN CAM tensor view exactly.  The upstream
        # implementation views BCHW storage as BHW(class,k) without permuting.
        maps = category.reshape(
            batch, height, width, self.classes, self.channels_per_class
        ).mean(-1)
        attention = (maps * scores).mean(-1).unsqueeze(1)
        return inputs * attention


class DRGCNNEncoder(nn.Module):
    """ImageNet EfficientNetV2-M with DRGCNN CAM before average pooling."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None
        base = efficientnet_v2_m(weights=weights)
        self.features = base.features
        # k=3 is implied by the paper's reported 52.88M encoder parameter count.
        self.category_attention = CategoryAttention(channels_per_class=3)
        self.avgpool = base.avgpool

    def forward(self, image: torch.Tensor):
        features = self.category_attention(self.features(image))
        return torch.flatten(self.avgpool(features), 1)


class DRGCNNSingle(nn.Module):
    """B2-S: matched CAM-EfficientNetV2-M eye grader without BFFN."""

    output_type = "scalar_grade_regression"

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.encoder = DRGCNNEncoder(pretrained)
        self.regressor = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 1))

    def forward(self, target: torch.Tensor, fellow: torch.Tensor | None = None):
        del fellow
        return self.regressor(self.encoder(target)).squeeze(1)


class DRGCNNPaired(nn.Module):
    """B2-P: selected frozen B2-S encoder plus official BFFN head."""

    output_type = "scalar_grade_regression"

    def __init__(self, encoder: DRGCNNEncoder):
        super().__init__()
        self.encoder = encoder
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.fusion = nn.Sequential(
            nn.Linear(2560, 1280),
            nn.MaxPool1d(2, 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(640, 320),
            nn.MaxPool1d(2, 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(160, 1, bias=False),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # The released BFFN calls model.train() after freezing encoder
        # parameters. Consequently its BatchNorm buffers update even though
        # gradients are disabled. Keep the cleaned benchmark default, while
        # allowing an isolated reproduction of that public-code behaviour.
        if os.environ.get("DRGCNN_BFFN_UPDATE_BN", "0") != "1":
            self.encoder.eval()
        return self

    def forward(self, target: torch.Tensor, fellow: torch.Tensor | None = None):
        if fellow is None:
            raise ValueError("DRGCNNPaired requires a fellow-eye tensor")
        with torch.no_grad():
            target_features = self.encoder(target)
            fellow_features = self.encoder(fellow)
        return self.fusion(torch.cat((target_features, fellow_features), 1).unsqueeze(1)).flatten()


def load_drgcnn_encoder(single_checkpoint: dict) -> DRGCNNEncoder:
    encoder = DRGCNNEncoder(pretrained=False)
    prefix = "encoder."
    state = {
        key[len(prefix) :]: value
        for key, value in single_checkpoint["model_state"].items()
        if key.startswith(prefix)
    }
    encoder.load_state_dict(state, strict=True)
    return encoder
