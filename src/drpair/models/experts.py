import timm
import torch
from torch import nn

from .ordinal import OrderedCoralHead


class SingleOrdinal(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = OrderedCoralHead(encoder.num_features)

    def forward(self, target):
        return self.head(self.encoder(target))


class PairedOrdinal(nn.Module):
    """One trainable encoder shared by target and fellow, ordered concatenation."""

    def __init__(self, encoder, dropout=0.2):
        super().__init__()
        self.encoder = encoder
        d = encoder.num_features
        self.fusion = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(dropout))
        self.head = OrderedCoralHead(d)

    def forward(self, target, fellow):
        ft, ff = self.encoder(target), self.encoder(fellow)
        return self.head(self.fusion(torch.cat([ft, ff], dim=1)))


def build_expert(cfg, paired):
    encoder = timm.create_model(
        cfg["model"]["arch"],
        pretrained=cfg["model"]["pretrained"],
        num_classes=0,
        global_pool="avg",
    )
    if paired:
        return PairedOrdinal(encoder, cfg["model"]["fusion_dropout"])
    return SingleOrdinal(encoder)
