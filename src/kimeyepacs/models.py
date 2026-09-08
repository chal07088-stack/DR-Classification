import torch
import torch.nn as nn
import timm


class SingleOrdinal(nn.Module):
    def __init__(self, arch: str, pretrained: bool):
        super().__init__()
        self.encoder = timm.create_model(arch, pretrained=pretrained, num_classes=0, global_pool="avg")
        self.num_features = int(self.encoder.num_features)
        self.head = nn.Linear(self.num_features, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class PairedOrdinal(nn.Module):
    def __init__(self, encoder: nn.Module, feat_dim: int):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        d = 2 * feat_dim
        self.head = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(d, d // 4),
            nn.ReLU(inplace=True),
            nn.Linear(d // 4, 4),
        )

    def forward(self, target: torch.Tensor, fellow: torch.Tensor) -> torch.Tensor:
        self.encoder.eval()
        with torch.no_grad():
            ft = self.encoder(target)
            ff = self.encoder(fellow)
        return self.head(torch.cat([ft, ff], dim=1).float())


def trainable_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
