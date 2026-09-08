"""CORAL shared score with explicitly ordered cutpoints (ordered-CORAL variant)."""

import torch
from torch import nn
from torch.nn import functional as F


class OrderedCoralHead(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.score = nn.Linear(features, 1, bias=False)
        self.first_cutpoint = nn.Parameter(torch.tensor(-1.5))
        self.raw_gaps = nn.Parameter(torch.full((3,), 0.54132485))

    def forward(self, x):
        cuts = torch.cat(
            [
                self.first_cutpoint[None],
                self.first_cutpoint + torch.cumsum(F.softplus(self.raw_gaps), 0),
            ]
        )
        return self.score(x) - cuts


def ordinal_targets(labels):
    return (labels[:, None] > torch.arange(4, device=labels.device)).float()
