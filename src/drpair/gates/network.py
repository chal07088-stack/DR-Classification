import torch
from torch import nn


class RollbackGate(nn.Module):
    def __init__(self, inputs, hidden=16, global_gate=False):
        super().__init__()
        outputs = 1 if global_gate else 4
        self.network = nn.Sequential(
            nn.Linear(inputs, hidden), nn.Tanh(), nn.Linear(hidden, outputs)
        )
        self.register_buffer("mean", torch.zeros(inputs))
        self.register_buffer("std", torch.ones(inputs))
        self.register_buffer("support_scale", torch.ones(outputs))
        nn.init.constant_(self.network[-1].bias, -3.0)

    def forward(self, x):
        alpha = self.network((x - self.mean) / self.std).sigmoid() * self.support_scale
        return alpha.expand(-1, 4) if alpha.shape[1] == 1 else alpha


def project_torch(q):
    # Decreasing isotonic regression: min over left ends, max over right ends.
    values = []
    for i in range(4):
        left_options = []
        for a in range(i + 1):
            left_options.append(
                torch.stack([q[:, a : b + 1].mean(1) for b in range(i, 4)], 1).max(1).values
            )
        values.append(torch.stack(left_options, 1).min(1).values)
    return torch.stack(values, 1)
