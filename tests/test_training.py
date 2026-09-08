import json

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from drpair.models import PairedOrdinal, SingleOrdinal
from drpair.models.ordinal import ordinal_targets
from drpair.training.ema import EMA, cpu_state
from drpair.training.engine import fit


class TinyEncoder(nn.Module):
    num_features = 6

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 6, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )

    def forward(self, x):
        return self.net(x)


class TinyData(Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, idx):
        return {
            "target": torch.full((3, 8, 8), (idx + 1) / 5),
            "fellow": torch.ones(3, 8, 8),
            "label": idx,
            "index": idx,
        }


def test_paired_encoder_gets_gradients_and_experts_are_independent():
    single, paired = SingleOrdinal(TinyEncoder()), PairedOrdinal(TinyEncoder())
    assert not ({id(p) for p in single.parameters()} & {id(p) for p in paired.parameters()})
    x = torch.randn(3, 3, 8, 8)
    logits = paired(x, x.flip(0))
    assert torch.all(torch.diff(logits, dim=1) <= 0)
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, ordinal_targets(torch.tensor([0, 2, 4]))
    )
    loss.backward()
    assert sum(p.grad.abs().sum() for p in paired.encoder.parameters()) > 0


def test_ema_snapshot_keeps_evaluated_weights_after_restore():
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1)
    ema = EMA(model, 0.5)
    model.weight.data.fill_(3)
    ema.update(model)
    with ema.applied(model):
        assert model.weight.item() == 2
        evaluated = cpu_state(model)
    assert model.weight.item() == 3
    assert evaluated["weight"].item() == 2


def test_partial_accumulation_window_and_saved_ema(tmp_path):
    torch.set_num_threads(2)
    model = SingleOrdinal(TinyEncoder())
    cfg = {
        "training": {
            "epochs": 1,
            "effective_batch": 4,
            "backbone_lr": 0.001,
            "head_lr": 0.001,
            "weight_decay": 0.0001,
            "ema_decay": 0.9,
            "grad_clip": 1.0,
        }
    }
    checkpoint, log = tmp_path / "model.pt", tmp_path / "log.jsonl"
    model, epoch = fit(
        model,
        DataLoader(TinyData(), batch_size=2),
        cfg,
        torch.device("cpu"),
        None,
        False,
        checkpoint,
        log,
    )
    record = json.loads(log.read_text().splitlines()[-1])
    assert record["samples"] == 5 and record["updates"] == 2 and epoch == 1
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["weights"] == "ema"
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, saved["model"][k])
