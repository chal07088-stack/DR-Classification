import torch

from drpair.benchmark.quick_models import (
    DRGCNNEncoder,
    DRGCNNPaired,
    HuangPaired,
    HuangSingle,
)
from drpair.benchmark.quick_training import alignment_key, calculate_metrics


def test_huang_output_shapes():
    single = HuangSingle(pretrained=False)
    paired = HuangPaired(single.encoder)
    paired.eval()
    image = torch.zeros(2, 3, 64, 64)
    assert paired(image, image).shape == (2,)
    assert not any(parameter.requires_grad for parameter in paired.encoder.parameters())


def test_metrics_and_alignment_key():
    metrics = calculate_metrics([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
    assert metrics["qwk"] == 1.0
    assert metrics["mae"] == 0.0
    assert alignment_key({"patient_id": "7", "target_eye": "left"}) == ("7", "left")


def test_drgcnn_encoder_parameter_interpretation():
    encoder = DRGCNNEncoder(pretrained=False)
    assert round(sum(parameter.numel() for parameter in encoder.parameters()) / 1e6, 2) == 52.88


def test_drgcnn_bffn_output_and_frozen_encoder():
    class DummyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(3, 1280)

        def forward(self, image):
            return self.layer(image.mean(dim=(2, 3)))

    paired = DRGCNNPaired(DummyEncoder())
    image = torch.zeros(2, 3, 8, 8)
    assert paired(image, image).shape == (2,)
    assert not any(parameter.requires_grad for parameter in paired.encoder.parameters())
