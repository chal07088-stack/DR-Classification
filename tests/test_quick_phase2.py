import torch

from drpair.benchmark.quick_models import HuangPaired, HuangSingle
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
