import numpy as np
import torch

from scripts.run_phase2b_huang import HuangFusionHead, decode_score


def test_huang_head_matches_dimension_proportions():
    head = HuangFusionHead(768)
    result = head(torch.randn(5, 768), torch.randn(5, 768))
    assert result.shape == (5,)
    assert head.network[0].in_features == 1536
    assert head.network[0].out_features == 768
    assert head.network[3].in_features == 384
    assert head.network[3].out_features == 192
    assert head.network[6].in_features == 96


def test_huang_decoder_uses_official_half_integer_thresholds():
    score = np.array([-2.0, 0.49, 0.50, 1.49, 1.50, 3.50, 7.0])
    np.testing.assert_array_equal(decode_score(score), [0, 0, 1, 1, 2, 4, 4])
