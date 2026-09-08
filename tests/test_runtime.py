from types import SimpleNamespace

import pytest

from drpair import runtime


def test_cuda_memory_cap_uses_explicit_device_index(monkeypatch):
    calls = []
    monkeypatch.setattr(runtime, "seed_everything", lambda *args: None)
    monkeypatch.setattr(runtime.torch, "set_num_threads", lambda *args: None)
    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime.torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(runtime.torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(
        runtime.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=16 * 1024**3),
    )
    monkeypatch.setattr(
        runtime.torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, index: calls.append((fraction, index)),
    )
    cfg = {
        "experiment": {"seed": 42},
        "runtime": {
            "deterministic": True,
            "cudnn_benchmark": False,
            "tf32": True,
            "memory_limit_gib": 13.0,
        },
    }
    device, dtype = runtime.setup_device(cfg)
    assert device.type == "cuda" and dtype == runtime.torch.bfloat16
    assert calls == [(13 / 16, 3)]
    cfg["runtime"]["memory_limit_gib"] = 16
    with pytest.raises(ValueError, match="below total VRAM"):
        runtime.setup_device(cfg)
