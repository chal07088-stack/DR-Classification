"""Measure one faithful training step without reading study images or checkpoints.

This is a hardware-capacity probe only.  Models use random initialization and
synthetic tensors, but precision, resolution, loss shape, and optimizer family
match the thesis runners closely enough to expose CUDA out-of-memory failures.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def baseline_model(method: str):
    from drpair.benchmark.quick_models import (
        DRGCNNPaired,
        DRGCNNSingle,
        HuangPaired,
        HuangSingle,
    )

    if method == "huang_single":
        return HuangSingle(pretrained=False), 512, False, False
    if method == "huang_pff":
        return HuangPaired(HuangSingle(pretrained=False).encoder), 512, True, False
    if method == "drgcnn_single":
        return DRGCNNSingle(pretrained=False), 512, False, False
    if method == "drgcnn_bffn":
        return DRGCNNPaired(DRGCNNSingle(pretrained=False).encoder), 512, True, False
    if method == "d_odr":
        source = ROOT / "third_party" / "D-ODR"
        sys.path.insert(0, str(source))
        from model.model import DRGradingModelWrapper

        model = DRGradingModelWrapper(
            backbone="vit_base_patch16_224",
            use_pretrained=False,
            img_size=224,
            lambda_odr=1.0,
            kernel_k=25,
        )
        return model, 224, False, True
    raise ValueError(method)


class IdeaStageA(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import convnext_small

        base = convnext_small(weights=None)
        self.features = base.features
        self.avgpool = base.avgpool
        self.norm = base.classifier[0]
        self.classifier = nn.Linear(768, 5)

    def forward(self, images: torch.Tensor):
        values = self.norm(self.avgpool(self.features(images))).flatten(1)
        return self.classifier(values)


def make(method: str):
    if method == "our_idea":
        return IdeaStageA(), 512, False, True
    return baseline_model(method)


def optimizer_for(model: nn.Module, method: str):
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if method == "huang_single":
        return torch.optim.SGD(params, lr=1e-3, momentum=0.9, nesterov=True, weight_decay=5e-4)
    if method == "drgcnn_single":
        return torch.optim.SGD(params, lr=5e-4, momentum=0.9, nesterov=True, weight_decay=5e-4)
    if method in {"huang_pff", "drgcnn_bffn"}:
        return torch.optim.SGD(params, lr=0.02, momentum=0.9, nesterov=True, weight_decay=5e-4)
    if method in {"d_odr", "our_idea"}:
        return torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4 if method == "d_odr" else 0.05)
    raise ValueError(method)


def run(method: str, batch: int):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, size, paired, use_amp = make(method)
    model = model.to(device).train()
    optimizer = optimizer_for(model, method)
    target = torch.randn(batch, 3, size, size, device=device)
    fellow = torch.randn_like(target) if paired else None
    labels = torch.randint(0, 5, (batch,), device=device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        if method == "d_odr":
            output = model(pixel_values=target, labels=labels)
            loss = output["loss"]
        elif method == "our_idea":
            loss = nn.functional.cross_entropy(model(target), labels)
        else:
            score = model(target, fellow)
            loss = nn.functional.mse_loss(score.float(), labels.float())
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(device)
    result = {
        "method": method,
        "physical_batch": batch,
        "resolution": size,
        "amp": use_amp,
        "loss": float(loss.detach()),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        "device_total_mib": round(torch.cuda.get_device_properties(0).total_memory / 2**20, 1),
        "status": "PASS",
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    del optimizer, model, target, fellow, labels, loss
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        required=True,
        choices=["huang_single", "huang_pff", "drgcnn_single", "drgcnn_bffn", "d_odr", "our_idea"],
    )
    parser.add_argument("--batch", required=True, type=int)
    args = parser.parse_args()
    try:
        run(args.method, args.batch)
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        print(
            json.dumps(
                {"method": args.method, "physical_batch": args.batch, "status": "OOM", "error": str(exc)},
                ensure_ascii=False,
            ),
            flush=True,
        )
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
