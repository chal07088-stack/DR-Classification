import argparse
import json
from pathlib import Path

import torch

from .models import PairedOrdinal, SingleOrdinal
from .runtime import configure_torch, device_and_amp, load_config


def _run(model, args, target, steps, device, amp_dtype):
    crit = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda" and amp_dtype is not None)):
            out = model(*args)
            loss = crit(out, target)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    return {
        "allocated_gb": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "reserved_gb": torch.cuda.max_memory_reserved() / 1024 ** 3,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--single-batches", default="16,12,8,6,4")
    ap.add_argument("--paired-batches", default="8,6,4,3,2")
    a = ap.parse_args()

    cfg = load_config(a.config)
    configure_torch(cfg)
    device, amp_dtype = device_and_amp(cfg["runtime"].get("precision", "bf16"))
    if device.type != "cuda":
        raise SystemExit("CUDA is required for VRAM probe")

    arch = cfg["model"]["arch"]
    size = int(cfg["model"]["image_size"])
    rows = []
    print(f"device={torch.cuda.get_device_name(0)} amp={amp_dtype}", flush=True)

    for batch in [int(x) for x in a.single_batches.split(",")]:
        try:
            model = SingleOrdinal(arch, pretrained=False).to(device)
            x = torch.randn(batch, 3, size, size, device=device)
            y = torch.randint(0, 2, (batch, 4), device=device).float()
            row = {"kind": "single", "batch": batch, **_run(model, (x,), y, a.steps, device, amp_dtype), "ok": True}
        except RuntimeError as e:
            row = {"kind": "single", "batch": batch, "ok": False, "error": str(e)}
        print(row, flush=True)
        rows.append(row)
        torch.cuda.empty_cache()

    base = SingleOrdinal(arch, pretrained=False).to(device)
    for batch in [int(x) for x in a.paired_batches.split(",")]:
        try:
            model = PairedOrdinal(base.encoder, base.num_features).to(device)
            x1 = torch.randn(batch, 3, size, size, device=device)
            x2 = torch.randn(batch, 3, size, size, device=device)
            y = torch.randint(0, 2, (batch, 4), device=device).float()
            row = {"kind": "paired", "batch": batch, **_run(model, (x1, x2), y, a.steps, device, amp_dtype), "ok": True}
        except RuntimeError as e:
            row = {"kind": "paired", "batch": batch, "ok": False, "error": str(e)}
        print(row, flush=True)
        rows.append(row)
        torch.cuda.empty_cache()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
