"""One disposable real-image probe per process; never produces research checkpoints."""

import argparse
import gc
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from drpair.data.images import make_loader
from drpair.data.manifest import read_manifest
from drpair.models import build_expert
from drpair.models.ordinal import ordinal_targets
from drpair.runtime import load_config, save_json, setup_device
from drpair.training.ema import EMA, cpu_state
from drpair.training.predictions import forward


def probe(cfg, paired, batch, iterations):
    device, dtype = setup_device(cfg)
    rows = [r for r in read_manifest(cfg["data"]["manifest"]) if r["split"] == "train"]
    order = np.random.default_rng(42).permutation(len(rows))[: batch * iterations]
    loader = make_loader([rows[i] for i in order], cfg, paired, True, 42, batch=batch)
    model = build_expert(cfg, paired).to(device).train()
    encoder = list(model.encoder.parameters())
    ids = {id(p) for p in encoder}
    heads = [p for p in model.parameters() if id(p) not in ids]
    t = cfg["training"]
    optimizer = torch.optim.AdamW(
        [{"params": encoder, "lr": t["backbone_lr"]}, {"params": heads, "lr": t["head_lr"]}],
        weight_decay=t["weight_decay"],
    )
    ema = EMA(model, t["ema_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    accum = t["effective_batch"] // batch
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    timings, losses = [], []
    wait_times = []
    iterator = iter(loader)
    samples = 0
    for step in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        item = next(iterator)
        loaded = time.perf_counter()
        y = item["label"].to(device)
        with torch.autocast("cuda", dtype=dtype):
            logits = forward(model, item, device, paired)
            loss = F.binary_cross_entropy_with_logits(logits.float(), ordinal_targets(y))
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite benchmark loss")
        scaler.scale(loss * len(y)).backward()
        samples += len(y)
        if (step + 1) % accum == 0 or step + 1 == iterations:
            scaler.unscale_(optimizer)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(samples)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), t["grad_clip"], error_if_nonfinite=True
            )
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                ema.update(model)
            optimizer.zero_grad(set_to_none=True)
            samples = 0
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
        wait_times.append(loaded - start)
        losses.append(loss.item())
        if step in (0, 4, iterations - 1):
            print(
                f"probe paired={paired} batch={batch} step={step + 1}/{iterations} "
                f"loss={losses[-1]:.5f}",
                flush=True,
            )
    # Exercise the EMA swap and snapshot too, while Adam states are still resident.
    with ema.applied(model), torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        model.eval()
        prediction = forward(model, item, device, paired).float().sigmoid()
        if not torch.isfinite(prediction).all():
            raise RuntimeError("Non-finite EMA prediction")
        saved = cpu_state(model)
        del saved
    torch.cuda.synchronize()
    result = {
        "status": "passed",
        "paired": paired,
        "batch": batch,
        "accumulation": accum,
        "effective_batch": t["effective_batch"],
        "iterations": iterations,
        "warmup_iterations": 5,
        "seconds_per_batch": float(np.mean(timings[5:])),
        "target_eyes_per_second": float(batch / np.mean(timings[5:])),
        "data_wait_seconds_per_batch": float(np.mean(wait_times[5:])),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "free_gib_after": torch.cuda.mem_get_info()[0] / 1024**3,
        "workers": cfg["runtime"]["workers"],
        "amp_dtype": str(dtype),
        "channels_last": cfg["runtime"].get("channels_last", False),
        "gpu": torch.cuda.get_device_name(),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "research_checkpoint_written": False,
    }
    del iterator, loader, model, optimizer, ema
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1.toml")
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--batch", required=True, type=int)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["runtime"].update(
        memory_limit_gib=13.0, channels_last=True, cpu_threads=4, workers=args.workers
    )
    if cfg["training"]["effective_batch"] % args.batch or args.iterations < 20:
        raise ValueError("Probe must keep effective batch and run at least 20 iterations")
    try:
        result = probe(cfg, args.paired, args.batch, args.iterations)
    except torch.cuda.OutOfMemoryError as exc:
        result = {
            "status": "memory_limit_reached",
            "paired": args.paired,
            "batch": args.batch,
            "workers": args.workers,
            "error": str(exc),
        }
    except Exception as exc:
        save_json(
            args.output,
            {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()},
        )
        raise
    save_json(args.output, result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
