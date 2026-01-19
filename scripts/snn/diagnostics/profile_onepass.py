import argparse
import time
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

from cifar_baseline.config import TrainConfig
from cifar_baseline.data import make_loaders
from cifar_baseline.utils import get_device, set_seed
from cifar_baseline.snn.snn_models import SpikingCifarCNN


@dataclass
class Conv2dSpec:
    cin: int
    cout: int
    h: int
    w: int
    k: int
    stride: int = 1
    padding: int = 0

    def hout_wout(self) -> Tuple[int, int]:
        # standard conv output size formula
        hout = (self.h + 2 * self.padding - self.k) // self.stride + 1
        wout = (self.w + 2 * self.padding - self.k) // self.stride + 1
        return hout, wout

    def macs_per_sample(self) -> int:
        hout, wout = self.hout_wout()
        return self.cout * hout * wout * (self.k * self.k * self.cin)

    def flops_per_sample(self) -> int:
        # 2 FLOPs per MAC 
        return 2 * self.macs_per_sample()


def bytes_tensor(b: int, c: int, h: int, w: int, bytes_per_el: int) -> int:
    return b * c * h * w * bytes_per_el


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=2)

    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--p-scale", type=float, default=3.0)
    ap.add_argument("--repeat-input", action="store_true")

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--compile", action="store_true")

    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    set_seed(args.seed)

    device = get_device()
    print("device:", device)

    cfg = TrainConfig()
    train_loader, _, _ = make_loaders(
        dataset_id=cfg.dataset_id,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    x, y = next(iter(train_loader))
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    model = SpikingCifarCNN(
        T=args.T,
        p_scale=args.p_scale,
        repeat_input=args.repeat_input,
        input_is_normalized=True,
    ).to(device)
    model.eval()

    if args.compile:
        model = torch.compile(model)  # type: ignore

    use_amp = bool(args.amp and device.type == "cuda")
    print(f"AMP: {use_amp}  compile: {bool(args.compile)}  repeat_input: {bool(args.repeat_input)}")
    print(f"batch: {tuple(x.shape)}  T={args.T}")

    # ---------- Static estimates (order of magnitude) ----------
    # These are "best effort" estimates based on a common CIFAR SNN layout:
    # conv1 -> LIF -> pool -> conv2 -> LIF -> pool -> head
    #
    # For te # of channels, update the specs below to match.
    B = args.batch_size
    T = args.T

    # Assumptions:
    # CIFAR input: 3x32x32
    # conv1: 3 -> 64, k=3, pad=1, stride=1 => 64x32x32
    # pool => 64x16x16
    # conv2: 64 -> 128, k=3, pad=1 => 128x16x16
    # pool => 128x8x8
    bytes_per_el = 2 if use_amp else 4

    conv1 = Conv2dSpec(cin=3, cout=64, h=32, w=32, k=3, stride=1, padding=1)
    h1, w1 = conv1.hout_wout()  # 32,32
    # after pool
    h1p, w1p = h1 // 2, w1 // 2  # 16,16

    conv2 = Conv2dSpec(cin=64, cout=128, h=h1p, w=w1p, k=3, stride=1, padding=1)
    h2, w2 = conv2.hout_wout()  # 16,16
    # after pool
    h2p, w2p = h2 // 2, w2 // 2  # 8,8

    flops_conv1 = conv1.flops_per_sample() * B * T
    flops_conv2 = conv2.flops_per_sample() * B * T
    flops_total = flops_conv1 + flops_conv2

    # LIF state traffic estimate (very approximate):
    # per LIF layer per timestep:
    # read activation + read v + write v + write spikes - 4x tensor pass
    lif1_bytes_per_t = 4 * bytes_tensor(B, 64, h1, w1, bytes_per_el)
    lif2_bytes_per_t = 4 * bytes_tensor(B, 128, h2, w2, bytes_per_el)
    lif_bytes_total = (lif1_bytes_per_t + lif2_bytes_per_t) * T

    # conv activation traffic (very rough lower bound): write output once per timestep
    conv_out_bytes_total = (
        bytes_tensor(B, 64, h1, w1, bytes_per_el) +
        bytes_tensor(B, 128, h2, w2, bytes_per_el)
    ) * T

    bytes_total_est = lif_bytes_total + conv_out_bytes_total
    intensity = (flops_total / bytes_total_est) if bytes_total_est > 0 else float("nan")

    print("\n--- Rough estimates (order of magnitude) ---")
    print(f"Assumed conv channels: conv1 3->64, conv2 64->128 (update in file if different)")
    print(f"FLOPs conv1: {flops_conv1/1e9:.2f} GFLOPs  | conv2: {flops_conv2/1e9:.2f} GFLOPs")
    print(f"Total FLOPs (conv only): {flops_total/1e9:.2f} GFLOPs  (for one forward over T)")
    print(f"Estimated bytes (LIF state + conv outputs): {bytes_total_est/1e9:.2f} GB  (one forward over T)")
    print(f"Operational intensity - {intensity:.2f} FLOPs/byte (low => bandwidth/overhead dominated)\n")

    # ---------- Runtime + profiler (top ops) ----------
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    # Warmup
    with torch.no_grad():
        for _ in range(args.warmup):
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
            else:
                logits = model(x)
                loss = F.cross_entropy(logits, y)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Timed run + profiling
    with torch.no_grad(), torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        t0 = time.time()
        for _ in range(args.steps):
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
            else:
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            if device.type == "cuda":
                torch.cuda.synchronize()
        t1 = time.time()

    per_iter = (t1 - t0) / max(1, args.steps)
    print(f"Measured: {per_iter*1000:.2f} ms/forward (avg over {args.steps} steps)")

    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    print("\n--- Profiler top operators ---")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=25))

    # Show peak CUDA memory if available
    if device.type == "cuda":
        try:
            peak = torch.cuda.max_memory_allocated() / (1024**2)
            print(f"\nPeak CUDA memory allocated (since start): {peak:.1f} MiB")
        except Exception:
            pass


if __name__ == "__main__":
    main()