import argparse
import platform
import time
from typing import List, Tuple

import torch
import torch.nn.functional as F

# Modules from src/cifar_baseline --------------------------------

from cifar_baseline.config import TrainConfig
from cifar_baseline.data import make_loaders
from cifar_baseline.utils import get_device, set_seed
from cifar_baseline.snn.snn_models import SpikingCifarCNN
from cifar_baseline.snn.es import ESConfig, _add_noise, _set_eval_seed
from cifar_baseline.snn.es_parallel import es_update_step_parallel_chunked

Batch = Tuple[torch.Tensor, torch.Tensor]  # (x, y)


# Model -----------------------------------------------------------
class DeterministicRateEncoder(torch.nn.Module):
    """
    - Deterministic rate encoder: returns p in [0,1] instead of sampling spikes.
    - Assumes CIFAR-10 normalization.
    """
    def __init__(self, p_scale: float, input_is_normalized: bool = True):
        super().__init__()
        self.p_scale = float(p_scale)
        self.input_is_normalized = bool(input_is_normalized)

        # Standard CIFAR-10 mean/std (torchvision)
        mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32).view(1, 3, 1, 1)
        std  = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_is_normalized:
            x = x * self.std + self.mean  # back to [0,1] approx
        p = (self.p_scale * x).clamp_(0.0, 1.0)
        return p

# Helpers ----------------------------------------------------------
def _default_num_workers() -> int:
    return 0 if platform.system() == "Darwin" else 4 # macOS has issues with multiprocessing sometimes


@torch.no_grad()
def _eval_loss_acc(
    model:     torch.nn.Module,
    batches:   List[Batch],
    device:    torch.device,
    use_amp:   bool,
    seed_base: int,
) -> Tuple[float, float]:
    
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for bi, (x, y) in enumerate(batches):
        _set_eval_seed(seed_base + 10_000 * bi, device)

        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
        else:
            logits = model(x)
            loss = F.cross_entropy(logits, y)

        bs = y.numel()
        total_loss += float(loss.item()) * bs
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_n += bs

    return total_loss / max(1, total_n), total_correct / max(1, total_n)


@torch.no_grad()
def _fitness_neg_ce(
    model:     torch.nn.Module,
    batches:   List[Batch],
    device:    torch.device,
    use_amp:   bool,
    seed_base: int,
) -> float:
    
    loss, _acc = _eval_loss_acc(model, batches, device=device, use_amp=use_amp, seed_base=seed_base)
    return -loss


def _collect_fixed_subset(loader, device: torch.device, n_samples: int) -> List[Batch]:
    out: List[Batch] = []
    seen = 0

    for x, y in loader:
        if seen >= n_samples:
            break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        remaining = n_samples - seen
        if y.numel() > remaining:
            x = x[:remaining]
            y = y[:remaining]

        out.append((x, y))
        seen += y.numel()

    if seen < n_samples:
        raise RuntimeError(f"Only collected {seen} samples, expected {n_samples}. Reduce --train-samples/--test-samples.")
    return out


@torch.no_grad()
def es_update_step(
    model:         torch.nn.Module,
    params:        List[torch.nn.Parameter],
    train_batches: List[Batch],
    cfg:           ESConfig,
    device:        torch.device,
    *,
    max_update: float,
) -> Tuple[float, float]:
    
    use_amp = bool(cfg.use_amp and device.type == "cuda") 

    # Base metrics for logging
    train_loss, train_acc = _eval_loss_acc(model, train_batches, device=device, use_amp=use_amp, seed_base=cfg.base_seed)

    if cfg.population_size % 2 != 0:
        raise ValueError("population_size must be even.") # avoid unpaired samples
    pairs = cfg.population_size // 2

    diffs: List[float] = [] # for (f_plus - f_minus)
    eps_cache: List[List[torch.Tensor]] = [] # store epsilons for each pair

    for k in range(pairs):
        eps_list = [torch.randn_like(p) for p in params]
        pair_seed = cfg.base_seed + 1_000_000 + k

        # +sigma
        _add_noise(params, eps_list, cfg.sigma)
        f_plus = _fitness_neg_ce(model, train_batches, device, use_amp, seed_base=pair_seed)

        # -sigma
        _add_noise(params, eps_list, -2.0 * cfg.sigma)
        f_minus = _fitness_neg_ce(model, train_batches, device, use_amp, seed_base=pair_seed)

        # restore
        _add_noise(params, eps_list, cfg.sigma)

        diffs.append(f_plus - f_minus)
        eps_cache.append(eps_list)

    d = torch.tensor(diffs, device=device, dtype=torch.float32)
    d = (d - d.mean()) / (d.std(unbiased=False) + 1e-8)

    grad_acc = [torch.zeros_like(p) for p in params]
    for di, eps_list in zip(d.tolist(), eps_cache):
        for g, e in zip(grad_acc, eps_list):
            g.add_(float(di) * e)

    coef = cfg.lr / (2.0 * pairs * cfg.sigma)
    for p, g in zip(params, grad_acc):
        delta = (coef * g).clamp_(-max_update, max_update)
        p.add_(delta)

    return train_loss, train_acc

# Main ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=25)

    ap.add_argument("--train-samples", type=int, default=5000)
    ap.add_argument("--test-samples", type=int, default=1000)

    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=_default_num_workers())

    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--p-scale", type=float, default=3.0)

    ap.add_argument("--pop", type=int, default=64)  # even
    ap.add_argument("--sigma", type=float, default=0.03)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--max-update", type=float, default=1e-3)

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--deterministic-encoder", action="store_true",
                help="Replace Bernoulli spike sampling with deterministic rate encoder (stabilizes ES).")

    ap.add_argument("--parallel-chunk", type=int, default=0,
                help="If >0 and CUDA, evaluate ES population in chunks using vmap (single-GPU parallel ES).")

    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print("device:", device)

    cfg = TrainConfig()
    train_loader, test_loader, _ = make_loaders(
        dataset_id=cfg.dataset_id,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    print(f"Collecting fixed train subset: {args.train_samples} samples ...")
    train_batches = _collect_fixed_subset(train_loader, device, args.train_samples)

    print(f"Collecting fixed test subset: {args.test_samples} samples ...")
    test_batches = _collect_fixed_subset(test_loader, device, args.test_samples)

    model = SpikingCifarCNN( 
        T                   =   args.T,
        p_scale             =   args.p_scale,
        input_is_normalized =   True,
        lif_decay           =   0.95,
        lif_th              =   1.0,
        reset_mode          =   "to_reset",
        num_classes         =   10,
        use_bn              =   False,
    ).to(device)

    if args.deterministic_encoder:
        if hasattr(model, "encoder"):
            model.encoder = DeterministicRateEncoder(p_scale=args.p_scale, input_is_normalized=True).to(device)
            print("Using deterministic rate encoder (no spike sampling).")
        else:
            print("Warning: model has no attribute 'encoder'. Deterministic encoder not applied.")

    es_cfg = ESConfig(
        population_size     =   args.pop,
        sigma               =   args.sigma,
        lr                  =   args.lr,
        iters               =   args.iters,
        batches_per_fitness =   1,
        use_amp             =   args.amp,
        base_seed           =   1234,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    use_amp = bool(es_cfg.use_amp and device.type == "cuda")

    for it in range(1, es_cfg.iters + 1):
        t0 = time.perf_counter()

        with torch.no_grad():
            if args.parallel_chunk > 0 and device.type == "cuda":
                # Log train metrics at current theta (before update)
                train_loss, train_acc = _eval_loss_acc(
                    model, train_batches, device=device, use_amp=use_amp, seed_base=es_cfg.base_seed
                )

                # Parallelized ES update (single GPU, chunked)
                es_update_step_parallel_chunked(
                    model,
                    train_batches,
                    population_size = args.pop,
                    sigma = args.sigma,
                    lr = args.lr,
                    max_update = args.max_update,
                    chunk_size = args.parallel_chunk, # number of pairs per chunk
                    use_amp = use_amp,
                    seed=es_cfg.base_seed + it,
                )
            else:
                train_loss, train_acc = es_update_step(
                    model, params, train_batches, es_cfg, device, max_update=args.max_update
                )

        dt = time.perf_counter() - t0

        if it == 1 or it % args.eval_every == 0:
            with torch.no_grad():
                _test_loss, test_acc = _eval_loss_acc(
                    model, test_batches, device=device, use_amp=use_amp, seed_base=9999
                )
            print(
                f"iter={it:04d}  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
                f"test_acc={test_acc:.3f}  "
                f"sec/iter={dt:.2f}"
            )

    print("Done.")


if __name__ == "__main__":
    main()


# (base) danieltorres@MacBook-Air-de-Daniel SNN-CIFAR10 % uv run python scripts/snn/diagnostics/es_subset_train_eval.py \
#   --iters 100 \
#   --eval-every 10 \
#   --train-samples 1000 \
#   --test-samples 500 \
#   --batch-size 128 \
#   --num-workers 0 \
#   --T 8 \
#   --p-scale 3.0 \
#   --pop 32 \
#   --sigma 0.05 \
#   --lr 0.01 \
#   --max-update 1e-3
# device: cpu
# Collecting fixed train subset: 1000 samples ...
# Collecting fixed test subset: 500 samples ...
# iter=0001  train_loss=2.3022  train_acc=0.109  test_acc=0.120  sec/iter=104.19
# iter=0010  train_loss=2.3013  train_acc=0.117  test_acc=0.138  sec/iter=123.15
# iter=0020  train_loss=2.3035  train_acc=0.105  test_acc=0.108  sec/iter=126.45
# iter=0030  train_loss=2.3038  train_acc=0.114  test_acc=0.108  sec/iter=124.69
# iter=0040  train_loss=2.3062  train_acc=0.107  test_acc=0.098  sec/iter=124.00
# iter=0050  train_loss=2.3066  train_acc=0.100  test_acc=0.076  sec/iter=125.38
# iter=0060  train_loss=2.3058  train_acc=0.094  test_acc=0.102  sec/iter=126.69
# iter=0070  train_loss=2.3051  train_acc=0.098  test_acc=0.076  sec/iter=125.82
# iter=0080  train_loss=2.3072  train_acc=0.098  test_acc=0.090  sec/iter=125.60
# iter=0090  train_loss=2.3056  train_acc=0.100  test_acc=0.104  sec/iter=125.27
# iter=0100  train_loss=2.3064  train_acc=0.111  test_acc=0.102  sec/iter=125.16
# Done.

# (base) danieltorres@MacBook-Air-de-Daniel SNN-CIFAR10 % uv run python scripts/snn/diagnostics/es_subset_train_eval.py \
#   --iters 80 \
#   --eval-every 10 \
#   --train-samples 1000 \
#   --test-samples 500 \
#   --batch-size 128 \
#   --num-workers 0 \
#   --T 8 \
#   --p-scale 3.0 \
#   --pop 16 \
#   --sigma 0.05 \
#   --lr 0.02 \
#   --max-update 3e-3 \
#   --deterministic-encoder
# device: cpu
# Collecting fixed train subset: 1000 samples ...
# Collecting fixed test subset: 500 samples ...
# Using deterministic rate encoder (no spike sampling).
# iter=0001  train_loss=2.3024  train_acc=0.111  test_acc=0.098  sec/iter=51.58
# iter=0010  train_loss=2.3002  train_acc=0.111  test_acc=0.110  sec/iter=61.10
# iter=0020  train_loss=2.3003  train_acc=0.099  test_acc=0.112  sec/iter=62.76
# iter=0030  train_loss=2.3023  train_acc=0.103  test_acc=0.108  sec/iter=63.26
# iter=0040  train_loss=2.3040  train_acc=0.086  test_acc=0.086  sec/iter=64.09
# iter=0050  train_loss=2.3026  train_acc=0.094  test_acc=0.112  sec/iter=64.19
# iter=0060  train_loss=2.3011  train_acc=0.109  test_acc=0.102  sec/iter=63.88
# iter=0070  train_loss=2.3030  train_acc=0.109  test_acc=0.106  sec/iter=64.34
# iter=0080  train_loss=2.3028  train_acc=0.111  test_acc=0.108  sec/iter=63.45
# Done.

# (base) danieltorres@MacBook-Air-de-Daniel SNN-CIFAR10 % >....                                                                                      
#   --train-samples 128 \
#   --test-samples 128 \
#   --batch-size 128 \
#   --num-workers 0 \
#   --T 8 \
#   --p-scale 3.0 \
#   --pop 16 \
#   --sigma 0.05 \
#   --lr 0.05 \
#   --max-update 1e-2 \
#   --deterministic-encoder
# device: cpu
# Collecting fixed train subset: 128 samples ...
# Collecting fixed test subset: 128 samples ...
# Using deterministic rate encoder (no spike sampling).
# iter=0001  train_loss=2.3055  train_acc=0.078  test_acc=0.133  sec/iter=7.03
# iter=0010  train_loss=2.3035  train_acc=0.086  test_acc=0.133  sec/iter=6.94
# iter=0020  train_loss=2.2977  train_acc=0.117  test_acc=0.102  sec/iter=7.16
# iter=0030  train_loss=2.2919  train_acc=0.125  test_acc=0.086  sec/iter=7.25
# iter=0040  train_loss=2.3113  train_acc=0.141  test_acc=0.078  sec/iter=7.37
# iter=0050  train_loss=2.3032  train_acc=0.125  test_acc=0.125  sec/iter=7.56
# iter=0060  train_loss=2.3470  train_acc=0.141  test_acc=0.062  sec/iter=7.64
# iter=0070  train_loss=2.3450  train_acc=0.117  test_acc=0.148  sec/iter=7.84
# iter=0080  train_loss=2.3468  train_acc=0.141  test_acc=0.164  sec/iter=7.89
# iter=0090  train_loss=2.3538  train_acc=0.078  test_acc=0.148  sec/iter=8.37
# iter=0100  train_loss=2.3961  train_acc=0.094  test_acc=0.070  sec/iter=8.06
# iter=0110  train_loss=2.3746  train_acc=0.117  test_acc=0.195  sec/iter=8.08
# iter=0120  train_loss=2.3874  train_acc=0.102  test_acc=0.117  sec/iter=8.03
# iter=0130  train_loss=2.4477  train_acc=0.117  test_acc=0.094  sec/iter=8.11
# iter=0140  train_loss=2.5776  train_acc=0.133  test_acc=0.070  sec/iter=8.13
# iter=0150  train_loss=2.4029  train_acc=0.141  test_acc=0.117  sec/iter=8.05
# Done.



# (base) danieltorres@MacBook-Air-de-Daniel SNN-CIFAR10 % uv run python scripts/snn/diagnostics/es_subset_train_eval.py \
#   --iters 200 \
#   --eval-every 10 \
#   --train-samples 128 \
#   --test-samples 128 \
#   --batch-size 128 \
#   --num-workers 0 \
#   --T 8 \
#   --p-scale 3.0 \
#   --pop 16 \
#   --sigma 0.05 \
#   --lr 0.01 \
#   --max-update 3e-3 \
#   --deterministic-encoder
# device: cpu
# Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
# Collecting fixed train subset: 128 samples ...
# Collecting fixed test subset: 128 samples ...
# Using deterministic rate encoder (no spike sampling).
# iter=0001  train_loss=2.3055  train_acc=0.078  test_acc=0.133  sec/iter=7.64
# iter=0010  train_loss=2.3047  train_acc=0.062  test_acc=0.148  sec/iter=6.95
# iter=0020  train_loss=2.3041  train_acc=0.047  test_acc=0.125  sec/iter=7.78
# iter=0030  train_loss=2.3050  train_acc=0.094  test_acc=0.102  sec/iter=7.85
# iter=0040  train_loss=2.3053  train_acc=0.094  test_acc=0.086  sec/iter=7.94
# iter=0050  train_loss=2.3039  train_acc=0.086  test_acc=0.102  sec/iter=7.96
# iter=0060  train_loss=2.3055  train_acc=0.109  test_acc=0.102  sec/iter=8.06
# iter=0070  train_loss=2.3068  train_acc=0.102  test_acc=0.109  sec/iter=7.99
# iter=0080  train_loss=2.3041  train_acc=0.125  test_acc=0.102  sec/iter=8.01
# iter=0090  train_loss=2.3007  train_acc=0.180  test_acc=0.078  sec/iter=8.15
# iter=0100  train_loss=2.3028  train_acc=0.133  test_acc=0.094  sec/iter=8.04
# iter=0110  train_loss=2.3085  train_acc=0.109  test_acc=0.102  sec/iter=8.08
# iter=0120  train_loss=2.3129  train_acc=0.109  test_acc=0.102  sec/iter=8.13
# iter=0130  train_loss=2.3082  train_acc=0.102  test_acc=0.117  sec/iter=8.28
# iter=0140  train_loss=2.3022  train_acc=0.117  test_acc=0.070  sec/iter=8.17
# iter=0150  train_loss=2.3046  train_acc=0.117  test_acc=0.094  sec/iter=8.13
# iter=0160  train_loss=2.3054  train_acc=0.117  test_acc=0.094  sec/iter=8.17
# iter=0170  train_loss=2.3109  train_acc=0.109  test_acc=0.102  sec/iter=8.11
# iter=0180  train_loss=2.3059  train_acc=0.109  test_acc=0.102  sec/iter=8.10
# iter=0190  train_loss=2.3078  train_acc=0.109  test_acc=0.102  sec/iter=8.11
# iter=0200  train_loss=2.3071  train_acc=0.109  test_acc=0.102  sec/iter=8.08
# Done.

# (base) danieltorres@MacBook-Air-de-Daniel SNN-CIFAR10 % uv run python scripts/snn/diagnostics/es_subset_train_eval.py \
#   --iters 200 \
#   --eval-every 10 \
#   --train-samples 128 \
#   --test-samples 128 \
#   --batch-size 128 \
#   --num-workers 0 \
#   --T 8 \
#   --p-scale 3.0 \
#   --pop 16 \
#   --sigma 0.10 \
#   --lr 0.02 \
#   --max-update 1e-2 \
#   --deterministic-encoder
# device: cpu
# Collecting fixed train subset: 128 samples ...
# Collecting fixed test subset: 128 samples ...
# Using deterministic rate encoder (no spike sampling).
# iter=0001  train_loss=2.3055  train_acc=0.078  test_acc=0.141  sec/iter=6.99
# iter=0010  train_loss=2.2915  train_acc=0.133  test_acc=0.070  sec/iter=7.86
# iter=0020  train_loss=2.2857  train_acc=0.195  test_acc=0.062  sec/iter=7.88
# iter=0030  train_loss=2.2914  train_acc=0.133  test_acc=0.086  sec/iter=8.01
# iter=0040  train_loss=2.3032  train_acc=0.055  test_acc=0.141  sec/iter=7.99
# iter=0050  train_loss=2.3095  train_acc=0.094  test_acc=0.133  sec/iter=8.12
# iter=0060  train_loss=2.3421  train_acc=0.062  test_acc=0.078  sec/iter=8.11
# iter=0070  train_loss=2.3645  train_acc=0.070  test_acc=0.148  sec/iter=8.05
# iter=0080  train_loss=2.3690  train_acc=0.086  test_acc=0.133  sec/iter=8.12
# iter=0090  train_loss=2.3609  train_acc=0.078  test_acc=0.133  sec/iter=8.05
# iter=0100  train_loss=2.3808  train_acc=0.070  test_acc=0.133  sec/iter=8.21
# iter=0110  train_loss=2.3733  train_acc=0.078  test_acc=0.133  sec/iter=8.06
# iter=0120  train_loss=2.3855  train_acc=0.078  test_acc=0.133  sec/iter=8.17
# iter=0130  train_loss=2.3528  train_acc=0.078  test_acc=0.062  sec/iter=8.12
# iter=0140  train_loss=2.3706  train_acc=0.109  test_acc=0.125  sec/iter=8.24
# iter=0150  train_loss=2.3485  train_acc=0.117  test_acc=0.078  sec/iter=8.21
# iter=0160  train_loss=2.4184  train_acc=0.062  test_acc=0.078  sec/iter=8.30
# iter=0170  train_loss=2.4732  train_acc=0.078  test_acc=0.055  sec/iter=8.15
# iter=0180  train_loss=2.4782  train_acc=0.078  test_acc=0.078  sec/iter=8.17
# iter=0190  train_loss=2.5634  train_acc=0.117  test_acc=0.094  sec/iter=8.27
# iter=0200  train_loss=2.6750  train_acc=0.086  test_acc=0.156  sec/iter=8.20
# Done.