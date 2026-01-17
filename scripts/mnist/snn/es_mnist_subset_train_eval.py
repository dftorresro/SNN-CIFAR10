from __future__ import annotations

import argparse
import time
from typing import List, Tuple

import torch
import torch.nn.functional as F

from cifar_baseline.mnist_data import MnistLoaderConfig, make_mnist_loaders
from cifar_baseline.snn.mnist_snn_models import SpikingMnistCNN
from cifar_baseline.snn.es import (
    ESConfig,
    pack_params,
    unpack_params_into,
    sample_antithetic_noise,
)
from cifar_baseline.snn.es_parallel import es_update_step_parallel_chunked


@torch.no_grad()
def minibatch_acc(model, x, y):
    logits = model(x)
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def collect_fixed_subset(loader, n_samples: int, device) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    chunks = []
    seen = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        chunks.append((x, y))
        seen += x.shape[0]
        if seen >= n_samples:
            break
    return chunks


@torch.no_grad()
def eval_on_batches(model, batches):
    tot = 0
    correct = 0
    for x, y in batches:
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        tot += y.numel()
    return correct / max(1, tot)


@torch.no_grad()
def loss_on_batches(model, batches, use_amp: bool):
    losses = []
    for x, y in batches:
        if use_amp and x.is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
        else:
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        losses.append(loss.detach())
    return torch.stack(losses).mean().item()


def es_update_step_sequential(model, params_flat, train_batches, cfg: ESConfig, use_amp: bool):
    base_loss = loss_on_batches(model, train_batches, use_amp=use_amp)
    base_fitness = -base_loss

    noises = sample_antithetic_noise(params_flat, cfg.population_size, cfg.seed)

    losses_plus = []
    losses_minus = []

    for i in range(cfg.population_size // 2):
        eps = noises[i]

        # +eps
        p_plus = params_flat + cfg.sigma * eps
        unpack_params_into(model, p_plus)
        losses_plus.append(loss_on_batches(model, train_batches, use_amp=use_amp))

        # -eps
        p_minus = params_flat - cfg.sigma * eps
        unpack_params_into(model, p_minus)
        losses_minus.append(loss_on_batches(model, train_batches, use_amp=use_amp))

    # restore base params
    unpack_params_into(model, params_flat)

    losses_plus_t = torch.tensor(losses_plus, device=params_flat.device, dtype=torch.float32)
    losses_minus_t = torch.tensor(losses_minus, device=params_flat.device, dtype=torch.float32)

    fitness_plus = -losses_plus_t
    fitness_minus = -losses_minus_t

    # rank-normalize (simple)
    all_fit = torch.cat([fitness_plus, fitness_minus], dim=0)
    ranks = all_fit.argsort().argsort().float()
    ranks = (ranks - ranks.mean()) / (ranks.std() + 1e-8)

    # split back
    r_plus = ranks[: fitness_plus.numel()]
    r_minus = ranks[fitness_plus.numel() :]

    # gradient estimate
    grad = torch.zeros_like(params_flat)
    for i in range(cfg.population_size // 2):
        grad += (r_plus[i] - r_minus[i]) * noises[i]
    grad /= (cfg.population_size * cfg.sigma)

    # update with clip
    update = cfg.lr * grad
    if cfg.max_update is not None:
        update = update.clamp(-cfg.max_update, cfg.max_update)

    params_flat = params_flat + update
    unpack_params_into(model, params_flat)

    return base_loss, params_flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--parallel-chunk", type=int, default=0)

    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--train-samples", type=int, default=1024)
    ap.add_argument("--test-samples", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=0)

    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--input-scale", type=float, default=1.0)

    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--sigma", type=float, default=0.03)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--max-update", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Dataa
    dl_cfg = MnistLoaderConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        normalize_01=True,
    )
    train_loader, test_loader = make_mnist_loaders(dl_cfg)

    print(f"Collecting fixed train subset: {args.train_samples} samples ...")
    train_batches = collect_fixed_subset(train_loader, args.train_samples, device)
    print(f"Collecting fixed test subset: {args.test_samples} samples ...")
    test_batches = collect_fixed_subset(test_loader, args.test_samples, device)

    # Model
    model = SpikingMnistCNN(T=args.T, input_scale=args.input_scale).to(device)
    model.eval()

    params = pack_params(model)

    es_cfg = ESConfig(
        iters=args.iters,
        population_size=args.pop,
        sigma=args.sigma,
        lr=args.lr,
        max_update=args.max_update,
        seed=args.seed,
        batches_per_fitness=len(train_batches),
        use_amp=args.amp,
    )

    t0 = time.time()
    for it in range(1, args.iters + 1):
        tic = time.time()

        if args.parallel_chunk > 0 and device.type == "cuda":
            params, base_loss = es_update_step_parallel_chunked(
                model=model,
                params=params,
                cfg=es_cfg,
                train_batches=train_batches,
                chunk_size=args.parallel_chunk,
            )
        else:
            base_loss, params = es_update_step_sequential(
                model=model,
                params_flat=params,
                train_batches=train_batches,
                cfg=es_cfg,
                use_amp=args.amp,
            )

        sec = time.time() - tic

        if it == 1 or it % args.eval_every == 0 or it == args.iters:
            train_loss = loss_on_batches(model, train_batches, use_amp=args.amp)
            train_acc = eval_on_batches(model, train_batches)
            test_acc = eval_on_batches(model, test_batches)
            print(
                f"iter={it:04d}  train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  test_acc={test_acc:.3f}  sec/iter={sec:.2f}"
            )

    print(f"Done. total_min={(time.time()-t0)/60.0:.2f}")


if __name__ == "__main__":
    main()