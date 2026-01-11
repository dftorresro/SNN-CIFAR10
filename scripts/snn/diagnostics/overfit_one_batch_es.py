import argparse
import platform
import time
import torch
import torch.nn.functional as F

from cifar_baseline.config import TrainConfig
from cifar_baseline.data import make_loaders
from cifar_baseline.utils import get_device, set_seed
from cifar_baseline.snn.snn_models import SpikingCifarCNN
from cifar_baseline.snn.es import ESConfig, _set_eval_seed, _add_noise  # uses existing helpers


@torch.no_grad()
def minibatch_acc(model, x, y, device, seed: int) -> float:
    _set_eval_seed(seed, device)
    logits = model(x)
    pred = logits.argmax(dim=1)
    return float((pred == y).float().mean().item())


@torch.no_grad()
def minibatch_neg_ce(model, x, y, device, seed: int, use_amp: bool) -> float:
    _set_eval_seed(seed, device)
    if use_amp and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
    else:
        logits = model(x)
        loss = F.cross_entropy(logits, y)
    return float((-loss).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--pop", type=int, default=128)  # must be even
    ap.add_argument("--sigma", type=float, default=0.03)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--p-scale", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--print-every", type=int, default=10)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print("device:", device)

    cfg = TrainConfig()
    cfg.batch_size = args.batch_size
    cfg.num_workers = 4 if platform.system() != "Darwin" else 0

    train_loader, _, _ = make_loaders(
        dataset_id=cfg.dataset_id,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=device,
    )

    # one fixed batch
    x0, y0 = next(iter(train_loader))
    x0 = x0.to(device, non_blocking=True)
    y0 = y0.to(device, non_blocking=True)

    model = SpikingCifarCNN(
        T=args.T,
        p_scale=args.p_scale,
        input_is_normalized=True,
        lif_decay=0.95,
        lif_th=1.0,
        reset_mode="to_reset",
        num_classes=10,
        use_bn=False,
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]

    es_cfg = ESConfig(
        population_size=args.pop,
        sigma=args.sigma,
        lr=args.lr,
        iters=args.iters,
        batches_per_fitness=1,
        use_amp=True,
        base_seed=1234,
    )

    if es_cfg.population_size % 2 != 0:
        raise ValueError("--pop must be even")

    pairs = es_cfg.population_size // 2
    use_amp = bool(es_cfg.use_amp and device.type == "cuda")
    max_update = 1e-3  # same cap you used

    for it in range(1, es_cfg.iters + 1):
        t0 = time.perf_counter()

        with torch.no_grad():

            # base fitness on the fixed batch
            base_f = minibatch_neg_ce(model, x0, y0, device, seed=es_cfg.base_seed, use_amp=use_amp)
            base_acc = minibatch_acc(model, x0, y0, device, seed=es_cfg.base_seed)

            diffs = []
            eps_cache = []

            for k in range(pairs):
                eps_list = [torch.randn_like(p) for p in params]
                pair_seed = es_cfg.base_seed + 1_000_000 + k

                # +sigma
                _add_noise(params, eps_list, es_cfg.sigma)
                f_plus = minibatch_neg_ce(model, x0, y0, device, seed=pair_seed, use_amp=use_amp)

                # -sigma
                _add_noise(params, eps_list, -2.0 * es_cfg.sigma)
                f_minus = minibatch_neg_ce(model, x0, y0, device, seed=pair_seed, use_amp=use_amp)

                # restore
                _add_noise(params, eps_list, es_cfg.sigma)

                diffs.append(f_plus - f_minus)
                eps_cache.append(eps_list)

            d = torch.tensor(diffs, device=device, dtype=torch.float32)
            d = (d - d.mean()) / (d.std(unbiased=False) + 1e-8)

            grad_acc = [torch.zeros_like(p) for p in params]
            for di, eps_list in zip(d.tolist(), eps_cache):
                for g, e in zip(grad_acc, eps_list):
                    g.add_(float(di) * e)

            coef = es_cfg.lr / (2.0 * pairs * es_cfg.sigma)
            for p, g in zip(params, grad_acc):
                delta = (coef * g).clamp_(-max_update, max_update)
                p.add_(delta)

            dt = (time.perf_counter() - t0) * 1000.0

            if it == 1 or it % args.print_every == 0:
                print(f"iter={it:04d}  loss={-base_f:.4f}  acc={base_acc:.3f}  ms={dt:.1f}")

    print("Done.")


if __name__ == "__main__":
    main()

#     (base) danieltorres@MacBook-Air-de-Daniel SNN-CIFAR10 % uv run python scripts/snn/diagnostics/overfit_one_batch_es.py --iters 300 --pop 128 --sigma 0.03 --lr 0.01 --batch-size 256 --T 16 --p-scale 3.0
# device: cpu
# iter=0001  loss=2.2973  acc=0.137  ms=216128.4
# iter=0010  loss=2.2925  acc=0.141  ms=241489.1
# iter=0020  loss=2.2865  acc=0.141  ms=245172.7
# iter=0030  loss=2.2831  acc=0.148  ms=242817.2
# iter=0040  loss=2.2798  acc=0.141  ms=241816.5
# iter=0050  loss=2.2761  acc=0.152  ms=239019.6
# iter=0060  loss=2.2747  acc=0.148  ms=240810.2
# iter=0070  loss=2.2706  acc=0.152  ms=241360.2
# iter=0080  loss=2.2669  acc=0.156  ms=240786.6
# iter=0090  loss=2.2607  acc=0.160  ms=240440.0
# iter=0100  loss=2.2511  acc=0.164  ms=286733.1