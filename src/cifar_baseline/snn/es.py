from __future__ import annotations 

from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class ESConfig:
    population_size         : int = 8   # even
    sigma                   : float = 0.05
    lr                      : float = 0.003
    iters                   : int = 200
    batches_per_fitness     : int = 4
    use_amp                 : bool = True
    base_seed               : int = 1234      # for deterministic stochastic encoding


def _iter_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]

@torch.no_grad()
def minibatch_acc(model, x, y, device, seed: int):
    _set_eval_seed(seed, device)  # so encoding randomness is consistent
    logits = model(x)
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


@torch.no_grad()
def _add_noise(params: list[torch.nn.Parameter], eps_list: list[torch.Tensor], scale: float) -> None:
    for p, e in zip(params, eps_list):
        p.add_(scale * e) # This noise is just an in-place addition


def _set_eval_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def fitness_neg_ce(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device, use_amp: bool, seed: int) -> float:
    # Make stochastic encoding identical across +eps and -eps for a given seed
    _set_eval_seed(seed, device)

    if use_amp and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
    else:
        logits = model(x)
        loss = F.cross_entropy(logits, y)

    return float((-loss).item())


def _get_batches(loader_iter, loader, device: torch.device, k: int):
    batches = []
    for _ in range(k):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        batches.append((x, y))
    return batches, loader_iter


@torch.no_grad()
def es_step(model, params, loader, loader_iter, cfg: ESConfig, device: torch.device):
    if cfg.population_size % 2 != 0:
        raise ValueError("population_size must be even.")
    pairs = cfg.population_size // 2
    use_amp = bool(cfg.use_amp and device.type == "cuda")

    batches, loader_iter = _get_batches(loader_iter, loader, device, cfg.batches_per_fitness)

    model.eval()

    # Base fitness (for logging) — use a fixed seed per iteration
    base_seed = cfg.base_seed
    base_vals = []
    for bi, (x, y) in enumerate(batches):
        seed = base_seed + 10_000 * bi
        base_vals.append(fitness_neg_ce(model, x, y, device, use_amp, seed))
    base_f = float(sum(base_vals) / len(base_vals))

    base_acc_vals = []
    for bi, (x, y) in enumerate(batches):
        seed = base_seed + 10_000 * bi
        base_acc_vals.append(minibatch_acc(model, x, y, device, seed))
    base_acc = float(sum(base_acc_vals) / len(base_acc_vals))

    # Store diffs and noise; normalize diffs before applying update
    diffs: list[float] = []
    eps_cache: list[list[torch.Tensor]] = []

    for k in range(pairs):
        eps_list = [torch.randn_like(p) for p in params]

        # SAME seed for + and - for this pair (makes encoding noise cancel )
        pair_seed = cfg.base_seed + 1_000_000 + k

        # +sigma
        _add_noise(params, eps_list, cfg.sigma)
        fp_vals = []
        for bi, (x, y) in enumerate(batches):
            seed = pair_seed + 10_000 * bi
            fp_vals.append(fitness_neg_ce(model, x, y, device, use_amp, seed))
        f_plus = float(sum(fp_vals) / len(fp_vals))

        # -sigma (from +sigma to -sigma is -2*sigma)
        _add_noise(params, eps_list, -2.0 * cfg.sigma)
        fm_vals = []
        for bi, (x, y) in enumerate(batches):
            seed = pair_seed + 10_000 * bi
            fm_vals.append(fitness_neg_ce(model, x, y, device, use_amp, seed))
        f_minus = float(sum(fm_vals) / len(fm_vals))

        # restore to theta
        _add_noise(params, eps_list, cfg.sigma)

        diffs.append(f_plus - f_minus)
        eps_cache.append(eps_list)

    # Normalize diffs to stabilize step size
    d = torch.tensor(diffs, device=device, dtype=torch.float32)
    d = (d - d.mean()) / (d.std(unbiased=False) + 1e-8)

    grad_acc = [torch.zeros_like(p) for p in params]
    for di, eps_list in zip(d.tolist(), eps_cache):
        for g, e in zip(grad_acc, eps_list):
            g.add_(float(di) * e)


    # Antithetic ES estimator
    coef = cfg.lr / (2.0 * pairs * cfg.sigma)

    max_update = 1e-3  # per-parameter elementwise cap to avoid large jumps

    for p, g in zip(params, grad_acc):
        delta = coef * g
        delta = delta.clamp_(-max_update, max_update)
        p.add_(delta)

    mean_fp = float((d + 0).mean().item())  # NOT MEANINGFULL NOW; keep logging simple
    mean_fm = mean_fp
    return base_f, base_acc, loader_iter


def train_es(model, loader, cfg: ESConfig, device: torch.device, *, print_every: int = 10) -> None:
    params = _iter_params(model)
    loader_iter = iter(loader)

    for it in range(1, cfg.iters + 1):
        base_f, base_acc, loader_iter = es_step(model, params, loader, loader_iter, cfg, device)

        if it % print_every == 0 or it == 1:
            print(f"iter={it:04d}  base_loss={-base_f:.4f}  base_acc={base_acc:.3f}")
