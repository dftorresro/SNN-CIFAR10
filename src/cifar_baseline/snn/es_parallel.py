# src/cifar_baseline/snn/es_parallel.py
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

# Try to import from torch.func (PyTorch 2.0+), else fall back to functorch
try:
    from torch.func import functional_call, vmap
except Exception:
    from functorch import vmap  
    from torch.func import functional_call  


Batch = Tuple[torch.Tensor, torch.Tensor]  # (x, y)

# Helpers ------------------------------------------------------------------
@torch.no_grad()
def _losses_for_params_batched(
    model:          torch.nn.Module,
    params_batched: Dict[str, torch.Tensor],  # each tensor has leading dim B
    buffers:        Dict[str, torch.Tensor],
    batches:        List[Batch],
    use_amp:        bool,
) -> torch.Tensor:
    """
    Returns losses of shape (B,) averaged across the provided batches.
    Uses vmap to evaluate many parameter sets in parallel on a single GPU.
    """

    def loss_one(params_one: Dict[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if use_amp and x.is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = functional_call(model, (params_one, buffers), (x,))
                return F.cross_entropy(logits, y)
        logits = functional_call(model, (params_one, buffers), (x,))
        return F.cross_entropy(logits, y)

    losses = None
    for (x, y) in batches:
        batch_losses = vmap(loss_one, in_dims=(0, None, None))(params_batched, x, y)  # (B,)
        losses = batch_losses if losses is None else (losses + batch_losses)

    return losses / float(len(batches))


@torch.no_grad()
def es_update_step_parallel_chunked(
    model:           torch.nn.Module,
    train_batches:   List[Batch],
    *,
    population_size: int,
    sigma:           float,
    lr:              float,
    max_update:      float,
    chunk_size:      int,
    use_amp:         bool,
    seed:            int = 1234,
) -> None:
    """
    Antithetic ES update where population evaluation is parallelized on ONE GPU via vmap,
    processed in chunks to control VRAM usage.
    - Uses fitness = -losss
    - No global std normalization (to avoid storing all diffs). So, just mean-center per chunk.
    """

    if population_size % 2 != 0:
        raise ValueError("population_size must be even.") # avoid unpaired samples
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0.")

    model.eval()

    base_params = {k: v for (k, v) in model.named_parameters()}
    buffers = {k: v for (k, v) in model.named_buffers()}

    half = population_size // 2
    coef = lr / (2.0 * half * sigma)

    grad_sum: Dict[str, torch.Tensor] = {k: torch.zeros_like(p) for k, p in base_params.items()}

    device = next(model.parameters()).device
    gen = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    gen.manual_seed(seed)

    start = 0
    while start < half:
        m = min(chunk_size, half - start)

        # Sample eps for this chunk
        eps_chunk: Dict[str, torch.Tensor] = {}
        for k, p in base_params.items():
            eps_chunk[k] = torch.randn((m,) + tuple(p.shape), device=p.device, dtype=p.dtype, generator=gen)

        # Build + and - params
        params_plus = {k: base_params[k].unsqueeze(0) + sigma * eps_chunk[k] for k in base_params.keys()}
        params_minus = {k: base_params[k].unsqueeze(0) - sigma * eps_chunk[k] for k in base_params.keys()}

        # Concatenate (+ then -) so we do one vmap pass
        params_cat = {k: torch.cat([params_plus[k], params_minus[k]], dim=0) for k in base_params.keys()}  # (2m, ...)

        # Candidate losses
        losses_cat = _losses_for_params_batched(model, params_cat, buffers, train_batches, use_amp=use_amp)  # (2m,)
        loss_plus = losses_cat[:m]
        loss_minus = losses_cat[m:]

        # fitness diff = f_plus - f_minus, with f=-loss => diff = loss_minus - loss_plus
        diff = (loss_minus - loss_plus)
        diff = diff - diff.mean()  # cheap variance reduction

        # Accumulate grad: sum of diff_i * eps_i
        for k, p in base_params.items():
            view = (m,) + (1,) * p.dim()
            grad_sum[k].add_((diff.view(view) * eps_chunk[k]).sum(dim=0))

        start += m

    # Apply update
    for k, p in base_params.items():
        delta = (coef * grad_sum[k]).clamp_(-max_update, max_update)
        p.add_(delta)

# (base) danieltorres@MacBook-Air-de-Daniel SNN-CIFAR10 % uv run python scripts/snn/diagnostics/es_subset_train_eval.py \
#   --iters 1 \
#   --eval-every 1 \
#   --train-samples 64 \
#   --test-samples 64 \
#   --batch-size 64 \
#   --num-workers 0 \
#   --T 4 \
#   --p-scale 3.0 \
#   --pop 8 \
#   --sigma 0.05 \
#   --lr 0.01 \
#   --max-update 3e-3 \
#   --parallel-chunk 8
# device: cpu
# Collecting fixed train subset: 64 samples ...
# Collecting fixed test subset: 64 samples ...
# iter=0001  train_loss=2.3010  train_acc=0.078  test_acc=0.094  sec/iter=1.00
# Done.