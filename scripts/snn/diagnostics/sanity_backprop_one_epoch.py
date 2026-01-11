import argparse
import platform
import torch
import torch.nn.functional as F

from cifar_baseline.config import TrainConfig
from cifar_baseline.data import make_loaders
from cifar_baseline.utils import get_device, set_seed
from cifar_baseline.snn.snn_models import SpikingCifarCNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--p-scale", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
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

    model = SpikingCifarCNN(
        T                       =   args.T,
        p_scale                 =   args.p_scale,
        input_is_normalized     =   True,
        lif_decay               =   0.95,
        lif_th                  =   1.0,
        reset_mode              =   "to_reset",
        num_classes             =   10,
        use_bn                  =   False,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for step, (x, y) in enumerate(train_loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 20 == 0:
            acc = (logits.argmax(dim=1) == y).float().mean().item()
            print(f"step={step:04d} loss={loss.item():.4f} acc={acc:.3f}")

        if step >= args.steps:
            break

    print("Done.")


if __name__ == "__main__":
    main()

# (base) danieltorres@MacBook-Air-de-Daniel SNN-CIFAR10 % uv run python scripts/snn/diagnostics/sanity_backprop_one_epoch.py --steps 200 --batch-size 128 --lr 1e-3 --T 8 --p-scale 3.0
# device: cpu
# step=0020 loss=2.2409 acc=0.172
# step=0040 loss=2.2040 acc=0.172
# step=0060 loss=2.1186 acc=0.180
# step=0080 loss=2.0754 acc=0.258
# step=0100 loss=2.0435 acc=0.227
# step=0120 loss=2.0114 acc=0.227
# step=0140 loss=2.0519 acc=0.242
# step=0160 loss=2.0236 acc=0.242
# step=0180 loss=2.0224 acc=0.289
# step=0200 loss=1.9191 acc=0.336
# Done.