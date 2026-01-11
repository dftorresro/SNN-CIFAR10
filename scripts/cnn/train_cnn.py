import torch

from cifar_baseline.config import TrainConfig
from cifar_baseline.utils import set_seed, get_device
from cifar_baseline.data import make_loaders
from cifar_baseline.models import SmallCifarCNN
from cifar_baseline.train import train_one_epoch, evaluate_accuracy, make_scheduler

def main():
    cfg = TrainConfig()

    set_seed(cfg.seed)
    device = get_device()
    print("device:", device)

    train_loader, test_loader, _ = make_loaders(
        dataset_id  =   cfg.dataset_id,
        batch_size  =   cfg.batch_size,
        num_workers =   cfg.num_workers,
        device      =   device,
    )

    model = SmallCifarCNN(num_classes=10, dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    use_amp = cfg.use_amp and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    print("AMP:", bool(scaler))

    scheduler = make_scheduler(optimizer, cfg.scheduler, cfg.epochs)

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, scaler=scaler, log_every=cfg.log_every
        )
        test_acc = evaluate_accuracy(model, test_loader, device)

        if scheduler is not None:
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
        else:
            lr = optimizer.param_groups[0]["lr"]

        print(f"epoch={epoch:02d} lr={lr:.2e} train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.3f} test_acc={test_acc:.3f}")

if __name__ == "__main__":
    main()
