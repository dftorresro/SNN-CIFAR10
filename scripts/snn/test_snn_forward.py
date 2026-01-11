import torch
import torch.nn.functional as F

from cifar_baseline.config import TrainConfig
from cifar_baseline.utils import get_device, set_seed
from cifar_baseline.data import make_loaders
from cifar_baseline.snn.snn_models import SpikingCifarCNN


def main():
    cfg = TrainConfig()
    cfg.num_workers = 0  # safe on the macbook cpu
    cfg.batch_size = 32  # small for a quick test on CPU

    set_seed(cfg.seed)
    device = get_device()
    print("device:", device)

    train_loader, _, _ = make_loaders(
        dataset_id  =   cfg.dataset_id,
        batch_size  =   cfg.batch_size,
        num_workers =   cfg.num_workers,
        device      =   device,
    )

    # SNN model
    model = SpikingCifarCNN(
        T                       =   16,
        p_scale                 =   1.0,
        input_is_normalized     =   True,  # CIFAR-10 images are normalized
        lif_decay               =   0.95,
        lif_th                  =   1.0,
        reset_mode              =   "to_reset",
        num_classes             =   10,
    ).to(device)

    x, y = next(iter(train_loader))
    x = x.to(device)
    y = y.to(device)

    with torch.no_grad():
        logits = model(x)
        loss = F.cross_entropy(logits, y)

    print("x:", tuple(x.shape), x.dtype)
    print("logits:", tuple(logits.shape), logits.dtype)
    print("loss:", float(loss.item()))


if __name__ == "__main__":
    main()
