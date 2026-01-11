import platform
import torch

from cifar_baseline.config import TrainConfig
from cifar_baseline.utils import get_device, set_seed
from cifar_baseline.data import make_loaders
from cifar_baseline.snn.snn_models import SpikingCifarCNN
from cifar_baseline.snn.es import ESConfig, train_es


def main():
    set_seed(42)
    device = get_device()
    print("device:", device)

    # Data config (CPU-safe defaults)
    cfg = TrainConfig()
    cfg.batch_size = 64
    cfg.num_workers = 0  # macOS spawn + HF transforms 
    if platform.system() != "Darwin" and device.type == "cuda":
        cfg.num_workers = 2

    train_loader, test_loader, _ = make_loaders(
        dataset_id  =   cfg.dataset_id,
        batch_size  =   cfg.batch_size,
        num_workers =   cfg.num_workers,
        device      =   device,
    )

    # SNN model
    model = SpikingCifarCNN(
        T                       =   16,
        p_scale                 =   3.0,
        input_is_normalized     =   True,
        lif_decay               =   0.95,
        lif_th                  =   1.0,
        reset_mode              =   "to_reset",
        num_classes             =   10,
        use_bn                  =   False,
    ).to(device)

    # ES config 
    es_cfg              = ESConfig(
    population_size     =   8,
    sigma               =   0.05,       # slightly larger sigma is fine
    lr                  =   0.002,      
    iters               =   400,
    batches_per_fitness =   8,          # reduces noise (slxower but much more stable)
    use_amp             =   False,
    base_seed=1234,
)

    train_es(model, train_loader, es_cfg, device, print_every=10)


if __name__ == "__main__":
    main()
