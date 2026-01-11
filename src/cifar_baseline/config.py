from dataclasses import dataclass

@dataclass
class TrainConfig:
    dataset_id          : str = "cifar10"     
    batch_size          : int = 128
    num_workers         : int = 0             # For CPU training need to be 0
    epochs              : int = 5

    lr                  : float = 3e-4
    weight_decay        : float = 1e-4

    use_amp             : bool = True         # auto-off on CPU
    scheduler           : str = "cosine"      # "cosine" or "none"

    seed                : int = 42
    log_every           : int = 100           # steps