from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset


class HFMnistTorchDataset(Dataset):
    """
    Wrap HuggingFace MNIST so DataLoader workers can pickle cleanly.
    HF returns items like {"image": PIL.Image, "label": int}.
    """

    def __init__(self, split: str, normalize_01: bool = True):
        super().__init__()
        ds = load_dataset("mnist", split=split)  # train / test
        self.ds = ds

        tfms = [transforms.ToTensor()]  # -> float32 in [0,1], shape [1,28,28]
        # If you want standard MNIST normalization later, you can add it here.
        # Pierre's suggested scheme is 0..1, so keep it simple by default.
        self.tfm = transforms.Compose(tfms) if normalize_01 else transforms.Compose(tfms)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        ex = self.ds[int(idx)]
        x = self.tfm(ex["image"])          # [1,28,28], float32
        y = int(ex["label"])
        return x, y


@dataclass
class MnistLoaderConfig:
    batch_size: int = 128
    num_workers: int = 0
    pin_memory: bool = True
    normalize_01: bool = True


def make_mnist_loaders(cfg: MnistLoaderConfig) -> Tuple[DataLoader, DataLoader]:
    train_ds = HFMnistTorchDataset(split="train", normalize_01=cfg.normalize_01)
    test_ds = HFMnistTorchDataset(split="test", normalize_01=cfg.normalize_01)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )
    return train_loader, test_loader