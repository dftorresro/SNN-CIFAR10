from datasets import load_dataset
from dataclasses import dataclass
from torchvision import transforms
import torch
from torch.utils.data import DataLoader

# CIFAR-10 normalization constants (from torchvision.datasets.CIFAR10)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

def make_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


@dataclass
class HFImageToTensor:
    tfm: object  # torchvision transform

    def __call__(self, batch):
        batch["pixel_values"] = [self.tfm(img) for img in batch["img"]]
        return batch

def collate_fn(batch):
    x = torch.stack([b["pixel_values"] for b in batch], dim=0)
    y = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return x, y

def make_loaders(dataset_id: str, batch_size: int, num_workers: int, device: torch.device):
    ds = load_dataset(dataset_id)

    train_tfm = make_transforms(train=True)
    test_tfm = make_transforms(train=False)

    train_ds = ds["train"]
    test_ds  = ds["test"]

    train_ds.set_transform(HFImageToTensor(train_tfm))
    test_ds.set_transform(HFImageToTensor(test_tfm))

    pin = (device.type == "cuda")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, collate_fn=collate_fn
    )

    label_names = ds["train"].features["label"].names
    return train_loader, test_loader, label_names

