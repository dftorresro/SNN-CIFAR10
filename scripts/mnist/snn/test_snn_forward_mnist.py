import torch

from cifar_baseline.mnist_data import MnistLoaderConfig, make_mnist_loaders
from cifar_baseline.snn.mnist_snn_models import SpikingMnistCNN


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    cfg = MnistLoaderConfig(batch_size=32, num_workers=0, pin_memory=(device.type == "cuda"))
    train_loader, _ = make_mnist_loaders(cfg)

    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)

    model = SpikingMnistCNN(T=8, input_scale=1.0).to(device)
    logits = model(x)

    loss = torch.nn.functional.cross_entropy(logits, y)

    print("x:", tuple(x.shape), x.dtype)
    print("logits:", tuple(logits.shape), logits.dtype)
    print("loss:", float(loss))


if __name__ == "__main__":
    main()