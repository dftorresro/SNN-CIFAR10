import torch
from torch import nn

# CIFAR- 10 normalization constants (from torchvision.datasets.CIFAR10)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def denormalize_cifar10(x: torch.Tensor) -> torch.Tensor:
    """
    If x was normalized with (mean, std), map it back to [0,1] approximately:
      x_denorm = x * std + mean
    """
    mean = torch.tensor(CIFAR10_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return x * std + mean


class BernoulliRateEncoder(nn.Module):
    """
    Rate encoder: converts an image tensor x into a spike tensor s_t by sampling:
      s_t = Bernoulli(p)

    Where p is derived from pixel intensity. (Poisson coding).

      - It generates spikes per time step.
      - It does NOT prebuild a [B, T, C, H, W] tensor (saves memory).
    """

    def __init__(self, p_scale: float = 1.0, input_is_normalized: bool = True):
        """
        p_scale:
          scales pixel intensity into probability. After scaling we clamp to [0,1].
          Example: p = clamp(p_scale * x, 0, 1)
          - if input_is_normalized we denormalize using CIFAR10 mean/std before clamping.
        """
        super().__init__()
        self.p_scale = float(p_scale)
        self.input_is_normalized = bool(input_is_normalized)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, 32, 32], either normalized or raw in [0,1]
        returns spikes of the same shape.
        """
        if self.input_is_normalized:
            x = denormalize_cifar10(x)

        # Ensure probabilities in [0,1]
        p = (self.p_scale * x).clamp_(0.0, 1.0)

        # Bernoulli sampling
        spikes = (torch.rand_like(p) < p).to(x.dtype)
        return spikes

class RepeatValueEncoder(torch.nn.Module):
    """
    Deterministic encoding:
      - inputs in [0,1]
      - repeat across time (we do it outside by calling encoder each step)
      - first conv sees float values (not spikes)
      - LIF sits on top of conv
    """

    def __init__(self, scale: float = 1.0, clamp_01: bool = True):
        super().__init__()
        self.scale = float(scale)
        self.clamp_01 = bool(clamp_01)

    def forward(self, x: torch.Tensor, T: int) -> torch.Tensor:
        # normalize to [0,1] if desired
        p = x * self.scale
        if self.clamp_01:
            p = p.clamp(0.0, 1.0)

        # repeat on time axis
        # output: [T, B, C, H, W]
        return p.unsqueeze(0).repeat(int(T), 1, 1, 1, 1)