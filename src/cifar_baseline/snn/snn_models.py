import torch
from torch import nn
import torch.nn.functional as F

from cifar_baseline.snn.lif import LIF
from cifar_baseline.snn.encoding import BernoulliRateEncoder


class SpikingCifarCNN(nn.Module):
    """
    Very simple spiking CNN for CIFAR-10:
      For t in 1..T:
        x_spk = encoder(x)
        h1 = LIF(conv1(x_spk))
        h1 = pool(h1_spk)
        h2 = LIF(conv2(h1))
        h2 = pool(h2_spk)
        logits_t = head(h2)
        logits_sum += logits_t
      logits = logits_sum / T
    """

    def __init__(
        self,
        T                   : int = 16,
        p_scale             : float = 1.0,
        input_is_normalized : bool = True,
        lif_decay           : float = 0.95,
        lif_th              : float = 1.0,
        reset_mode          : str = "to_reset",
        num_classes         : int = 10,
        use_bn              : bool = False,   
        repeat_input        : bool = False,
    ):
        super().__init__()
        self.T = int(T)
        self.repeat_input = repeat_input
        self.input_is_normalized = input_is_normalized

        # CIFAR-10 mean/std (torchvision); used for repeat-input mode
        mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32).view(1, 3, 1, 1)
        std  = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("cifar_mean", mean)
        self.register_buffer("cifar_std", std)

        self.encoder = BernoulliRateEncoder(p_scale=p_scale, input_is_normalized=input_is_normalized)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64) if use_bn else nn.Identity()
        self.lif1 = LIF(decay=lif_decay, v_th=lif_th, v_reset=0.0, reset_mode=reset_mode)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(128) if use_bn else nn.Identity()
        self.lif2 = LIF(decay=lif_decay, v_th=lif_th, v_reset=0.0, reset_mode=reset_mode)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, 32, 32]
        returns logits: [B, 10]
        """
        B = x.size(0)

        v1 = None
        v2 = None
        logits_sum = torch.zeros((B, 10), device=x.device, dtype=x.dtype)

        # If repeat_input=True: feed float inputs to conv1 every timestep 
        if self.repeat_input:
            x_in = x
            if self.input_is_normalized:
                # back to approx [0,1]
                x_in = x_in * self.cifar_std + self.cifar_mean
            x_in = x_in.clamp(0.0, 1.0)

        for _ in range(self.T):
            if not self.repeat_input:
                # Original path: sample spikes each timestep
                x_in = self.encoder(x)  # [B,3,32,32] spikes (0/1) or rates

            # First conv runs on x_in (float or spikes depending on mode), then LIF
            h1_cur = self.bn1(self.conv1(x_in))
            if v1 is None:
                v1 = self.lif1.init_state(h1_cur)
            h1_spk, v1 = self.lif1(h1_cur, v1)
            h1 = F.avg_pool2d(h1_spk, kernel_size=2)  # -> 16x16

            h2_cur = self.bn2(self.conv2(h1))
            if v2 is None:
                v2 = self.lif2.init_state(h2_cur)
            h2_spk, v2 = self.lif2(h2_cur, v2)
            h2 = F.avg_pool2d(h2_spk, kernel_size=2)  # -> 8x8

            logits_sum = logits_sum + self.head(h2)

        return logits_sum / float(self.T)