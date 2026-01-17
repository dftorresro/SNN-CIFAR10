from __future__ import annotations

import torch
import torch.nn as nn

from cifar_baseline.snn.lif import LIF
from cifar_baseline.snn.encoding import RepeatValueEncoder


class SpikingMnistCNN(nn.Module):
    """
    Hybrid SNN for MNIST, designed to be smalll:
      - encoder returns float inputs in [0,1] (repeated each timestep)
      - conv sees floats
      - LIF after conv blocks creates spikes
      - logits are accumulated across time (rate-style readout)
    """

    def __init__(
        self,
        T: int = 8,
        input_scale: float = 1.0,
        lif_decay: float = 0.95,
        lif_v_th: float = 1.0,   # <-- rename
        num_classes: int = 10,
    ):
        super().__init__()
        self.T = int(T)

        self.encoder = RepeatValueEncoder(scale=input_scale, clamp_01=True)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.lif1 = LIF(decay=lif_decay, v_th=lif_v_th)   # <-- v_th
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.lif2 = LIF(decay=lif_decay, v_th=lif_v_th)   # <-- v_th
        self.pool2 = nn.MaxPool2d(2)

        self.fc = nn.Linear(64 * 7 * 7, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,1,28,28] float
        x_seq = self.encoder(x, T=self.T)  # [T,B,1,28,28] 
        mem1 = None
        mem2 = None

        out_acc = 0.0

        for t in range(self.T):
            xt = x_seq[t]  # [B,1,28,28]

            z1 = self.conv1(xt)
            if mem1 is None:
                mem1 = self.lif1.init_state(z1)
            s1, mem1 = self.lif1(z1, mem1)
            p1 = self.pool1(s1)

            z2 = self.conv2(p1)
            if mem2 is None:
                mem2 = self.lif2.init_state(z2)
            s2, mem2 = self.lif2(z2, mem2)
            p2 = self.pool2(s2)

            h = p2.flatten(1)
            out_acc = out_acc + self.fc(h)

        logits = out_acc / float(self.T)
        return logits