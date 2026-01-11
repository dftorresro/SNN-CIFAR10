import torch
from torch import nn


class LIF(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) neuron layer.
      v <- decay * v + input_current
      s <- 1 if v >= v_th else 0
      v <- v_reset if s==1 else v   (or subtract threshold)

    This version uses hard threshold (non-differentiable) which is OK for ES training.
    """

    def __init__(self, decay: float = 0.95, v_th: float = 1.0, v_reset: float = 0.0, reset_mode: str = "to_reset"):
        """
          - "to_reset": v = v_reset when spike
          - "subtract": v = v - v_th when spike (keeps residual)
        """
        super().__init__()
        if not (0.0 < decay <= 1.0):
            raise ValueError("decay must be in (0, 1].")
        if reset_mode not in ("to_reset", "subtract"):
            raise ValueError("reset_mode must be 'to_reset' or 'subtract'.")

        self.decay = float(decay)
        self.v_th = float(v_th)
        self.v_reset = float(v_reset)
        self.reset_mode = reset_mode

    def init_state(self, like: torch.Tensor) -> torch.Tensor:
        """Return a zero membrane state with the same shape/device/dtype."""
        return torch.zeros_like(like)

    def forward(self, input_current: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          input_current: tensor of any shape
          v: membrane state tensor, same shape as input_current

        Returns:
          s: spike tensor (0/1 float) same shape
          v_new: updated membrane
        """
        v = self.decay * v + input_current
        s = (v >= self.v_th).to(input_current.dtype)

        if self.reset_mode == "to_reset":
            v_new = torch.where(s.bool(), torch.full_like(v, self.v_reset), v)
        else:  # subtract
            v_new = v - s * self.v_th

        return s, v_new
