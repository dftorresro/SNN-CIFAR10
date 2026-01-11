import torch
import torch.nn.functional as F

# Utility functions for training and evaluation ---------------------
def make_scheduler(optimizer, kind: str, epochs: int):
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if kind == "none":
        return None
    raise ValueError(f"Unknown scheduler: {kind}")

@torch.no_grad()
def evaluate_accuracy(model, loader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total

def train_one_epoch(model, loader, optimizer, device: torch.device, scaler=None, log_every: int = 100):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0

    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is None:
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
        else:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

        if log_every and (step % log_every == 0):
            print(f"  step={step:04d} loss={loss.item():.4f}")

    return total_loss / total, total_correct / total
