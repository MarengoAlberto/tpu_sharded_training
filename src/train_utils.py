import torch
import math


def build_cosine_warmup(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int = 1,
    min_lr_ratio: float = 0.05,  # final LR = base_lr * min_lr_ratio
):
    """
    Epoch-based scheduler: call scheduler.step() ONCE at the END of each epoch.
    Keeps LR constant within an epoch; updates between epochs only.
    """
    assert total_epochs > 0, "total_epochs must be > 0"
    warmup_epochs = max(0, int(warmup_epochs))
    remain = max(1, total_epochs - warmup_epochs)

    def lr_lambda(epoch_idx: int):
        # Linear warmup: from 1/warmup -> 1.0 of base LR
        if warmup_epochs > 0 and epoch_idx < warmup_epochs:
            return float(epoch_idx + 1) / float(warmup_epochs)

        # Cosine decay: from 1.0 -> min_lr_ratio over remaining epochs
        progress = (epoch_idx - warmup_epochs) / float(remain)
        progress = min(max(progress, 0.0), 1.0)  # clamp [0,1]
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    # Scales each param group's base LR by the returned factor (keeps group ratios intact)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_param_groups(model, wd=5e-4):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_norm = ("bn" in n.lower()) or ("norm" in n.lower())
        if is_norm or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay,    "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]
