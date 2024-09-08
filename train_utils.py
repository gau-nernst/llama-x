from torch import nn


def freeze_params(model: nn.Module, prefixes: list[str]):
    frozen_names = []
    for prefix in prefixes:
        for name, param in model.named_parameters():
            if name == prefix or name.startswith(f"{prefix}."):
                frozen_names.append(name)
                param.requires_grad_(False)

    if frozen_names:
        print("Freeze the following parameters:")
        for name in frozen_names:
            print(f"  - {name}")


def get_grad_norm(model: nn.Module):
    return sum(p.grad.square().sum().item() for p in model.parameters() if p.grad is not None) ** 0.5


def print_model_stats(model: nn.Module):
    print(f"No. of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"No. of non-trainable params: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
    print(f"No. of buffers: {sum(p.numel() for p in model.buffers()):,}")


class LRScheduler:
    def __init__(
        self,
        lr: float,
        n_steps: int,
        warmup: float,
        decay: float,
    ) -> None:
        self.t1 = int(n_steps * warmup)
        self.t2 = int(n_steps * (1 - decay))
        self.t3 = n_steps
        self.lr = lr

    def get_lr(self, step: int):
        if step < self.t1:
            return self.lr * step / self.t1
        if step < self.t2:
            return self.lr
        if step < self.t3:
            return self.lr * (self.t3 - step) / (self.t3 - self.t2)
        return self.lr
