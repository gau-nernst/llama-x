import torch
import torch.nn.functional as F
from torch import Tensor, nn

aten = torch.ops.aten


def apply_linear_adapter_(model: nn.Module, adapter: str | None, **kwargs):
    if adapter is None:
        return

    cls = dict(lora=LoRALinear, dora=DoRALinear)[adapter]
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.__class__ = cls
            m.init_adapter(**kwargs)


class LoRALinear(nn.Linear):
    def init_adapter(self, rank: int = 8, alpha: float = 8.0) -> None:
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        self.rank = rank
        self.alpha = alpha
        self.scale = self.alpha / self.rank

        if rank > 0:
            dtype = self.weight.dtype
            self.lora_a = nn.Parameter(torch.empty(rank, self.in_features, dtype=dtype))
            self.lora_b = nn.Parameter(torch.empty(self.out_features, rank, dtype=dtype))

            nn.init.kaiming_normal_(self.lora_a, a=5**0.5)
            nn.init.zeros_(self.lora_b)

    def extra_repr(self):
        return f"{super().extra_repr()}, rank={self.rank}, alpha={self.alpha}"

    def forward(self, x: Tensor):
        out = F.linear(x, self.weight, self.bias)
        if self.rank > 0:
            out = out + x @ self.lora_a.T @ self.lora_b.T * self.scale
        return out


class DoRALinear(LoRALinear):
    def init_adapter(self, rank: int = 8, alpha: float = 8.0) -> None:
        super().init_adapter(rank, alpha)
        if self.rank > 0:
            self.m = nn.Parameter(self.weight.norm(p=2, dim=1))

    def forward(self, x: Tensor):
        out = F.linear(x, self.weight)
        if self.rank > 0:
            out = out + x @ self.lora_a.T @ self.lora_b.T * self.scale
            d_weight = self.lora_b.detach() @ self.lora_a.detach() * self.scale
            norm = (self.weight + d_weight).norm(p=2, dim=1)
            out = out * (self.m / norm)
        if self.bias is not None:
            out = out + self.bias
        return out
