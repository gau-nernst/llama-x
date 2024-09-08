import torch
import torch.nn.functional as F
from torch import Tensor, nn

aten = torch.ops.aten


class LoRALinear(nn.Linear):
    def init_lora(self, rank: int = 8, alpha: float = 8.0) -> None:
        self.rank = rank
        self.alpha = alpha

        if rank > 0:
            dtype = self.weight.dtype
            self.lora_a = nn.Parameter(torch.empty(rank, self.in_features, dtype=dtype))
            self.lora_b = nn.Parameter(torch.empty(self.out_features, rank, dtype=dtype))

            nn.init.kaiming_normal_(self.lora_a, a=5**0.5)
            nn.init.zeros_(self.lora_b)

    def forward(self, x: Tensor):
        out = F.linear(x, self.weight, self.bias)
        if self.rank > 0:
            out = out + x @ self.lora_a.T @ self.lora_b.T * (self.alpha / self.rank)
        return out

    @staticmethod
    def convert_model(model: nn.Module, rank: int = 8, alpha: float = 8.0):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                m.__class__ = LoRALinear
                m.init_lora(rank, alpha)
        return model
