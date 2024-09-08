from torch import nn

from .int8 import Int8LinearWeight


def quantize_linear_(model: nn.Module, quantize: str | None, **kwargs):
    if quantize is None:
        return

    fn = dict(int8=Int8LinearWeight)[quantize]
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.weight = nn.Parameter(fn(m.weight.detach(), **kwargs), requires_grad=False)
