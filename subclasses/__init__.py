from functools import partial

from torch import nn
from torchao.prototype.quantized_training import (
    Int8MixedPrecisionTrainingConfig,
    Int8MixedPrecisionTrainingLinearWeight,
)

from .int8 import Int8LinearWeight


def quantize_linear_(model: nn.Module, quantize: str | None, requires_grad: bool = False, **kwargs):
    if quantize is None:
        return

    fn = dict(
        int8=Int8LinearWeight.from_float,
        int8mp=partial(Int8MixedPrecisionTrainingLinearWeight, config=Int8MixedPrecisionTrainingConfig()),
    )[quantize]
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.weight = nn.Parameter(fn(m.weight.detach(), **kwargs), requires_grad=requires_grad)
