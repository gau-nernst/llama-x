import torch
from torch import Tensor, nn
from torch._higher_order_ops.out_dtype import out_dtype

aten = torch.ops.aten


class Int8LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 8.0, quantize_act: bool = False) -> None:
        super().__init__()
        assert linear.bias is None
        self.rank = rank
        self.alpha = alpha
        self.quantize_act = quantize_act

        weight_i8, weight_scale = _quantize_int8(linear.weight.detach())
        self.register_buffer("weight_i8", weight_i8, persistent=False)
        self.register_buffer("weight_scale", weight_scale, persistent=False)

        if rank > 0:
            dtype = linear.weight.dtype
            self.lora_a = nn.Parameter(torch.empty(rank, linear.in_features, dtype=dtype))
            self.lora_b = nn.Parameter(torch.empty(linear.out_features, rank, dtype=dtype))

            nn.init.kaiming_normal_(self.lora_a, a=5**0.5)
            nn.init.zeros_(self.lora_b)

    def forward(self, x: Tensor):
        out = _Int8WeightOnlyLinear.apply(x, self.weight_i8, self.weight_scale, self.quantize_act)
        if self.rank > 0:
            out = out + x @ self.lora_a.T @ self.lora_b.T * (self.alpha / self.rank)
        return out

    @staticmethod
    def convert_model(model: nn.Module, rank: int = 8, alpha: float = 8.0, quantize_act: bool = False):
        if isinstance(model, nn.Linear):
            return Int8LoRALinear(model, rank, alpha, quantize_act)
        for name, child in model.named_children():
            setattr(model, name, Int8LoRALinear.convert_model(child, rank, alpha, quantize_act))
        return model


# @torch.compile(dynamic=True)
def _quantize_int8(x: Tensor):
    dtype = x.dtype
    x = x.float()
    scale = x.abs().amax(1) / 127
    x = x / scale.clip(1e-12).view(-1, 1)
    x = x.round().to(torch.int8)
    return x, scale.to(dtype)


def _int8_mm(A: Tensor, B: Tensor):
    return out_dtype(aten.mm.default, torch.int32, A, B)


class _Int8WeightOnlyLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight_i8: Tensor, weight_scale: Tensor, quantize_act: bool):
        ctx.save_for_backward(weight_i8, weight_scale)

        if quantize_act:
            _input = input.view(-1, weight_i8.shape[1])
            input_i8, input_scale = _quantize_int8(_input)
            out_i32 = _int8_mm(input_i8, weight_i8.T)
            out = out_i32 * input_scale.view(-1, 1) * weight_scale
            out = out.view(*input.shape[:-1], weight_i8.shape[0])

        else:
            # NOTE: we have to .T before .to(input.dtype) for torch.compile() mixed matmul to work
            out = (input @ weight_i8.T.to(input.dtype)) * weight_scale

        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        weight_i8, weight_scale = ctx.saved_tensors
        grad_input = (grad_output * weight_scale) @ weight_i8.to(grad_output.dtype)
        return grad_input, None, None
