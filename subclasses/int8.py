import torch
import torch.nn.functional as F
from torch import Tensor

aten = torch.ops.aten


def quantize_int8_rowwise(x: Tensor):
    dtype = x.dtype
    x = x.float()
    scale = x.abs().amax(1) / 127
    x = x / scale.clip(1e-12).view(-1, 1)
    x = x.round().to(torch.int8)
    return x, scale.to(dtype)


class Int8LinearWeight(Tensor):
    @staticmethod
    @torch._dynamo.disable
    def __new__(cls, int_data: Tensor, scale: Tensor, dynamic_int8_act: bool = False):
        return Tensor._make_wrapper_subclass(
            cls,
            int_data.shape,
            dtype=scale.dtype,
            device=int_data.device,
        )

    @torch._dynamo.disable
    def __init__(self, int_data: Tensor, scale: Tensor, dynamic_int8_act: bool = False):
        assert int_data.dtype is torch.int8
        assert int_data.ndim == 2
        assert scale.ndim == 2
        self.int_data = int_data
        self.scale = scale
        self.dynamic_int8_act = dynamic_int8_act

    def __tensor_flatten__(self):
        return ["int_data", "scale"], [self.dynamic_int8_act]

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None):
        return cls(tensor_data_dict["int_data"], tensor_data_dict["scale"], *tensor_attributes)

    @classmethod
    def from_float(cls, tensor: Tensor, int8_mm_forward: bool = False):
        return cls(*quantize_int8_rowwise(tensor), int8_mm_forward)

    def dequantize(self):
        return self.int_data * self.scale.view(-1, 1)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(shape={tuple(self.shape)}, config={self.config}, "
            f"dtype={self.dtype}, device={self.device}, requires_grad={self.requires_grad})"
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or dict()

        if func is F.linear:
            return _Int8Linear.apply(*args, **kwargs)

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if func in (aten.detach.default, aten.clone.default):
            return cls(
                func(args[0].int_data, *args[1:], **kwargs),
                func(args[0].scale, *args[1:], **kwargs),
                args[0].config,
            )

        elif func in (aten._to_copy.default,):
            device = kwargs.get("device", None)
            dtype = kwargs.get("dtype", None)
            return cls(
                args[0].int_data.to(device=device),
                args[0].scale.to(device=device, dtype=dtype),
                args[0].config,
            )

        elif func is aten.copy_.default:
            if isinstance(args[0], cls) and isinstance(args[1], cls):
                args[0].int_data.copy_(args[1].int_data)
                args[0].scale.copy_(args[1].scale)

            elif isinstance(args[0], cls):
                int_data, scale = quantize_int8_rowwise(args[1])
                args[0].int_data.copy_(int_data)
                args[0].scale.copy_(scale)

            else:
                args[0].copy_(args[1].dequantize())

            return args[0]

        raise NotImplementedError(f"{cls.__name__} dispatch: attempting to run {func}, this is not supported")


class _Int8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: Int8LinearWeight):
        ctx.save_for_backward(weight.int_data, weight.scale)

        if weight.dynamic_int8_act:
            input_i8, input_scale = quantize_int8_rowwise(input.view(-1, weight.shape[1]))
            out = torch._int_mm(input_i8, weight.int_data.T) * input_scale.view(-1, 1) * weight.scale
            out = out.view(*input.shape[:-1], -1)

        else:
            # NOTE: we have to .T before .to(input.dtype) for torch.compile() mixed matmul to work
            out = (input @ weight.int_data.T.to(input.dtype)) * weight.scale

        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        weight_i8, weight_scale = ctx.saved_tensors
        grad_input = (grad_output * weight_scale) @ weight_i8.to(grad_output.dtype)
        return grad_input, None
