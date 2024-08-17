# code in this file has been adapted from
# https://github.com/pytorch/torchtune
# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py

from typing import NamedTuple

import safetensors
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


class Llama3_1Config(NamedTuple):
    embed_dim: int
    num_layers: int
    head_dim: int
    num_heads: int
    num_kv_heads: int
    intermediate_dim: int
    max_seq_len: int = 2048
    vocab_size: int = 128256
    attn_dropout: float = 0.0
    rope_base: int = 50_000
    activation_checkpointing: bool = False


class Llama3ScaledRoPE(nn.Module):
    def __init__(self, config: Llama3_1Config) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_base = config.rope_base
        self.max_seq_len = config.max_seq_len
        self.is_cache_built = False

    def reset_parameters(self):
        self._rope_init()

    def _rope_init(self):
        freqs = 1.0 / (
            self.rope_base ** (torch.arange(0, self.head_dim, 2)[: (self.head_dim // 2)].float() / self.head_dim)
        )
        theta = self.apply_scaling(freqs)
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)
        self.is_cache_built = True

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        seq_idx = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def apply_scaling(self, freqs: torch.Tensor):
        scale_factor = 8
        low_freq_factor = 1
        high_freq_factor = 4
        old_context_len = 8192

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        new_freqs = []
        for freq in freqs:
            wavelen = 2 * torch.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / scale_factor)
            else:
                assert low_freq_wavelen != high_freq_wavelen
                smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
        return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

    def forward(self, x: Tensor, *, input_pos: Tensor | None = None) -> Tensor:
        if not self.is_cache_built:
            with torch.device(x.device):
                self._rope_init()

        seq_len = x.size(1)
        rope_cache = self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        x_out = x_out.flatten(3)
        return x_out.type_as(x)


class Attention(nn.Module):
    def __init__(self, config: Llama3_1Config) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.embed_dim = config.embed_dim
        self.attn_dropout = config.attn_dropout
        self.head_dim = config.head_dim

        self.wq = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.num_heads * self.head_dim, self.embed_dim, bias=False)
        self.rope = Llama3ScaledRoPE(config)
        self.kv_cache = None

    def forward(self, x: Tensor, *, mask: Tensor | None = None, input_pos: Tensor | None = None) -> Tensor:
        B, L, _ = x.shape

        q = self.wq(x).view(B, L, self.num_heads, self.head_dim)
        k = self.wk(x).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.wv(x).view(B, L, self.num_kv_heads, self.head_dim)

        q = self.rope(q).transpose(1, 2)
        k = self.rope(k).transpose(1, 2)
        v = v.transpose(1, 2)

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        q_per_kv = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(q_per_kv, dim=1)
        v = v.repeat_interleave(q_per_kv, dim=1)

        if mask is not None:
            mask = mask[:, None, :, :]

        is_causal = self.kv_cache is None and mask is None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_dropout, is_causal=is_causal)
        out = out.transpose(1, 2).reshape(B, L, self.embed_dim)
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self, config: Llama3_1Config):
        super().__init__()
        self.w1 = nn.Linear(config.embed_dim, config.intermediate_dim, bias=False)
        self.w3 = nn.Linear(config.embed_dim, config.intermediate_dim, bias=False)
        self.w2 = nn.Linear(config.intermediate_dim, config.embed_dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.act(self.w1(x)) * self.w3(x))


class TransformerLayer(nn.Module):
    def __init__(self, config: Llama3_1Config) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(config.embed_dim, eps=1e-5)
        self.attention = Attention(config)
        self.ffn_norm = nn.RMSNorm(config.embed_dim, eps=1e-5)
        self.feed_forward = FeedForward(config)

    def forward(self, x: Tensor, *, mask: Tensor | None = None, input_pos: Tensor | None = None) -> Tensor:
        x = x + self.attention(self.attention_norm(x), mask=mask, input_pos=input_pos)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class Llama3_1(nn.Module):
    def __init__(self, config: Llama3_1Config) -> None:
        super().__init__()
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_layers)])
        self.norm = nn.RMSNorm(config.embed_dim, eps=1e-5)
        self.output = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.activation_checkpointing = config.activation_checkpointing

    def forward(self, x: Tensor, *, mask: Tensor | None = None, input_pos: Tensor | None = None) -> Tensor:
        x = self.tok_embeddings(x)
        for layer in self.layers:
            if self.activation_checkpointing:
                x = checkpoint(layer, x, mask=mask, input_pos=input_pos)
            else:
                x = layer(x, mask=mask, input_pos=input_pos)
        x = self.norm(x)
        x = self.output(x)
        return x


def _rename_hf_key(key: str):
    return (
        key.removeprefix("model.")
        .replace("embed_tokens", "tok_embeddings")
        .replace("self_attn.q_proj", "attention.wq")
        .replace("self_attn.k_proj", "attention.wk")
        .replace("self_attn.v_proj", "attention.wv")
        .replace("self_attn.o_proj", "attention.wo")
        .replace("mlp.gate_proj", "feed_forward.w1")
        .replace("mlp.up_proj", "feed_forward.w3")
        .replace("mlp.down_proj", "feed_forward.w2")
        .replace("input_layernorm", "attention_norm")
        .replace("post_attention_layernorm", "ffn_norm")
        .replace("lm_head", "output")
    )


def _build_model(model_id: str, filenames: list[str], **kwargs):
    config = Llama3_1Config(**kwargs)
    with torch.device("meta"):
        model = Llama3_1(config).eval()

    state_dict = dict()
    for filename in filenames:
        filepath = hf_hub_download(model_id, filename)

        if filepath.endswith(".safetensors"):
            with safetensors.safe_open(filepath, framework="pt") as f:
                for k in f.keys():
                    state_dict[_rename_hf_key(k)] = f.get_tensor(k)

        else:
            this_state_dict = torch.load(filepath, map_location="cpu", weights_only=True, mmap=True)
            state_dict.update(this_state_dict)

    model.load_state_dict(state_dict, assign=True)
    return model


def llama3_1_8b(**kwargs):
    return _build_model(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        ["original/consolidated.00.pth"],
        embed_dim=4096,
        head_dim=128,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        intermediate_dim=14_336,
        **kwargs,
    )


def llama3_1_4b(**kwargs):
    return _build_model(
        "nvidia/Llama-3.1-Minitron-4B-Width-Base",
        ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
        embed_dim=3072,
        head_dim=128,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        intermediate_dim=9216,
        **kwargs,
    )
