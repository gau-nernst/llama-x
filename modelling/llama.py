# code in this file has been adapted from
# https://github.com/pytorch/torchtune
# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py

import json
from typing import NamedTuple

import safetensors
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, list_repo_files
from torch import Tensor, nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.utils.checkpoint import checkpoint


class LlamaConfig(NamedTuple):
    embed_dim: int
    num_layers: int
    head_dim: int
    num_heads: int
    num_kv_heads: int
    intermediate_dim: int
    max_seq_len: int = 2048
    vocab_size: int = 128_256  # Llama3
    attn_dropout: float = 0.0
    rope_base: int = 50_000
    is_llama3_1: bool = False
    activation_checkpointing: bool = False


def scale_llama3_1_rope(freqs: torch.Tensor):
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


def build_rope(config: LlamaConfig):
    theta = 1.0 / (config.rope_base ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim))
    if config.is_llama3_1:
        theta = scale_llama3_1_rope(theta)
    seq_idx = torch.arange(config.max_seq_len, dtype=torch.float32)
    idx_theta = torch.einsum("i, j -> ij", seq_idx, theta)
    return torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)


def apply_rope(x: Tensor, rope: Tensor) -> Tensor:
    rope = rope.view(1, x.shape[1], 1, -1, 2)
    out = x.float().unflatten(-1, (-1, 2))
    out = torch.stack(
        [
            out[..., 0] * rope[..., 0] - out[..., 1] * rope[..., 1],
            out[..., 1] * rope[..., 0] + out[..., 0] * rope[..., 1],
        ],
        -1,
    )
    return out.flatten(3).type_as(x)


class KVCache(nn.Module):
    def __init__(self, batch_size: int, config: LlamaConfig, dtype: torch.dtype):
        super().__init__()
        shape = (batch_size, config.num_kv_heads, config.max_seq_len, config.head_dim)
        self.register_buffer("k_cache", torch.zeros(shape, dtype=dtype), persistent=False)
        self.register_buffer("v_cache", torch.zeros(shape, dtype=dtype), persistent=False)

    def update(self, input_pos: Tensor, k: Tensor, v: Tensor):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k.shape[2], (input_pos.shape, k.shape)
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k
        v_out[:, :, input_pos] = v
        return k_out, v_out


class Attention(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
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
        self.kv_cache = None

    def forward(
        self,
        x: Tensor,
        rope: Tensor,
        *,
        mask: Tensor | None = None,
        input_pos: Tensor | None = None,
        block_mask: Tensor | None = None,
    ) -> Tensor:
        B, L, _ = x.shape
        q = self.wq(x).view(B, L, self.num_heads, self.head_dim)
        k = self.wk(x).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.wv(x).view(B, L, self.num_kv_heads, self.head_dim)

        q = apply_rope(q, rope).transpose(1, 2)
        k = apply_rope(k, rope).transpose(1, 2)
        v = v.transpose(1, 2)

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        if block_mask is not None:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            out = flex_attention(q, k, v, block_mask=block_mask)

        else:
            is_causal = self.kv_cache is None and mask is None
            dropout = self.attn_dropout if self.training else 0.0
            out = F.scaled_dot_product_attention(q, k, v, mask, dropout, is_causal, enable_gqa=True)

        out = out.transpose(1, 2).reshape(B, L, self.num_heads * self.head_dim)
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.w1 = nn.Linear(config.embed_dim, config.intermediate_dim, bias=False)
        self.w3 = nn.Linear(config.embed_dim, config.intermediate_dim, bias=False)
        self.w2 = nn.Linear(config.intermediate_dim, config.embed_dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.act(self.w1(x)) * self.w3(x))


class TransformerLayer(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(config.embed_dim, eps=1e-5)
        self.attention = Attention(config)
        self.ffn_norm = nn.RMSNorm(config.embed_dim, eps=1e-5)
        self.feed_forward = FeedForward(config)

    def forward(
        self,
        x: Tensor,
        rope: Tensor,
        *,
        mask: Tensor | None = None,
        input_pos: Tensor | None = None,
        block_mask: Tensor | None = None,
    ) -> Tensor:
        x = x + self.attention(self.attention_norm(x), rope, mask=mask, input_pos=input_pos, block_mask=block_mask)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class Llama(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_layers)])
        self.norm = nn.RMSNorm(config.embed_dim, eps=1e-5)
        self.output = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.config = config

    def build_cache(self, inference: bool = False):
        self.register_buffer("rope", build_rope(self.config), persistent=False)

        if inference:
            for layer in self.layers:
                layer.attention.kv_cache = KVCache(1, self.config, self.tok_embeddings.weight.dtype)

            L = self.config.max_seq_len
            self.register_buffer("causal_mask", torch.tril(torch.ones(L, L, dtype=torch.bool)), persistent=False)

    @torch._dynamo.disable
    def build_block_mask(self, x: Tensor, prefix_lengths: Tensor):
        def prefix_mask(b, h, q_idx, kv_idx):
            return (kv_idx <= prefix_lengths[b]) | (q_idx >= kv_idx)

        B, L = x.shape
        return create_block_mask(prefix_mask, B, self.config.num_heads, L, L)

    def forward(
        self,
        x: Tensor,
        *,
        input_pos: Tensor | None = None,
        prefix_lengths: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> Tensor:
        if prefix_lengths is not None:
            assert input_pos is None
            block_mask = self.build_block_mask(x, prefix_lengths)
        else:
            block_mask = None

        # this is used for inference i.e. generate
        mask = self.causal_mask[None, None, input_pos] if input_pos is not None else None

        x = self.tok_embeddings(x)
        rope = self.rope[: x.shape[1]]
        for layer in self.layers:
            if self.config.activation_checkpointing:
                x = checkpoint(
                    layer, x, rope, mask=mask, input_pos=input_pos, block_mask=block_mask, use_reentrant=False
                )
            else:
                x = layer(x, rope, mask=mask, input_pos=input_pos, block_mask=block_mask)

        x = self.output(self.norm(x))
        if labels is not None:
            x = F.cross_entropy(x.view(-1, x.shape[-1]).float(), labels.view(-1))
        return x

    @staticmethod
    def from_hf(model_id: str, **kwargs):
        config = _get_hf_config(model_id)
        config = config._replace(**kwargs)
        with torch.device("meta"):
            model = Llama(config).eval()

        # we cannot build cache under meta device context. thus, build cache after loading weights
        model.load_state_dict(_get_hf_state_dict(model_id), assign=True)
        model.build_cache()
        return model


def _get_hf_config(model_id: str):
    config_path = hf_hub_download(model_id, "config.json")
    hf_config = json.load(open(config_path))
    assert hf_config["architectures"][0] == "LlamaForCausalLM"

    config = LlamaConfig(
        embed_dim=hf_config["hidden_size"],
        num_layers=hf_config["num_hidden_layers"],
        head_dim=hf_config.get("head_dim", hf_config["hidden_size"] // hf_config["num_attention_heads"]),
        num_heads=hf_config["num_attention_heads"],
        num_kv_heads=hf_config["num_key_value_heads"],
        intermediate_dim=hf_config["intermediate_size"],
        vocab_size=hf_config["vocab_size"],
    )
    if "rope_theta" in hf_config:
        config = config._replace(rope_base=hf_config["rope_theta"])
    # TODO: may need more rigorous check
    if hf_config.get("rope_scaling", None) is not None:
        config = config._replace(is_llama3_1=hf_config["rope_scaling"]["rope_type"] == "llama3")
    return config


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


def _get_hf_state_dict(model_id: str):
    for ext in (".safetensors", ".bin"):
        filenames = [x for x in list_repo_files(model_id) if x.endswith(ext)]
        if filenames:
            break

    if not filenames:
        raise RuntimeError(f"No weights found for {model_id=}")

    state_dict = dict()
    for filename in filenames:
        filepath = hf_hub_download(model_id, filename)
        if filepath.endswith(".safetensors"):
            with safetensors.safe_open(filepath, framework="pt") as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
        else:
            state_dict.update(torch.load(filepath, map_location="cpu", weights_only=True, mmap=True))
    state_dict = {_rename_hf_key(k): v for k, v in state_dict.items()}
    return state_dict
