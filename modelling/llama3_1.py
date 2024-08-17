# code in this file has been adapted from
# https://github.com/pytorch/torchtune
# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


class Llama3ScaledRoPE(nn.Module):
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.is_cache_built = False

    def reset_parameters(self):
        self._rope_init()

    def _rope_init(self):
        freqs = 1.0 / (self.base ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim))
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
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 4096,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert num_heads % num_kv_heads == 0
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        self.wq = nn.Linear(embed_dim, num_heads * head_dim, bias=False)
        self.wk = nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(embed_dim, embed_dim, bias=False)
        self.rope = Llama3ScaledRoPE(dim=head_dim, max_seq_len=max_seq_len, base=500_000)
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
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.act(self.w1(x)) * self.w3(x))


class TransformerLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
        intermediate_dim: int,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        head_dim = embed_dim // num_heads
        self.attention_norm = nn.RMSNorm(embed_dim, eps=1e-5)
        self.attention = Attention(embed_dim, num_heads, num_kv_heads, head_dim, max_seq_len, attn_dropout)
        self.ffn_norm = nn.RMSNorm(embed_dim, eps=1e-5)
        self.feed_forward = FeedForward(embed_dim, intermediate_dim)

    def forward(self, x: Tensor, *, mask: Tensor | None = None, input_pos: Tensor | None = None) -> Tensor:
        x = x + self.attention(self.attention_norm(x), mask=mask, input_pos=input_pos)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class Llama3_1(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_dim: int,
        max_seq_len: int = 2048,
        vocab_size: int = 128256,
        attn_dropout: float = 0.0,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList(
            [
                TransformerLayer(embed_dim, num_heads, num_kv_heads, max_seq_len, intermediate_dim, attn_dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.RMSNorm(embed_dim, eps=1e-5)
        self.output = nn.Linear(embed_dim, vocab_size, bias=False)
        self.activation_checkpointing = activation_checkpointing

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


def llama3_1_8b(**kwargs):
    with torch.device("meta"):
        model = Llama3_1(
            embed_dim=4096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            intermediate_dim=14_336,
            **kwargs,
        )

    state_dict_path = hf_hub_download("meta-llama/Meta-Llama-3.1-8B-Instruct", "original/consolidated.00.pth")
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state_dict, assign=True)
    return model
