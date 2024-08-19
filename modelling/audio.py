from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from torchaudio.transforms import MelSpectrogram

from .llama3_1 import Llama3_1, Llama3_1Config, llama3_1_4b, llama3_1_8b


class AudioConfig(NamedTuple):
    sample_rate: int = 16_000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 128


class Llama3_1Audio(Llama3_1):
    def __init__(self, config: Llama3_1Config, audio_config: AudioConfig = AudioConfig()):
        super().__init__(config)
        self.audio_config = audio_config

        # inspired by Whisper encoder
        self.conv = nn.Sequential(
            nn.Conv1d(audio_config.n_mels, config.embed_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv1d(config.embed_dim, config.embed_dim, 3, 2, 1),
            nn.GELU(),
        )

    def build_cache(self):
        super().build_cache()
        self.melspec = MelSpectrogram(**self.audio_config._asdict())

    def forward(
        self,
        audio: Tensor,
        tokens: Tensor,
        *,
        mask: Tensor | None = None,
        input_pos: Tensor | None = None,
    ) -> Tensor:
        # we need to slice the last time step to make it a nice multiple
        audio = self.melspec(audio)[..., :-1].clip(1e-12).log()  # (B, n_mels, L)
        audio = audio - audio.mean(2, keepdim=True)  # cmn
        audio = audio.to(dtype=self.conv[0].weight.dtype)
        audio = self.conv(audio).transpose(1, 2)

        tok_embs = self.tok_embeddings(tokens)
        x = torch.cat([audio, tok_embs], dim=1)

        rope = self.rope[: x.shape[1]]
        for layer in self.layers:
            if self.config.activation_checkpointing:
                x = checkpoint(layer, x, rope, mask=mask, input_pos=input_pos, use_reentrant=False)
            else:
                x = layer(x, rope, mask=mask, input_pos=input_pos)
        x = self.norm(x)
        x = self.output(x)
        return x


def _build_audio_model(base_model: Llama3_1, **kwargs):
    audio_config = AudioConfig(**kwargs)
    with torch.device("meta"):
        model = Llama3_1Audio(base_model.config, audio_config).eval()

    incompat_keys = model.load_state_dict(base_model.state_dict(), strict=False, assign=True)
    if incompat_keys:
        print(incompat_keys)

    # these weights don't exist in state_dict. must manually initialize them from meta device.
    model.conv.to_empty(device="cpu")
    model.conv.to(dtype=model.tok_embeddings.weight.dtype)
    for m in model.conv.modules():
        if isinstance(m, nn.Conv1d):
            m.reset_parameters()

    model.build_cache()
    return model


def llama3_1_audio_8b(**kwargs):
    audio_kwargs = {k: kwargs.pop(k) for k in kwargs if k in AudioConfig._fields}
    base_model = llama3_1_8b(**kwargs)
    return _build_audio_model(base_model, **audio_kwargs)


def llama3_1_audio_4b(**kwargs):
    audio_kwargs = {k: kwargs.pop(k) for k in kwargs if k in AudioConfig._fields}
    base_model = llama3_1_4b(**kwargs)
    return _build_audio_model(base_model, **audio_kwargs)
