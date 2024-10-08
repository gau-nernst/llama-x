from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from torchaudio.transforms import MelSpectrogram

from .llama import Llama, LlamaConfig, _get_hf_config, _get_hf_state_dict


class AudioConfig(NamedTuple):
    sample_rate: int = 16_000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 128


class LlamaAudio(Llama):
    def __init__(self, config: LlamaConfig, audio_config: AudioConfig = AudioConfig()):
        super().__init__(config)
        self.audio_config = audio_config

        # inspired by Whisper encoder
        self.audio_embed = nn.Sequential(
            nn.Conv1d(audio_config.n_mels, config.embed_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv1d(config.embed_dim, config.embed_dim, 3, 2, 1),
            nn.GELU(),
        )

    def build_cache(self, inference: bool = False):
        super().build_cache(inference)
        self.melspec = MelSpectrogram(**self.audio_config._asdict(), norm="slaney", mel_scale="slaney")
        self.melspec.spectrogram.forward = torch._dynamo.disable(self.melspec.spectrogram.forward)

    def forward(
        self,
        audio: Tensor | None,
        tokens: Tensor,
        *,
        input_pos: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> Tensor:
        # this is used for inference i.e. generate
        mask = self.causal_mask[None, None, input_pos] if input_pos is not None else None

        x = self.tok_embeddings(tokens)

        if audio is not None:
            # we need to slice the last time step to make it a nice multiple
            audio = self.melspec(audio)[..., :-1].clip(1e-12).log10()  # (B, n_mels, L)
            audio = audio - audio.mean(2, keepdim=True)  # cmn
            audio = audio.to(dtype=self.tok_embeddings.weight.dtype)
            if self.config.activation_checkpointing:
                audio = checkpoint(self.audio_embed, audio, use_reentrant=False)
            else:
                audio = self.audio_embed(audio)
            audio = audio.transpose(1, 2)

            # prefix audio
            x = torch.cat([audio, x], dim=1)

        rope = self.rope[: x.shape[1]]
        for layer in self.layers:
            if self.config.activation_checkpointing:
                x = checkpoint(layer, x, rope, mask=mask, input_pos=input_pos, use_reentrant=False)
            else:
                x = layer(x, rope, mask=mask, input_pos=input_pos)

        if audio is not None:
            x = x[:, audio.shape[1] :]  # remove audio embs
        x = self.output(self.norm(x))
        if labels is not None:
            x = F.cross_entropy(x.view(-1, x.shape[-1]).float(), labels.view(-1))
        return x

    @staticmethod
    def from_hf(model_id: str, **kwargs):
        audio_kwargs = {k: kwargs.pop(k) for k in kwargs if k in AudioConfig._fields}
        audio_config = AudioConfig(**audio_kwargs)
        config = _get_hf_config(model_id)
        config = config._replace(**kwargs)
        with torch.device("meta"):
            model = LlamaAudio(config, audio_config).eval()

        incompat_keys = model.load_state_dict(_get_hf_state_dict(model_id), strict=False, assign=True)
        if incompat_keys:
            print(incompat_keys)

        # these weights don't exist in state_dict. must manually initialize them from meta device.
        model.audio_embed.to_empty(device="cpu")
        model.audio_embed.to(dtype=model.tok_embeddings.weight.dtype)
        for m in model.audio_embed.modules():
            if isinstance(m, nn.Conv1d):
                m.reset_parameters()

        # we cannot build cache under meta device context. thus, build cache after loading weights
        model.build_cache()
        return model
