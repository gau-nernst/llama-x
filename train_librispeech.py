# please download LibriSpeech from here: https://www.openslr.org/12
#
# some statistics
# split           |    audio duration    |    llama2 tokens   |
#                 | max   | P99.9 | P99  | max  | P99.9 | P99 |
# ----------------|-------|-------|------|------|-------|-----|
# train-clean-100 | 24.5  | 17.1  | 16.7 |  91  |  78   | 69  |
# train-clean-360 | 29.7  | 17.1  | 16.7 | 110  |  78   | 69  |
# dev-clean       | 32.6  | 32.3  | 23.8 | 133  | 116   | 84  |
# test-clean      | 35.0  | 32.8  | 25.5 | 130  | 104   | 82  |

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import wandb
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from llama_tokenizers import get_tokenizer
from modelling import AudioConfig, LlamaAudio, apply_linear_adapter_
from subclasses import quantize_linear_
from train_utils import LRScheduler, freeze_params, get_grad_norm, get_optimizer_class, print_model_stats


class LibriSpeech(IterableDataset):
    def __init__(
        self,
        data_dir: str,
        tokenizer: str,
        audio_duration: float,
        seq_len_multiple: int,
        batch_size: int,
        audio_config: AudioConfig = AudioConfig(),
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.audio_duration = audio_duration
        self.seq_len_multiple = seq_len_multiple
        self.batch_size = batch_size
        self.audio_config = audio_config

        _tokenizer = get_tokenizer(tokenizer)
        self.samples = []
        for file in self.data_dir.glob("**/*.trans.txt"):
            for line in open(file):
                audio_fname, text = line.rstrip().split(" ", 1)
                audio_path = str((file.parent / f"{audio_fname}.flac").relative_to(self.data_dir))
                tokens = _tokenizer(f" {text.lower()}.")
                self.samples.append((audio_path, tokens))

        self.samples.sort()
        self.bos_id = _tokenizer.bos_id
        self.eos_id = _tokenizer.eos_id
        self.pad_id = _tokenizer.pad_id

    def _prepare_batch(self, batch: list[tuple[Tensor, Tensor]]):
        audio_batch, _tokens_batch = zip(*batch)

        audio_length = int(self.audio_duration * self.audio_config.sample_rate)
        audio_batch = [F.pad(x, (0, audio_length - x.shape[0])) for x in audio_batch]
        audio_batch = torch.stack(audio_batch, dim=0)

        tokens_length = math.ceil(max(len(x) for x in _tokens_batch) / self.seq_len_multiple) * self.seq_len_multiple
        tokens_batch = []
        labels_batch = []
        for tokens in _tokens_batch:
            pad = tokens_length - len(tokens)
            tokens_batch.append(tokens + [self.pad_id] * pad)
            labels_batch.append(tokens[1:] + [-100] * (pad + 1))

        tokens_batch = torch.tensor(tokens_batch)
        labels_batch = torch.tensor(labels_batch)

        return audio_batch, tokens_batch, labels_batch

    def __iter__(self):
        batch = []
        audio = []
        tokens = [self.bos_id]
        duration = 0

        while True:
            # NOTE: we don't partition the data. just shuffle it with different seeds across workers
            # re-think this...
            for idx in torch.randperm(len(self.samples)):
                # pack samples until we reach seq_len
                this_audio_path, this_tokens = self.samples[idx]
                this_audio, fs = torchaudio.load(self.data_dir / this_audio_path)
                assert fs == self.audio_config.sample_rate
                this_audio = this_audio.mean(0)

                # TODO: think about how to handle this better
                this_duration = this_audio.shape[0] / fs
                if this_duration > self.audio_duration:
                    continue

                if duration + this_duration > self.audio_duration:
                    audio = torch.cat(audio, dim=0)
                    tokens.append(self.eos_id)

                    batch.append((audio, tokens))
                    if len(batch) == self.batch_size:
                        yield self._prepare_batch(batch)
                        batch = []

                    audio = []
                    tokens = [self.bos_id]
                    duration = 0

                audio.append(this_audio)
                tokens.extend(this_tokens)
                duration += this_duration


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama_v1.1")
    parser.add_argument("--tokenizer", default="llama2")
    parser.add_argument("--adapter")
    parser.add_argument("--adapter_kwargs", type=json.loads, default=dict())
    parser.add_argument("--quantize")
    parser.add_argument("--quantize_kwargs", type=json.loads, default=dict())
    parser.add_argument("--freeze_prefixes", nargs="+", default=[])
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--audio_duration", type=float, default=40)
    parser.add_argument("--seq_len_multiple", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--gradient_accumulation", type=int, default=1)

    parser.add_argument("--optim", default="AdamW")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--warmup", type=float, default=0.0)
    parser.add_argument("--decay", type=float, default=0.0)
    parser.add_argument("--clip_grad_norm", type=float)

    parser.add_argument("--resume")
    parser.add_argument("--ckpt_interval", type=int, default=1000)
    parser.add_argument("--project")
    parser.add_argument("--run_name")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    args.torch_version = torch.__version__
    assert args.batch_size % args.gradient_accumulation == 0
    if args.seed is not None:
        torch.manual_seed(args.seed)

    model = LlamaAudio.from_hf(args.model, max_seq_len=4096)
    if args.activation_checkpointing:
        model.enable_activation_checkpointing()
    freeze_params(model, args.freeze_prefixes)
    quantize_linear_(model.layers, args.quantize, **args.quantize_kwargs)
    apply_linear_adapter_(model.layers, args.adapter, **args.adapter_kwargs)
    # TODO: handle quantization/LoRA for LM head separately
    if args.compile:
        model.compile()

    model.cuda()
    print_model_stats(model)

    optim = get_optimizer_class(args.optim)(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True)
    lr_schedule = LRScheduler(args.lr, args.n_steps, args.warmup, args.decay)

    ds = LibriSpeech(
        args.dataset_dir,
        args.tokenizer,
        args.audio_duration,
        args.seq_len_multiple,
        args.batch_size // args.gradient_accumulation,
        model.audio_config,
    )
    dloader = iter(DataLoader(ds, batch_size=None, num_workers=args.n_workers, pin_memory=True))

    save_dir = Path("runs/librispeech") / f"{args.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(project=args.project, name=args.run_name, config=args, dir="/tmp")

    step = 0

    if args.resume is not None:
        state_dict = torch.load(args.resume, map_location="cpu", weights_only=True, mmap=True)
        step = state_dict["step"]
        model.load_state_dict(state_dict["model"])
        optim.load_state_dict(state_dict["optim"])

    log_interval = 50
    pbar = tqdm(initial=step, total=args.n_steps, dynamic_ncols=True)
    model.train()
    n_toks = 0
    time0 = time.perf_counter()

    while step < args.n_steps:
        for _ in range(args.gradient_accumulation):
            audio, tokens, labels = next(dloader)
            loss = model(audio.cuda(), tokens.cuda(), labels=labels.cuda())
            (loss / args.gradient_accumulation).backward()
            n_toks += (labels != -100).sum()

        lr_schedule.set_lr(optim, step)

        if args.clip_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        else:
            grad_norm = None

        if step % log_interval == 0:
            log_dict = dict(
                loss=loss.item(),
                grad_norm=get_grad_norm(model) if grad_norm is None else grad_norm,
                lr=optim.param_groups[0]["lr"],
                max_memory_allocated=torch.cuda.max_memory_allocated() / 1e9,
                max_memory_reserved=torch.cuda.max_memory_reserved() / 1e9,
            )
            if step > 0:
                time1 = time.perf_counter()
                log_dict["toks_per_second"] = n_toks / (time1 - time0)
                log_dict["audio_secs_per_second"] = (args.audio_duration * args.batch_size) / (time1 - time0)
                n_toks = 0
                time0 = time1
            run.log(log_dict, step=step)
            pbar.set_postfix(loss=log_dict["loss"])

        optim.step()
        optim.zero_grad()

        step += 1
        pbar.update()

        if args.ckpt_interval > 0 and step % args.ckpt_interval == 0:
            ckpt = dict(
                step=step,
                model=model.state_dict(),
                optim=optim.state_dict(),
            )
            torch.save(ckpt, save_dir / "last.pth")

    run.finish()
