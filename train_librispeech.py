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


def next_multiple(x: int, n: int) -> int:
    return (x + n - 1) // n * n


def pad_to_multiple(data_list: list[Tensor], multiple: int, fill_value: int = 0):
    max_length = max(x.shape[0] for x in data_list)
    target_length = next_multiple(max_length, multiple)
    padded_data = [F.pad(x, (0, target_length - x.shape[0]), value=fill_value) for x in data_list]
    return torch.stack(padded_data, dim=0)


class LibriSpeech(IterableDataset):
    def __init__(
        self,
        data_dirs: list[str],
        tokenizer_name: str,
        audio_duration_multiple: float,
        text_len_multiple: int,
        batch_size: int,
        audio_config: AudioConfig = AudioConfig(),
    ) -> None:
        super().__init__()
        self.audio_len_multiple = int(audio_duration_multiple * audio_config.sample_rate)
        self.text_len_multiple = text_len_multiple
        self.batch_size = batch_size
        self.audio_config = audio_config

        tokenizer = get_tokenizer(tokenizer_name)
        self.samples = []
        for data_dir in data_dirs:
            for file in Path(data_dir).glob("**/*.trans.txt"):
                for line in open(file):
                    audio_fname, text = line.rstrip().split(" ", 1)
                    audio_path = file.parent / f"{audio_fname}.flac"
                    # the extra space is needed for Llama3 tokenizer, but not for Llama2 tokenizer
                    tokens = tokenizer(f" {text.lower()}.", add_bos=True, add_eos=True)
                    self.samples.append((audio_path, torch.tensor(tokens, dtype=torch.int64)))
        self.samples.sort()

    def __iter__(self):
        epoch_idx = 0
        samples = self.samples
        n = len(samples)

        while True:
            # NOTE: we don't partition the data. just shuffle it with different seeds across workers
            indices = torch.randperm(n)
            samples = [samples[idx] for idx in indices]

            for i in range(0, n - self.batch_size + 1, self.batch_size):
                duration = 0
                n_toks = 0
                audio_batch = []
                tokens_batch = []
                labels_batch = []
                for audio_path, tokens in samples[i : i + self.batch_size]:
                    audio, fs = torchaudio.load(audio_path)
                    assert fs == self.audio_config.sample_rate
                    audio = audio.mean(0)
                    duration += audio.shape[0] / fs
                    n_toks += len(tokens) - 1
                    audio_batch.append(audio)
                    tokens_batch.append(tokens[:-1])
                    labels_batch.append(tokens[1:])

                audio_batch = pad_to_multiple(audio_batch, self.audio_len_multiple)
                tokens_batch = pad_to_multiple(tokens_batch, self.text_len_multiple)
                labels_batch = pad_to_multiple(labels_batch, self.text_len_multiple, fill_value=-100)
                yield audio_batch, tokens_batch, labels_batch, duration, n_toks, epoch_idx

            epoch_idx += 1


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

    parser.add_argument("--dataset_dirs", nargs="+", required=True)
    parser.add_argument("--audio_duration_multiple", type=float, default=8.0)
    parser.add_argument("--text_len_multiple", type=int, default=128)
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
    quantize_linear_(model.layers, args.quantize, **args.quantize_kwargs)
    apply_linear_adapter_(model.layers, args.adapter, **args.adapter_kwargs)
    freeze_params(model, args.freeze_prefixes)
    # TODO: handle quantization/LoRA for LM head separately
    if args.compile:
        model.compile()

    model.cuda()
    print_model_stats(model)

    optim = get_optimizer_class(args.optim)(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True)
    lr_schedule = LRScheduler(args.lr, args.n_steps, args.warmup, args.decay)

    ds = LibriSpeech(
        args.dataset_dirs,
        args.tokenizer,
        args.audio_duration_multiple,
        args.text_len_multiple,
        args.batch_size // args.gradient_accumulation,
        model.audio_config,
    )
    dloader = iter(DataLoader(ds, batch_size=None, num_workers=args.n_workers, pin_memory=True))

    args.save_dir = Path("runs/librispeech") / f"{args.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.save_dir.mkdir(parents=True, exist_ok=True)
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
    n_toks = audio_secs = 0
    audio_hrs_seen = 0
    time0 = time.perf_counter()

    while step < args.n_steps:
        for _ in range(args.gradient_accumulation):
            audio, tokens, labels, audio_secs_batch, n_toks_batch, epoch_idx = next(dloader)
            loss, text_norm, audio_norm = model(audio.cuda(), tokens.cuda(), labels=labels.cuda())
            (loss / args.gradient_accumulation).backward()
            audio_secs += audio_secs_batch
            audio_hrs_seen += audio_secs_batch / 3600
            n_toks += n_toks_batch

        lr_schedule.set_lr(optim, step)

        if args.clip_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        else:
            grad_norm = None

        if step % log_interval == 0:
            log_dict = dict(
                epoch=epoch_idx,
                loss=loss.item(),
                text_norm=text_norm.item(),
                audio_norm=audio_norm.item(),
                grad_norm=get_grad_norm(model) if grad_norm is None else grad_norm,
                audio_embed_grad_norm=get_grad_norm(model.audio_embed),
                lr=optim.param_groups[0]["lr"],
                max_memory_allocated=torch.cuda.max_memory_allocated() / 1e9,
                max_memory_reserved=torch.cuda.max_memory_reserved() / 1e9,
            )
            run.log(log_dict, step=step)
            pbar.set_postfix(loss=log_dict["loss"])

        optim.step()
        optim.zero_grad()

        step += 1
        pbar.update()

        if step % log_interval == 0:
            time1 = time.perf_counter()
            log_dict = dict(
                audio_hrs_seen=audio_hrs_seen,
                toks_per_second=n_toks / (time1 - time0),
                audio_secs_per_second=audio_secs / (time1 - time0),
            )
            n_toks = audio_secs = 0
            time0 = time1
            run.log(log_dict, step=step)

        if args.ckpt_interval > 0 and step % args.ckpt_interval == 0:
            ckpt = dict(
                step=step,
                model=model.state_dict(),
                optim=optim.state_dict(),
            )
            torch.save(ckpt, args.save_dir / "last.pth")

    run.finish()
