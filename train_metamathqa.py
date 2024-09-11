import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from datasets import load_dataset, load_from_disk
from torch import Tensor
from torch.nn.attention.flex_attention import create_block_mask
from tqdm import tqdm

from llama_tokenizers import get_tokenizer
from modelling import Llama, apply_linear_adapter_
from subclasses import quantize_linear_
from train_utils import LRScheduler, freeze_params, get_grad_norm, get_optimizer_class, print_model_stats


def next_multiple(x: int, n: int) -> int:
    return (x + n - 1) // n * n


def _data_iter_padding(tokens_list: list[Tensor], batch_size: int, seq_len_multiple: int = 256):
    n = len(tokens_list)

    while True:
        # shuffle
        indices = torch.randperm(n)
        tokens_list = [tokens_list[idx] for idx in indices]

        for i in range(0, n - batch_size + 1, batch_size):
            tokens_batch = tokens_list[i : i + batch_size]
            max_length = max(next_multiple(x.shape[0] - 1, seq_len_multiple) for x in tokens_batch)

            inputs = torch.zeros(batch_size, max_length, dtype=torch.int64)
            labels = torch.full((batch_size, max_length), -100, dtype=torch.int64)
            for _i, tokens in enumerate(tokens_batch):
                n_toks = tokens.shape[0] - 1
                inputs[_i, :n_toks] = tokens[:-1]
                labels[_i, :n_toks] = tokens[1:]

            yield inputs.cuda(), labels.cuda(), None


def _data_iter_document_mask(tokens_list: list[Tensor], seq_len: int):
    inputs = torch.zeros(seq_len, dtype=torch.int64)
    labels = torch.full((seq_len,), -100, dtype=torch.int64)
    doc_ids = torch.zeros(seq_len, dtype=torch.int64)
    i = 0
    doc_idx = 0

    while True:
        # shuffle
        indices = torch.randperm(len(tokens_list))
        tokens_list = [tokens_list[idx] for idx in indices]

        for tokens in tokens_list:
            if i + len(tokens) - 1 > seq_len:
                doc_ids = doc_ids.cuda()

                def mask_mod(b, h, q_idx, kv_idx):
                    return (doc_ids[q_idx] == doc_ids[kv_idx]) & (q_idx >= kv_idx)

                block_mask = create_block_mask(mask_mod, 1, None, seq_len, seq_len, _compile=True)
                yield inputs.view(1, -1).cuda(), labels.view(1, -1).cuda(), block_mask

                inputs = torch.zeros(seq_len, dtype=torch.int64)
                labels = torch.full((seq_len,), -100, dtype=torch.int64)
                doc_ids = torch.zeros(seq_len, dtype=torch.int64)
                i = 0

            l = len(tokens) - 1
            inputs[i : i + l] = tokens[:-1]
            labels[i : i + l] = tokens[1:]
            doc_ids[i : i + l] = doc_idx
            i += l
            doc_idx += 1


def get_metamathqa(
    tokenizer_name: str,
    document_mask: bool,
    batch_size: int,
    max_seq_len: int,
    seq_len_multiple: int = 256,
):
    # sequence length stats
    #
    # tokenizer | max  | P99.9 | P99
    # ----------|------|-------|-----
    # Llama2    | 2982 | 1178  | 751
    # Llama3    | 2318 | 1089  | 678

    ds_path = f"metamathqa_{tokenizer_name}"
    if Path(ds_path).exists():
        ds = load_from_disk(ds_path)

    else:
        tokenizer = get_tokenizer(tokenizer_name)

        def apply_template(example):
            prompt = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{query}\n\n"
                "### Response: Let's think step by step."
            ).format(query=example["query"])
            answer = " {response}".format(response=example["response"])
            prompt_tokens = tokenizer(prompt, add_bos=True)
            answer_tokens = tokenizer(answer, add_eos=True)
            return dict(input_ids=(prompt_tokens + answer_tokens)[: max_seq_len + 1])

        ds = load_dataset("meta-math/MetaMathQA", split="train").with_format("torch")
        ds = ds.map(apply_template, remove_columns=ds.features)
        ds.save_to_disk(ds_path)

    if document_mask:
        data_iter = _data_iter_document_mask(ds["input_ids"], batch_size * max_seq_len)
    else:
        data_iter = _data_iter_padding(ds["input_ids"], batch_size, seq_len_multiple)
    return data_iter, len(ds)


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

    parser.add_argument("--document_mask", action="store_true")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--seq_len_multiple", type=int, default=256)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation", type=int, default=1)

    parser.add_argument("--optim", default="AdamW")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--warmup", type=float, default=0.0)
    parser.add_argument("--decay", type=float, default=0.0)
    parser.add_argument("--clip_grad_norm", type=float)

    parser.add_argument("--ckpt_interval", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--project")
    parser.add_argument("--run_name")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    args.torch_version = torch.__version__
    assert args.batch_size % args.gradient_accumulation == 0
    if args.seed is not None:
        torch.manual_seed(args.seed)
    if args.profile:
        args.n_steps = 10

    model_max_seq_len = args.max_seq_len * (args.batch_size if args.document_mask else 1)
    model = Llama.from_hf(args.model, max_seq_len=model_max_seq_len)
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
    print(model)

    optim = get_optimizer_class(args.optim)(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True)
    lr_schedule = LRScheduler(args.lr, args.n_steps, args.warmup, args.decay)

    train_data_iter, train_size = get_metamathqa(
        args.tokenizer,
        args.document_mask,
        args.batch_size // args.gradient_accumulation,
        args.max_seq_len,
        seq_len_multiple=args.seq_len_multiple,
    )
    print(f"Training dataset size: {train_size:,}")

    args.save_dir = Path("runs/metamathqa") / f"{args.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.save_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(project=args.project, name=args.run_name, config=args, dir="/tmp")

    step = 0
    pbar = tqdm(total=args.n_steps, dynamic_ncols=True)
    model.train()
    n_toks = 0
    time0 = time.perf_counter()

    if args.profile:
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=4, active=2, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs"),
        )
        prof.start()

    while step < args.n_steps:
        for _ in range(args.gradient_accumulation):
            inputs, labels, block_mask = next(train_data_iter)
            # do a train step with longest seq_len to reserve enough memory + avoid memory fragmentation
            if not args.document_mask and step == 0:
                pad = args.max_seq_len - inputs.shape[1]
                inputs = F.pad(inputs, (0, pad))
                labels = F.pad(labels, (0, pad), value=-100)

            loss = model(inputs, labels=labels, block_mask=block_mask)
            (loss / args.gradient_accumulation).backward()
            n_toks += (labels != -100).sum()

        lr_schedule.set_lr(optim, step)

        if args.clip_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        else:
            grad_norm = None

        if step % args.log_interval == 0:
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
            torch.save(ckpt, args.save_dir / "last.pth")

        if args.profile:
            prof.step()

    if args.profile:
        prof.stop()

    run.finish()
