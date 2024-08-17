import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import math
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from torch import Tensor
from tqdm import tqdm

import modelling
from modelling import Int8LoRALinear, Llama3Tokenizer
from train_utils import get_grad_norm, print_model_stats


def _data_iter(tokens_list: list[Tensor], batch_size: int, seq_len_multiple: int = 256):
    n = len(tokens_list)

    while True:
        # shuffle
        tokens_list = [tokens_list[idx] for idx in torch.randperm(n)]

        for i in range(0, n - batch_size + 1, batch_size):
            tokens_batch = tokens_list[i : i + batch_size]
            max_length = max(math.ceil(x.shape[0] / seq_len_multiple) * seq_len_multiple for x in tokens_batch)

            inputs = torch.zeros(batch_size, max_length, dtype=torch.int64)
            labels = torch.full((batch_size, max_length), -100, dtype=torch.int64)
            lengths = torch.empty(batch_size, dtype=torch.int64)
            for _i, tokens in enumerate(tokens_batch):
                n_toks = tokens.shape[0]
                inputs[_i, :n_toks] = tokens
                labels[_i, :n_toks] = tokens
                lengths[_i] = n_toks

            yield inputs.cuda(), labels.cuda(), lengths


def get_metamathqa(batch_size: int, max_seq_len: int, seq_len_multiple: int = 256):
    tokenizer = Llama3Tokenizer()

    def apply_template(example):
        text = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{query}\n\n"
            "### Response: Let's think step by step. {response}"
        ).format(query=example["query"], response=example["response"])
        return dict(input_ids=tokenizer(text, add_bos=True, add_eos=True)[:max_seq_len])

    ds = load_dataset("meta-math/MetaMathQA", split="train").with_format("torch")
    tokens_list = ds.map(apply_template, remove_columns=ds.features)["input_ids"]
    return _data_iter(tokens_list, batch_size, seq_len_multiple), len(ds)


def get_loss(model, inputs, labels):
    logits = model(inputs)[:, :-1].flatten(0, 1)
    labels = labels[:, 1:].flatten()
    return F.cross_entropy(logits, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3_1_4b")
    parser.add_argument("--freeze_embedding_layer", action="store_true")
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--seq_len_multiple", type=int, default=256)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_steps", type=int, default=1000)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)

    parser.add_argument("--ckpt_interval", type=int, default=1000)
    parser.add_argument("--project")
    parser.add_argument("--run_name", default="debug")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    model: modelling.Llama3_1 = getattr(modelling, args.model)(
        max_seq_len=args.max_seq_len,
        activation_checkpointing=args.activation_checkpointing,
    )
    if args.freeze_embedding_layer:
        model.tok_embeddings.requires_grad_(False)
    Int8LoRALinear.convert_model(model.layers)

    # quantize, non-trainble, no LoRA
    model.output = Int8LoRALinear.convert_model(model.output, rank=0)
    model.cuda()
    print_model_stats(model)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True)

    train_data_iter, train_size = get_metamathqa(
        args.batch_size,
        args.max_seq_len,
        seq_len_multiple=args.seq_len_multiple,
    )
    print(f"Training dataset size: {train_size:,}")
    print(f"Each epoch will takes {train_size // args.batch_size:,} iters to finish")

    save_dir = Path("runs/metamathqa") / f"{args.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(project=args.project, name=args.run_name, config=args, dir="/tmp")

    step = 0
    log_interval = 50
    pbar = tqdm(total=args.n_steps, dynamic_ncols=True)
    model.train()
    n_toks = 0
    time0 = time.perf_counter()

    while step < args.n_steps:
        inputs, labels, lengths = next(train_data_iter)
        loss_fn = torch.compile(get_loss, fullgraph=True) if args.compile else get_loss
        loss = loss_fn(model, inputs, labels)
        loss.backward()
        n_toks += lengths.sum()

        if step % log_interval == 0:
            log_dict = dict(
                loss=loss.item(),
                grad_norm=get_grad_norm(model),
                lr=optim.param_groups[0]["lr"],
                max_memory_allocated=torch.cuda.max_memory_allocated(),
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
            torch.save(ckpt, save_dir / "last.pth")

    run.finish()
