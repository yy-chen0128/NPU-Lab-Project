#!/usr/bin/env python3
"""
武侠风格 SFT：在金庸对话/续写数据上做监督微调。

数据格式（课程云盘的 wuxia_sft/*.jsonl）：
    {"messages": [{"role": "user", "content": ...},
                  {"role": "assistant", "content": ...}]}

训练目标只算 assistant 部分：prompt 与 padding 的 target 置 -1。
基座默认是 DAPT 后的 `wuxia-dapt`，保存到 `chatsft_checkpoints/<out-tag>`，
之后可以用 `python -m scripts.chat_cli -i sft -g <out-tag>` 推理。

用法
----
    python scripts/sft_train.py \
        --base-tag wuxia-dapt --base-step 500 \
        --data-dir /mnt/workspace/nanochat-data/wuxia_sft \
        --out-tag wuxia-sft --iters 800 --batch 8 --seq 512
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.common import autodetect_device_type, compute_init, get_base_dir
from nanochat.engine import Engine


def build_example(tokenizer, user: str, assistant: str, seq_len: int):
    """返回 (input_ids, targets)，targets 只保留 assistant 部分，其余 -1。"""
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    asst_start = tokenizer.encode_special("<|assistant_start|>")
    asst_end = tokenizer.encode_special("<|assistant_end|>")

    u = tokenizer.encode(user)
    a = tokenizer.encode(assistant)
    ids = [bos, user_start] + u + [user_end, asst_start] + a + [asst_end]
    # 只对 assistant 内容 + assistant_end 计损失
    is_asst = [False] * len(ids)
    start = len(ids) - len(a) - 1
    for i in range(start, len(ids)):
        is_asst[i] = True

    # 超长时保留尾部（含 assistant），保证至少能看到要学的部分
    if len(ids) > seq_len + 1:
        ids = ids[-(seq_len + 1):]
        is_asst = is_asst[-(seq_len + 1):]

    x = ids[:-1]
    tgt = [ids[i + 1] if is_asst[i + 1] else -1 for i in range(len(ids) - 1)]
    return x, tgt


def load_split(tokenizer, path: str, seq_len: int):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            msgs = obj["messages"]
            user = next((m["content"] for m in msgs if m["role"] == "user"), "")
            asst = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            if not user or not asst:
                continue
            x, y = build_example(tokenizer, user, asst, seq_len)
            if len(x) == 0 or all(t == -1 for t in y):
                continue
            data.append((x, y))
    return data


def make_batch(batch, seq_len, device):
    B = len(batch)
    x = torch.zeros(B, seq_len, dtype=torch.long)
    y = torch.full((B, seq_len), -1, dtype=torch.long)
    for i, (xi, yi) in enumerate(batch):
        n = min(len(xi), seq_len)
        x[i, :n] = torch.tensor(xi[:n], dtype=torch.long)
        y[i, :n] = torch.tensor(yi[:n], dtype=torch.long)
    return x.to(device), y.to(device)


def set_lr(optimizer, step, iters, warmup):
    mult = min(1.0, (step + 1) / max(1, warmup))
    mult *= max(0.1, 1.0 - step / max(1, iters))
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * mult
    return mult


@torch.no_grad()
def evaluate(model, data, seq_len, device, batch_size, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n = 0
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        x, y = make_batch(batch, seq_len, device)
        loss = model(x, y, loss_reduction="none")
        valid = (y != -1).sum().item()
        total_loss += float(loss.sum().item())
        total_tokens += valid
        n += 1
        if max_batches and n >= max_batches:
            break
    model.train()
    return total_loss / max(1, total_tokens)


def main() -> int:
    ap = argparse.ArgumentParser(description="武侠风格 SFT")
    ap.add_argument("--base-tag", default="wuxia-dapt")
    ap.add_argument("--base-step", type=int, default=500)
    ap.add_argument("--data-dir", default="/mnt/workspace/nanochat-data/wuxia_sft")
    ap.add_argument("--out-tag", default="wuxia-sft")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    model, tokenizer, meta = load_model(
        "base", device, phase="train", model_tag=args.base_tag, step=args.base_step)
    model.train()
    print(f"SFT 基座：{args.base_tag} step {args.base_step}，"
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M 参数")

    train_path = os.path.join(args.data_dir, "train.jsonl")
    val_path = os.path.join(args.data_dir, "val.jsonl")
    t0 = time.time()
    train_data = load_split(tokenizer, train_path, args.seq)
    val_data = load_split(tokenizer, val_path, args.seq)
    print(f"数据：train={len(train_data)}  val={len(val_data)}  "
          f"tokenize {time.time() - t0:.1f}s")

    optimizer = model.setup_optimizer(
        matrix_lr=args.lr,
        embedding_lr=args.lr * 10,
        unembedding_lr=args.lr * 0.5,
        scalar_lr=0.05,
        weight_decay=0.0,
    )

    best_val = float("inf")
    step = 0
    t_start = time.time()
    order = list(range(len(train_data)))
    random.shuffle(order)
    pos = 0

    while step < args.iters:
        if pos + args.batch > len(order):
            random.shuffle(order)
            pos = 0
        idx = order[pos:pos + args.batch]
        pos += args.batch
        batch = [train_data[i] for i in idx]
        x, y = make_batch(batch, args.seq, device)

        optimizer.zero_grad(set_to_none=True)
        loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lrm = set_lr(optimizer, step, args.iters, args.warmup)
        optimizer.step()

        step += 1
        if step % 10 == 0:
            elapsed = time.time() - t_start
            tok_s = step * args.batch * args.seq / max(1e-6, elapsed)
            print(f"step {step:4d}/{args.iters} | loss {loss.item():.4f} | "
                  f"lr_mult {lrm:.3f} | {tok_s:,.0f} tok/s | {elapsed / 60:.1f}m",
                  flush=True)

        if step % args.val_every == 0 or step == args.iters:
            val_loss = evaluate(model, val_data, args.seq, device,
                                args.batch, max_batches=40)
            best_val = min(best_val, val_loss)
            print(f"  [val] step {step}: masked loss {val_loss:.4f} "
                  f"(best {best_val:.4f})", flush=True)

    model.eval()
    ckpt_dir = os.path.join(get_base_dir(), "chatsft_checkpoints", args.out_tag)
    meta_out = {
        "step": args.iters,
        "val_bpb": best_val,
        "model_config": asdict(model.config),
        "user_config": vars(args),
        "device_batch_size": args.batch,
        "max_seq_len": args.seq,
        "total_batch_size": args.batch * args.seq,
        "dataloader_state_dict": {},
        "loop_state": {"min_val_bpb": best_val},
    }
    save_checkpoint(ckpt_dir, args.iters, model.state_dict(), None, meta_out)
    print(f"已保存 SFT checkpoint：{ckpt_dir} step {args.iters}")

    # 生成一条样例，方便快速肉眼检查
    try:
        prompts = [
            "请用金庸武侠小说的笔法续写：那少年提剑立在崖边，",
            "师父，陆老前辈，咱们拿这恶贼怎么办？",
        ]
        bos = tokenizer.get_bos_token_id()
        user_start = tokenizer.encode_special("<|user_start|>")
        user_end = tokenizer.encode_special("<|user_end|>")
        asst_start = tokenizer.encode_special("<|assistant_start|>")
        engine = Engine(model, tokenizer)
        for prompt in prompts:
            toks = ([bos, user_start] + tokenizer.encode(prompt)
                    + [user_end, asst_start])
            sample, _ = engine.generate_batch(
                toks, num_samples=1, max_tokens=120,
                temperature=0.7, top_k=50)
            out = tokenizer.decode(sample[0][len(toks):])
            print("\n" + "-" * 60)
            print("用户：", prompt)
            print("武侠 SFT：", out)
    except Exception as e:                                  # noqa: BLE001
        print(f"[warn] 样例生成失败：{e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
