#!/usr/bin/env python3
"""
武侠微调的盲评材料生成器。

对同一批 held-out 提示，分别用
  * 基座 d6
  * DAPT（wuxia-dapt）
  * SFT（wuxia-sft）
生成续写，随机标成 A/B/C，并附上 val.jsonl 里的真实续写作为人类对照。
输出 Markdown + 答案 key（key 与盲评表分开保存）。

用法
----
    python scripts/eval_wuxia.py --n 3 --max-tokens 160
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_init, get_base_dir
from nanochat.engine import Engine


def read_prompts(path, n):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            msgs = obj["messages"]
            user = next((m["content"] for m in msgs if m["role"] == "user"), "")
            asst = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            if user.startswith("续写：") and 60 <= len(user) <= 350:
                out.append((user, asst))
            if len(out) >= n:
                break
    return out


def generate(model, tokenizer, user, source, max_tokens, temperature, top_k):
    bos = tokenizer.get_bos_token_id()
    asst_end = tokenizer.encode_special("<|assistant_end|>")
    if source != "sft":
        # 基座 / DAPT 没有见过“续写：”这种指令，喂原始上下文更公平。
        # 用朴素 generate（不靠 KV cache 的停表逻辑），跳过开头的 bos，
        # 避免“第一个 token 就是文档边界”导致空输出。
        prompt = user[len("续写："):] if user.startswith("续写：") else user
        ids = [bos] + tokenizer.encode(prompt)
        out = []
        for seed in (42, 43, 44):
            out = []
            for tok in model.generate(ids, max_tokens, temperature, top_k, seed=seed):
                if tok in (bos, asst_end):
                    if out:
                        break
                    continue
                out.append(tok)
            if out:
                break
        return tokenizer.decode(out)

    prompt = user
    toks = ([bos, tokenizer.encode_special("<|user_start|>")]
            + tokenizer.encode(prompt)
            + [tokenizer.encode_special("<|user_end|>"),
               tokenizer.encode_special("<|assistant_start|>")])
    engine = Engine(model, tokenizer)
    out = ""
    for seed in (42, 43, 44, 45, 46):
        sample, _ = engine.generate_batch(
            toks, num_samples=1, max_tokens=max_tokens,
            temperature=temperature, top_k=top_k, seed=seed)
        out = tokenizer.decode(sample[0][len(toks):]).strip()
        if out:
            break
    return out


def md_escape(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", "<br>")


def main() -> int:
    ap = argparse.ArgumentParser(description="武侠微调盲评材料")
    ap.add_argument("--val", default="/mnt/workspace/nanochat-data/wuxia_sft/val.jsonl")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    device_type = autodetect_device_type()
    ddp, rank, local_rank, world, device = compute_init(device_type)

    prompts = read_prompts(args.val, args.n)
    if not prompts:
        print("没有找到合适的 val 提示", file=sys.stderr)
        return 1

    specs = [
        ("base", "d6-zh", None, "基座 d6"),
        ("base", "wuxia-dapt", 500, "DAPT"),
        ("sft", "wuxia-sft", None, "SFT"),
    ]
    models = []
    for source, tag, step, label in specs:
        model, tokenizer, meta = load_model(
            source, device, phase="eval", model_tag=tag, step=step)
        models.append((label, source, model, tokenizer))
        print(f"loaded {label}: {tag} step {meta['step']}")

    results = []
    for i, (user, reference) in enumerate(prompts, 1):
        outputs = {}
        for label, source, model, tokenizer in models:
            outputs[label] = generate(
                model, tokenizer, user, source, args.max_tokens,
                args.temperature, args.top_k)
        labels = ["A", "B", "C"]
        random.shuffle(labels)
        mapping = {lab: label for lab, label in zip(labels, [m[0] for m in models])}
        results.append(dict(index=i, prompt=user, reference=reference,
                            outputs=outputs, mapping=mapping))
        print(f"prompt {i}: generated")

    out_dir = os.path.join(get_base_dir(), "eval")
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, "wuxia_blind_eval.md")
    key_path = os.path.join(out_dir, "wuxia_blind_key.json")

    lines = [
        "# 武侠微调盲评",
        "",
        "三份输出 A/B/C 分别来自基座 / DAPT / SFT（顺序已打乱），",
        "另有“人类对照”是 val.jsonl 里的真实续写。请按 1–5 分打分：",
        "武侠语感、情节连贯性、是否出戏。",
        "",
    ]
    key = {}
    for r in results:
        lines += [f"## 提示 {r['index']}", "", f"> {md_escape(r['prompt'])}", "",
                  "| 编号 | 输出 |", "|---|---|"]
        for lab in ("A", "B", "C"):
            lines.append(f"| {lab} | {md_escape(r['outputs'][r['mapping'][lab]])} |")
        lines.append(f"| 人类对照 | {md_escape(r['reference'])} |")
        lines.append("")
        key[str(r["index"])] = r["mapping"]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(key_path, "w", encoding="utf-8") as f:
        json.dump(key, f, ensure_ascii=False, indent=2)

    print(f"\n盲评表：{md_path}\n答案 key：{key_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
