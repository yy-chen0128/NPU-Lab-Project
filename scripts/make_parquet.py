#!/usr/bin/env python3
"""
把 `data/corpus/` 下的纯文本切成训练用的 parquet 分片。

为什么需要它
------------
`nanochat/dataset.py` 只读 `*.parquet`，而随包的起步语料是 `.txt`。
README 里"纯文本目录也能训"在这版代码里并不成立（见 docs/01_任务清单.md
之外的实现差异）。本脚本把语料转成 dataloader 认识的格式：

  * 每行一个文档，列名必须是 `text`
  * 文件名排序后**最后一个**会被当作验证集（dataset.py 的约定）

用法
----
    python scripts/make_parquet.py
    python scripts/make_parquet.py --src data/corpus --out data/corpus_parquet
    python scripts/make_parquet.py --val-files 2 --max-chars 1500

输出
----
    <out>/train.parquet   训练分片
    <out>/val.parquet     验证分片（排序在最后）
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    print(f"  [warn] {path.name}: 无法按 utf-8/gb18030 解码，按 utf-8 替换坏字节",
          file=sys.stderr)
    return raw.decode("utf-8", errors="replace")


def iter_docs(text: str, max_chars: int):
    """按空行切段；过短的段合并，过长的段再切开。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"\n\s*\n", text)
    buf = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) > max_chars:
            if buf:
                yield buf
                buf = ""
            for i in range(0, len(part), max_chars):
                yield part[i:i + max_chars]
        elif not buf:
            buf = part
        elif len(buf) + 1 + len(part) <= max_chars:
            buf = f"{buf}\n{part}"
        else:
            yield buf
            buf = part
    if buf:
        yield buf


def write_shard(path: Path, docs: list[str]) -> int:
    table = pa.table({"text": docs})
    pq.write_table(table, path)
    return path.stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser(description="txt 语料 -> parquet 分片")
    ap.add_argument("--src", default="data/corpus", help="纯文本目录")
    ap.add_argument("--out", default="data/corpus_parquet", help="输出目录")
    ap.add_argument("--val-files", type=int, default=1,
                    help="按文件名排序，最后 N 个文件作为验证集（默认 1）")
    ap.add_argument("--max-chars", type=int, default=2000,
                    help="单个文档的最大字符数（默认 2000）")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    files = sorted(p for p in src.glob("*.txt") if p.is_file())
    if not files:
        print(f"错误：{src} 下没有 .txt 文件", file=sys.stderr)
        return 1
    if args.val_files < 1 or args.val_files >= len(files):
        print(f"错误：--val-files 必须在 1..{len(files) - 1} 之间", file=sys.stderr)
        return 1

    train_files = files[:-args.val_files]
    val_files = files[-args.val_files:]
    out.mkdir(parents=True, exist_ok=True)

    def build(paths: list[Path]) -> list[str]:
        docs: list[str] = []
        for p in paths:
            text = read_text(p)
            got = list(iter_docs(text, args.max_chars))
            docs.extend(got)
            print(f"  {p.name}: {len(text):,} 字符 -> {len(got):,} 段")
        return docs

    print(f"训练集（{len(train_files)} 个文件）：")
    train_docs = build(train_files)
    print(f"验证集（{len(val_files)} 个文件）：")
    val_docs = build(val_files)

    train_path = out / "train.parquet"
    val_path = out / "val.parquet"
    train_size = write_shard(train_path, train_docs)
    val_size = write_shard(val_path, val_docs)

    print()
    print(f"  [ok] {train_path}  {len(train_docs):,} 段  {train_size / 2**20:.2f} MiB")
    print(f"  [ok] {val_path}    {len(val_docs):,} 段  {val_size / 2**20:.2f} MiB")
    print(f"  训练时：export NANOCHAT_DATA_DIR={out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
