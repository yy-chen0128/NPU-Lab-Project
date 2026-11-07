#!/usr/bin/env python3
"""
扫描 `torch_npu.npu_fusion_attention` 的 mask / sparse_mode / pre_tockens 组合。

为什么需要它
------------
昇腾融合注意力的参数语义有几处**只能靠实测确认**的坑：

  * 不传 `atten_mask` 时，`pre_tockens` / `next_tockens` 会被忽略，
    算子退化成非因果注意力（数值静默错误，不报错）；
  * 滑窗必须用 band 模式（`sparse_mode=4`），
    在因果模式（2/3）上再加 `pre_tockens` 不会生效；
  * `atten_mask` 的尺寸固定为 2048×2048，不是 (T, T)；
  * mask 的语义与 PyTorch 相反：True 表示**屏蔽**。

本脚本把 full / window 两组参数扫一遍，与 SDPA 逐元素对比，
打印 max|Δ| 和耗时。输出直接可作为进阶挑战 1 的“数值一致性证据”。

用法
----
    python scripts/check_npu_attention.py
    python scripts/check_npu_attention.py --batch 2 --seq 512 --heads 4 --window 128
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from nanochat.flash_attention import _npu_causal_mask


def require_npu():
    if not hasattr(torch, "npu"):
        print("本脚本需要 NPU（torch.npu 不存在）。")
        sys.exit(1)
    import torch_npu  # noqa: F401
    if torch.npu.device_count() == 0:
        print("没有可见的 NPU 设备。")
        sys.exit(1)
    torch.npu.set_device(0)


def timed(fn, warmup=2, iters=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def sdpa_full(q, k, v):
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        is_causal=True).transpose(1, 2)


def sdpa_window(q, k, v, w):
    T = q.size(1)
    row = torch.arange(T, device=q.device).unsqueeze(1)
    col = torch.arange(T, device=q.device).unsqueeze(0)
    keep = (col <= row) & ((row - col) <= w)
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=keep).transpose(1, 2)


def call_fused(q, k, v, mask, sparse_mode, pre=None, nxt=0):
    import torch_npu
    kw = dict(head_num=q.size(2), input_layout="BSND",
              atten_mask=mask, scale=1.0 / math.sqrt(q.size(-1)),
              sparse_mode=sparse_mode)
    if pre is not None:
        kw["pre_tockens"] = pre
        kw["next_tockens"] = nxt
    return torch_npu.npu_fusion_attention(q, k, v, **kw)[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="扫描昇腾融合注意力参数")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--window", type=int, default=128)
    args = ap.parse_args()
    require_npu()

    B, T, H, D, W = args.batch, args.seq, args.heads, args.head_dim, args.window
    torch.manual_seed(0)
    q = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="npu:0")
    k = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="npu:0")
    v = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="npu:0")
    mask = _npu_causal_mask(T, q.device)

    ref_full = sdpa_full(q, k, v).float()
    ref_win = sdpa_window(q, k, v, W).float()

    def report(label, out, ref):
        d = (out.float() - ref).abs().max().item()
        print(f"  {label:<34} max|Δ| = {d:.6e}")
        return d

    print("=" * 76)
    print(f"  融合注意力参数扫描  B={B} T={T} H={H} D={D} W={W}")
    print("=" * 76)

    print("\n[全长层] 参考 = SDPA(is_causal=True)")
    report("mask + sparse_mode=2", call_fused(q, k, v, mask, 2), ref_full)
    report("mask + sparse_mode=3", call_fused(q, k, v, mask, 3), ref_full)
    report("无 mask + sparse_mode=2", call_fused(q, k, v, None, 2), ref_full)
    report("无 mask + pre/next", call_fused(q, k, v, None, 2, pre=W), ref_full)

    print("\n[滑窗层] 参考 = SDPA(显式 (T,T) mask, (row-col)<=W)")
    for pre in (W - 1, W, W + 1, 2 * W):
        report(f"mask + sparse_mode=4 pre={pre} nxt=0",
               call_fused(q, k, v, mask, 4, pre=pre, nxt=0), ref_win)
    report("mask + sparse_mode=2 pre=W nxt=0",
           call_fused(q, k, v, mask, 2, pre=W, nxt=0), ref_win)

    print("\n[耗时] fwd, 5 iters")
    print(f"  SDPA full    {timed(lambda: sdpa_full(q, k, v)):8.3f} ms")
    print(f"  fused full   {timed(lambda: call_fused(q, k, v, mask, 2)):8.3f} ms")
    print(f"  SDPA window  {timed(lambda: sdpa_window(q, k, v, W)):8.3f} ms")
    print(f"  fused window {timed(lambda: call_fused(q, k, v, mask, 4, pre=W)):8.3f} ms")
    print("\n结论：full 必须 mask + sparse_mode=2；window 必须 mask + sparse_mode=4 + pre_tockens=W。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
