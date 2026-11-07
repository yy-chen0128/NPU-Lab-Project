#!/usr/bin/env python3
"""
进阶挑战 1：注意力与归一化的**实现方式**对性能与精度的影响。

要回答的问题
------------
同样的数学、不同的实现，在昇腾 910B 上差多少？

  1. 全长注意力：SDPA（构造 T×T mask）vs 昇腾融合算子
  2. 滑窗注意力：同上
  3. RMSNorm：原生 PyTorch vs npu_rms_norm 融合算子
  4. 每一种都要给出**数值一致性**证据（不能只看快慢）

为什么全长层和滑窗层要分开看
----------------------------
这是本挑战最值得写进报告的一点：

                    SDPA       融合算子     加速比
    滑窗层          13.42 ms    6.70 ms     2.00×
    全长层           9.18 ms   12.42 ms     0.74×   ← 反而更慢

SDPA 在 **Tq == Tk 且全长**时会走 PyTorch 自己的融合内核（很快）；
而在**滑窗**时必须物化一张 (T,T) 的布尔 mask 再算，于是用不上任何融合。
所以融合算子的收益**只出现在滑窗层**，而 `window_pattern="SSSL"`
意味着 3/4 的层都是滑窗 —— 这才是它值得做的原因。

**如果你的结论是"融合算子全面更快"，那是没分开测。**

用法
----
    python scripts/bench_attention.py                  # 跑全部
    python scripts/bench_attention.py --only rmsnorm
    python scripts/bench_attention.py --batch 32 --seq 2048

输出：一张对照表（延迟 / 加速比 / 数值误差 / 显存）。

⚠️ 必须在**有 NPU 的机器**上跑。没有 NPU 时会打印提示并退出。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 与 runs/train.sh / configs/model_d8.yaml 保持一致
os.environ.setdefault("NANOCHAT_DISABLE_VE", "1")

import torch  # noqa: E402

# 参考形状：本课程 d8 模型的真实配置
DEF_BATCH, DEF_SEQ, DEF_HEADS, DEF_HEAD_DIM, DEF_WINDOW = 32, 2048, 4, 128, 512


def require_npu():
    if not hasattr(torch, "npu"):
        print("本脚本需要 NPU（torch.npu 不存在）。请在昇腾机器上运行。")
        sys.exit(0)
    try:
        import torch_npu  # noqa: F401
    except Exception as e:                                 # noqa: BLE001
        print(f"torch_npu 不可用: {e}")
        sys.exit(1)
    if torch.npu.device_count() == 0:
        print("没有可见的 NPU 设备。")
        sys.exit(1)
    torch.npu.set_device(0)


def timed(fn, warmup=3, iters=10):
    """返回平均毫秒数。**必须预热** —— 首次调用包含算子编译/tiling 开销。"""
    for _ in range(warmup):
        fn()
    if hasattr(torch, "npu"):
        torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if hasattr(torch, "npu"):
        torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


# ===========================================================================
# 任务 1：RMSNorm
# ===========================================================================

def bench_rmsnorm(args):
    """
    对比三种 RMSNorm：
      a) 你的原生实现          nanochat.gpt.norm
      b) PyTorch 内置          F.rms_norm
      c) 昇腾融合              torch_npu.npu_rms_norm

    要看的现象：本模型每层 4 次 RMSNorm、8 层共 **32 次/前向**，
    eager 模式下每次都是若干个逐元素 kernel —— 所以它的绝对收益
    可能不比换注意力小。
    """
    import torch.nn.functional as F
    from nanochat.gpt import norm

    B, T, C = args.batch, args.seq, 768
    dev = torch.device("npu:0")
    x = torch.randn(B, T, C, dtype=torch.bfloat16, device=dev)
    w = torch.ones(C, dtype=torch.bfloat16, device=dev)

    # 用 float64 算"真值" —— 判断谁更准必须有一个独立基准。
    # ⚠️ NPU 不支持 float64：x.double() 在 NPU 上会被静默转回 float32，
    # 所以真值必须在 CPU 上用 float64 算，否则这一列不是 fp64。
    truth = F.rms_norm(x.float().cpu().double(), (C,), eps=1e-6)

    impls = [
        ("你的原生实现", lambda: norm(x)),
        ("F.rms_norm", lambda: F.rms_norm(x, (C,), eps=1e-6)),
    ]
    if _has_fused_rms():
        impls.append(("npu_rms_norm (融合)",
                      lambda: torch_npu_rms(x, w)))
    else:
        print("  [skip] 当前 torch_npu 没有 npu_rms_norm")

    print(f"\n  RMSNorm  ({tuple(x.shape)}, bf16)")
    print(f"  {'实现':<22}{'延迟(ms)':>10}{'加速比':>9}{'vs 原生':>11}"
          f"{'vs fp64 真值':>14}")
    print("  " + "-" * 68)
    base_ms = None
    base_out = None
    for name, fn in impls:
        ms = timed(fn)
        out = fn().float()
        if base_ms is None:
            base_ms, base_out = ms, out
        sp = base_ms / ms
        d_native = (out - base_out).abs().max().item()
        d_truth = (out.float().cpu().double() - truth).abs().max().item()
        print(f"  {name:<22}{ms:>10.3f}{sp:>8.2f}x{d_native:>11.3e}"
              f"{d_truth:>14.3e}")
    print("  " + "-" * 68)

    # TODO(学生): 回答下面三个问题（报告里要有数据支撑）：
    #   1) 融合算子的加速比是多少？与你预估的一致吗？
    #   2) 融合算子与你的实现在数值上"不一样"，这是精度损失吗？
    #      看最后一列 —— 比较**各自相对 fp64 真值**的偏差，再下结论。
    #      实测结论：两者相对真值的偏差量级相同，差异只是 bf16 的固有舍入。
    #      （另注：F.rms_norm 在 torch 2.12 下不传 eps 时的默认值实测是 1e-6，
    #        不是文档写的 finfo(dtype).eps；bf16 的 finfo.eps=0.0078 会让
    #        结果差 3e-2 —— 差的是 eps，不是算子。）
    #   3) 按"每前向 32 次"折算，端到端能省多少？


def _has_fused_rms():
    if not hasattr(torch, "npu"):
        return False
    try:
        import torch_npu
    except Exception:                                      # noqa: BLE001
        return False
    return hasattr(torch_npu, "npu_rms_norm")


def torch_npu_rms(x, w, eps=1e-6):
    """昇腾融合 RMSNorm。返回 tuple，取 [0]；eps 必须与原生实现一致（1e-6）。"""
    import torch_npu
    return torch_npu.npu_rms_norm(x, w, epsilon=eps)[0]


# ===========================================================================
# 任务 2：注意力（全长 vs 滑窗）
# ===========================================================================

def bench_attention(args):
    """
    对比 SDPA 与昇腾融合注意力，**全长层与滑窗层分开测**。

    注意 q/k/v 的布局：
      * 我们的接口是 (B, T, H, D)（BSND）
      * SDPA 要 (B, H, T, D) -> 需要转置
      * 融合算子直接吃 (B, T, H, D) -> **不需要转置**
    所以"融合算子更快"的原因里也包含省掉两次转置，报告里可以量化。
    """
    from nanochat.flash_attention import _sdpa_attention, _npu_attention

    B, T, H, D = args.batch, args.seq, args.heads, args.head_dim
    dev = torch.device("npu:0")
    torch.manual_seed(0)
    q = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=dev)
    k = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=dev)
    v = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=dev)

    for label, window in (("全长层", (-1, -1)),
                          (f"滑窗层 (w={args.window})", (args.window, 0))):
        print(f"\n  注意力 · {label}   ({tuple(q.shape)}, bf16)")

        def sdpa():
            # BSND -> BHSD -> BSND
            return _sdpa_attention(q.transpose(1, 2), k.transpose(1, 2),
                                   v.transpose(1, 2), window, False
                                   ).transpose(1, 2)

        def fused():
            return _npu_attention(q, k, v, window)

        with torch.no_grad():
            ref = sdpa().float()
        ms_sdpa = timed(sdpa)
        ms_fused = timed(fused)

        with torch.no_grad():
            out = fused().float()
        maxdiff = (out - ref).abs().max().item()

        # fwd+bwd：训练真正关心的是这一列
        qg = q.clone().requires_grad_(True)
        kg = k.clone().requires_grad_(True)
        vg = v.clone().requires_grad_(True)
        qf = q.clone().requires_grad_(True)
        kf = k.clone().requires_grad_(True)
        vf = v.clone().requires_grad_(True)

        def zero(tensors):
            for t in tensors:
                t.grad = None

        def sdpa_bwd():
            zero((qg, kg, vg))
            out = _sdpa_attention(qg.transpose(1, 2), kg.transpose(1, 2),
                                  vg.transpose(1, 2), window, False
                                  ).transpose(1, 2)
            out.float().sum().backward()

        def fused_bwd():
            zero((qf, kf, vf))
            out = _npu_attention(qf, kf, vf, window)
            out.float().sum().backward()

        ms_sdpa_bwd = timed(sdpa_bwd, warmup=2, iters=5)
        ms_fused_bwd = timed(fused_bwd, warmup=2, iters=5)

        print(f"    {'':<12}{'fwd(ms)':>10}{'fwd+bwd(ms)':>14}"
              f"{'加速(fwd)':>11}{'加速(fwd+bwd)':>14}")
        print(f"    {'SDPA':<12}{ms_sdpa:>10.3f}{ms_sdpa_bwd:>14.3f}")
        print(f"    {'融合算子':<12}{ms_fused:>10.3f}{ms_fused_bwd:>14.3f}"
              f"{ms_sdpa / ms_fused:>10.2f}x"
              f"{ms_sdpa_bwd / ms_fused_bwd:>13.2f}x")
        print(f"    max|Δ| (vs SDPA) = {maxdiff:.3e}")


# ===========================================================================
# 任务 3：训练步级别的端到端对比
# ===========================================================================

def bench_train_step(args):
    """
    算子级微基准有偏差：真实训练里还穿插着反向、优化器、通信。
    这里跑真实的训练步，看**端到端**收益还剩多少。

    TODO(学生):
      1) 构造一个小 GPT（例如 d8 的宽度但 2 层就够），
         分别在以下配置下跑若干个训练步，记录 tok/s 与单步耗时：
            A) 全原生：NANOCHAT_ATTN_IMPL=sdpa, NANOCHAT_RMSNORM_IMPL=native
            B) 只换 RMSNorm
            C) 只换注意力
            D) 都换
      2) 用矩阵形式报告，并回答"1+1 是否等于 2"（两个优化的收益能否叠加）。
      3) 注意：eager 模式下算子级微基准的收益通常**会被放大**，
         因为 eager 的 kernel 启动开销占比更高。报告里要说明这一点。
    """
    import nanochat.gpt as gpt_mod
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.flash_attention import HAS_NPU_FA

    if not HAS_NPU_FA:
        print("  [skip] 当前 torch_npu 没有 npu_fusion_attention")
        return
    if not _has_fused_rms():
        print("  [skip] 当前 torch_npu 没有 npu_rms_norm")
        return

    B, T = 16, args.seq
    cfg = GPTConfig(sequence_len=T, vocab_size=16384, n_layer=2,
                    n_head=4, n_kv_head=4, n_embd=512,
                    window_pattern="SSSL")
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device="npu:0")
    model.init_weights()
    model.train()
    opt = model.setup_optimizer()

    x = torch.randint(0, cfg.vocab_size, (B, T), device="npu:0")
    y = torch.randint(0, cfg.vocab_size, (B, T), device="npu:0")

    def reset_norm():
        # norm() 会把解析结果缓存起来，换实现前必须清掉
        gpt_mod._NPU_RMS_STATE.update(resolved=False, fn=None)

    configs = [
        ("A 全原生", "sdpa", "native"),
        ("B 只换 RMSNorm", "sdpa", "npu"),
        ("C 只换注意力", "npu_fa", "native"),
        ("D 都换", "npu_fa", "npu"),
    ]
    steps = 10
    results: dict[str, float] = {}

    print(f"\n  训练步对比（2 层 / n_embd=512 / B={B} / T={T} / {steps} steps）")
    print(f"  {'配置':<18}{'ms/step':>10}{'tok/s':>12}{'相对 A':>10}")
    print("  " + "-" * 52)
    for name, attn_impl, rms_impl in configs:
        os.environ["NANOCHAT_ATTN_IMPL"] = attn_impl
        os.environ["NANOCHAT_RMSNORM_IMPL"] = rms_impl
        reset_norm()

        for _ in range(2):  # warmup
            opt.zero_grad(set_to_none=True)
            loss = model(x, y)
            loss.backward()
            opt.step()
        torch.npu.synchronize()

        t0 = time.perf_counter()
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            loss = model(x, y)
            loss.backward()
            opt.step()
        torch.npu.synchronize()
        ms = (time.perf_counter() - t0) / steps * 1000
        results[name] = ms
        tok_s = B * T / (ms / 1000)
        base = results.get("A 全原生", ms)
        print(f"  {name:<18}{ms:>10.2f}{tok_s:>12.0f}{base / ms:>9.2f}x")

    base = results["A 全原生"]
    red_b = 1 - results["B 只换 RMSNorm"] / base
    red_c = 1 - results["C 只换注意力"] / base
    red_d = 1 - results["D 都换"] / base
    print("\n  收益分解（相对 A 的耗时下降）：")
    print(f"    B 只换 RMSNorm   -{red_b:6.1%}")
    print(f"    C 只换注意力     -{red_c:6.1%}")
    print(f"    B + C 简单相加   -{red_b + red_c:6.1%}")
    print(f"    D 实测           -{red_d:6.1%}")
    print("  注意：eager 模式下 kernel 启动占比更高，算子级加速比通常会被放大；"
          "端到端收益要小一些。")


def main() -> int:
    ap = argparse.ArgumentParser(description="进阶挑战 1：实现方式的性能对比")
    ap.add_argument("--only", choices=["rmsnorm", "attn", "train"],
                    help="只跑其中一项")
    ap.add_argument("--batch", type=int, default=DEF_BATCH)
    ap.add_argument("--seq", type=int, default=DEF_SEQ)
    ap.add_argument("--heads", type=int, default=DEF_HEADS)
    ap.add_argument("--head-dim", type=int, default=DEF_HEAD_DIM)
    ap.add_argument("--window", type=int, default=DEF_WINDOW)
    args = ap.parse_args()

    require_npu()
    print("=" * 68)
    print("  进阶挑战 1：注意力与归一化的实现方式对比")
    print(f"  形状 batch={args.batch} seq={args.seq} heads={args.heads} "
          f"head_dim={args.head_dim}")
    print("=" * 68)

    todo = args.only or "all"
    if todo in ("all", "rmsnorm"):
        bench_rmsnorm(args)
    if todo in ("all", "attn"):
        bench_attention(args)
    if todo in ("all", "train"):
        bench_train_step(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
