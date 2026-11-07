#!/usr/bin/env python3
"""
进阶挑战 2：不同**规模**的模型 —— 从"能跑通"到"scaling law"。

要回答的问题
------------
1. 参数量、算力、验证损失之间是什么关系？能不能拟合出一条 power law？
2. **给定固定的算力预算，应该把预算花在"更大的模型"还是"更多的数据"上？**
   这是 Chinchilla 的核心问题，也是本挑战最有价值的产出。
3. 小模型（d4/d6）到底能不能用来做代码调试？它给出的结论能外推吗？

为什么这个挑战值得做
--------------------
"小模型先跑通、再上大模型"是工程常识，但**什么时候小模型的结论不可信**
并不显然。本挑战要求你把这个边界测出来 —— 这是研究报告里最容易出彩的部分。

用法
----
    # 1) 先看每个规模要多少算力（不训练，纯估算，几秒出结果）
    python scripts/bench_scaling.py plan

    # 2) 训练一个规模（需要 NPU；小规模几分钟就能出结果）
    python scripts/bench_scaling.py train --depth 4 --epochs 1

    # 3) 汇总已训练的所有规模，拟合 scaling law
    python scripts/bench_scaling.py fit

    # 4) 一键跑完全部规模（耗时长，建议后台运行）
    python scripts/bench_scaling.py sweep --depths 4 6 8 --epochs 1

结果写在 `scaling_results.jsonl`（每行一个规模），`fit` 会读它。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 与 runs/train.sh 保持一致：课程配置关闭 value embeddings。
# 否则 plan 里的参数量会多出 value_embeds（d8 约 +25M），scaling 横轴就不对了。
os.environ.setdefault("NANOCHAT_DISABLE_VE", "1")
os.environ.setdefault("WANDB_MODE", "offline")
# 与 runs/train.sh 一致：NPU 上 TorchInductor 没有调度器，必须走 eager。
# 否则 base_train 的 compile_enabled("") 会误判为可编译并在第一个 step 崩溃。
os.environ.setdefault("NANOCHAT_DISABLE_COMPILE", "1")
os.environ.setdefault("NANOCHAT_ENFORCE_EAGER", "1")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "scaling_results.jsonl")

# 昇腾 910B bf16 的理论峰值（TFLOPS）。实测 MFU 通常 15~25%。
PEAK_TFLOPS = 376.0


# ===========================================================================
# plan：不训练，先算清楚每个规模要多少算力
# ===========================================================================

def cmd_plan(args):
    """
    用模型自己的 estimate_flops() 算每个规模的预算。

    这一节**不需要 NPU**，也不需要训练 —— 先把账算清楚，
    再决定跑哪些规模，比盲目开跑重要得多。
    """
    from nanochat.gpt import GPTConfig

    data_bytes = _corpus_bytes()
    print(f"  语料：{data_bytes / 2**20:.1f} MiB"
          f"（{args.tokens_per_byte:.2f} token/字节的假设下 ≈ "
          f"{data_bytes * args.tokens_per_byte / 1e6:.2f} M token/epoch）")
    print()
    print(f"  {'depth':>6}{'params':>12}{'FLOPs/token':>15}"
          f"{'tok:param':>11}{'1 epoch 步数':>14}{'单卡估时':>11}")
    print("  " + "-" * 70)
    for d in args.depths:
        cfg = _config_for_depth(d)
        # 参数量的解析式在 src/budget.py 里；这里用一个极小的模型实例直接数
        from nanochat.gpt import GPT
        import torch
        with torch.device("meta"):
            model = GPT(cfg)
        n_params = sum(p.numel() for p in model.parameters())
        flops = model.estimate_flops()
        tokens_per_epoch = data_bytes * args.tokens_per_byte
        steps = int(tokens_per_epoch * args.epochs // args.total_batch)
        tflops = flops * steps * args.total_batch / 1e12
        # 单位：TFLOPs / (TFLOPS) = 秒，再 /3600 得小时
        hours = tflops / (PEAK_TFLOPS * args.mfu) / 3600.0
        print(f"  {d:>6}{n_params / 1e6:>11.2f}M{flops / 1e9:>14.4f}G"
              f"{tokens_per_epoch / n_params:>11.1f}{steps:>14}"
              f"{hours:>10.2f}h")
    print("  " + "-" * 70)
    print(f"  假设：峰值 {PEAK_TFLOPS} TFLOPS × MFU {args.mfu:.0%}；"
          f"total_batch={args.total_batch}")

    # TODO(学生):
    #   1) 上面的"单卡估时"用的是假设的 MFU。**跑一个规模实测 tok/s**，
    #      反推真实 MFU，再用它重算整张表。报告里必须写明
    #      "估算值"与"实测值"分别是多少（这是实验报告的基本要求）。
    #   2) 参数量随 depth 大致按 d^2 增长、FLOPs/token 也按 d^2 增长，
    #      而 tok:param 是模型自己算出来的。请你解释：
    #      **为什么"固定算力下最优的模型大小"不等于"最大的模型"？**
    #   3) 挑出你能在时间预算内跑完的 3 个规模，说明选择理由。


# ===========================================================================
# train：训练一个规模
# ===========================================================================

def cmd_train(args):
    depth = args.depth
    tag = f"scale-d{depth}"

    iters = args.iters
    if iters is None:
        tokens_per_epoch = _corpus_bytes() * args.tokens_per_byte
        iters = max(1, int(tokens_per_epoch * args.epochs // args.total_batch))
        print(f"  epochs={args.epochs:g} -> iters={iters} "
              f"(≈{tokens_per_epoch * args.epochs / 1e6:.2f}M tokens)")

    cmd = [
        sys.executable, "-m", "scripts.base_train",
        f"--depth={depth}",
        f"--max-seq-len={args.seq}",
        f"--device-batch-size={args.device_batch}",
        f"--total-batch-size={args.total_batch}",
        f"--num-iterations={iters}",
        f"--eval-every={args.eval_every}",
        "--eval-tokens=1000000",
        "--sample-every=-1",
        "--save-every=1000",
        f"--model-tag={tag}",
        f"--run={tag}",
    ]
    if args.data_dir:
        os.environ["NANOCHAT_DATA_DIR"] = args.data_dir
    print("  " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=REPO)
    if r.returncode:
        return r.returncode

    # 训练完立刻评测，把结果写进结果文件
    bpb = _eval_bpb(tag, args)
    _append_result({
        "depth": depth, "model_tag": tag, "iters": iters,
        "seq": args.seq, "total_batch": args.total_batch,
        "val_bpb": bpb,
    })
    print(f"  已记录 {tag}: val_bpb={bpb}")
    return 0


def _eval_bpb(tag, args):
    """跑 base_eval 拿 val_bpb。**必须用 BPB 而不是 loss** —— 见下方说明。"""
    env = dict(os.environ)
    cmd = [sys.executable, "-m", "scripts.base_eval",
           "--eval=bpb", f"--model-tag={tag}",
           f"--device-batch-size={args.device_batch}",
           "--split-tokens=1000000"]
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    print(r.stdout[-2000:])
    m = re.search(r"val[^\n]*?bpb[^\n]*?([0-9]+\.[0-9]+)", r.stdout, re.I)
    if not m:
        m = re.search(r"([0-9]+\.[0-9]+)", r.stdout.split("val")[-1]
                      if "val" in r.stdout else "")
    return float(m.group(1)) if m else float("nan")


def _append_result(rec):
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ===========================================================================
# fit：拟合 scaling law
# ===========================================================================

def cmd_fit(args):
    """
    读 scaling_results.jsonl，拟合 L = a * N^(-alpha)。

    做法：对 log(L) 与 log(N) 做最小二乘（一次多项式拟合），
    斜率就是 -alpha。

    ⚠️ **只有 2 个点拟合不出可信的幂律。** 至少 3 个，最好 4 个以上，
       且规模要拉开（例如 d4/d6/d8/d12 而不是 d7/d8）。
       如果你的点都在同一个数量级，拟合出来的 alpha 没有意义 ——
       报告里要如实说明这一点（**这比硬拟合一个漂亮的数字更重要**）。
    """
    if not os.path.exists(RESULTS):
        print(f"  找不到 {RESULTS}，先跑 train 或 sweep。")
        return 1
    recs = [json.loads(l) for l in open(RESULTS, encoding="utf-8")
            if l.strip()]
    recs = [r for r in recs if r.get("val_bpb") == r.get("val_bpb")]  # 去 NaN
    if len(recs) < 2:
        print(f"  只有 {len(recs)} 个有效结果，至少要 2 个才能拟合。")
        return 1

    import math

    import numpy as np

    # 去重：同一 (depth, iters, total_batch) 只保留最后一条
    dedup = {}
    for r in recs:
        key = (r["depth"], r["iters"], r["total_batch"])
        dedup[key] = r
    recs = sorted(dedup.values(), key=lambda x: (x["depth"], x["iters"]))

    def real_params(depth):
        from nanochat.gpt import GPT
        import torch
        cfg = _config_for_depth(depth, seq=512, vocab=16384)
        with torch.device("meta"):
            model = GPT(cfg)
        return model.num_scaling_params()["total"]

    params_cache = {}
    rows = []
    for r in recs:
        d = r["depth"]
        n = params_cache.setdefault(d, real_params(d))
        D = r["iters"] * r["total_batch"]
        rows.append(dict(depth=d, N=n, D=D, L=r["val_bpb"],
                         iters=r["iters"], tag=r.get("model_tag", "")))

    print(f"  {'depth':>6}{'params':>12}{'D(tokens)':>14}{'val_bpb':>10}")
    print("  " + "-" * 44)
    for x in rows:
        print(f"  {x['depth']:>6}{x['N'] / 1e6:>10.2f}M{x['D'] / 1e6:>12.2f}M"
              f"{x['L']:>10.4f}")
    print("  " + "-" * 44)

    if len({x['depth'] for x in rows}) < 3:
        print("  ⚠️ 只有 <3 个规模，拟合没有意义；请补跑 depth。")
        return 1

    N = np.array([x["N"] for x in rows], dtype=float)
    D = np.array([x["D"] for x in rows], dtype=float)
    L = np.array([x["L"] for x in rows], dtype=float)
    n_depths = len({x["depth"] for x in rows})
    n_D = len({int(d) for d in D})

    if n_D >= 2 and n_depths >= 3:
        # L(N, D) = E + A * N^-alpha + B * D^-beta
        def fit_2d(nn, dd, ll):
            best = None
            for alpha in np.arange(0.05, 1.001, 0.025):
                for beta in np.arange(0.05, 1.001, 0.025):
                    X = np.stack([np.ones_like(nn), nn ** (-alpha),
                                  dd ** (-beta)], axis=1)
                    coef, *_ = np.linalg.lstsq(X, ll, rcond=None)
                    # 约束：E >= 0、A/B > 0，否则会拟合出负的不可约损失，
                    # 进而给出荒谬的最优规模。
                    if coef[0] < 0 or coef[1] <= 0 or coef[2] <= 0:
                        continue
                    pred = X @ coef
                    sse = float(((ll - pred) ** 2).sum())
                    if best is None or sse < best[0]:
                        best = (sse, float(alpha), float(beta), coef)
            return best

        best = fit_2d(N, D, L)
        if best is None:
            print("  ⚠️ 二项式拟合没有找到正系数解；退回单变量拟合。")
            n_D = 1
        else:
            sse, alpha, beta, coef = best
            E, A, B = (float(c) for c in coef)
            X = np.stack([np.ones_like(N), N ** (-alpha), D ** (-beta)], axis=1)
            pred = X @ coef
            r2 = 1 - sse / float(((L - L.mean()) ** 2).sum())
            print(f"\n  拟合 L = E + A·N^-α + B·D^-β")
            print(f"    E={E:.4f}  A={A:.3e}  α={alpha:.3f}")
            print(f"    B={B:.3e}  β={beta:.3f}  R²={r2:.4f}")
            if alpha <= 0.0501 or beta <= 0.0501:
                print("    ⚠️ α 或 β 落在网格下界 0.05：本实验的规模/数据跨度"
                      "不足以稳定识别指数，结论只作趋势参考。")

            # 留一验证：每次留一个 (N,D) 点，用其余点重拟合
            errs = []
            for i in range(len(N)):
                mask = np.ones(len(N), dtype=bool)
                mask[i] = False
                sub = fit_2d(N[mask], D[mask], L[mask])
                if sub is None:
                    continue
                Xi = np.stack([np.ones(1), N[i:i + 1] ** (-sub[1]),
                               D[i:i + 1] ** (-sub[2])], axis=1)
                pred_i = float((Xi @ sub[3])[0])
                errs.append(abs(pred_i - L[i]))
            if errs:
                print(f"    留一验证 max|Δ|={max(errs):.4f}  mean|Δ|={np.mean(errs):.4f}")

            # 固定算力 C=6ND，求最优 N
            observed_budget = 6 * float(N.max()) * float(D.max())
            budget = float(os.environ.get("SCALING_BUDGET_FLOPS",
                                          str(observed_budget)))
            ng = np.logspace(np.log10(N.min()), np.log10(N.max()), 200)
            dg = budget / (6 * ng)
            ok = dg >= 1
            lg = E + A * ng[ok] ** (-alpha) + B * dg[ok] ** (-beta)
            j = int(np.argmin(lg))
            n_opt, d_opt = float(ng[ok][j]), float(dg[ok][j])
            print(f"\n  固定算力 C={budget:.2e} FLOPs（C=6ND，"
                  f"默认取 6·N_max·D_max）:")
            print(f"    实测范围：N={N.min() / 1e6:.2f}–{N.max() / 1e6:.2f}M，"
                  f"D={D.min() / 1e6:.2f}–{D.max() / 1e6:.2f}M")
            print(f"    N_opt={n_opt / 1e6:.2f}M  D_opt={d_opt / 1e6:.2f}M tokens"
                  f"  D/N={d_opt / n_opt:.1f}")
            if j == 0 or j == len(lg) - 1:
                print("    ⚠️ 最优解落在搜索边界：这个预算下 L 对 N 仍单调，"
                      "外推不可信；报告里要如实说明。")
            if E < 0 or alpha <= 0.0501 or beta <= 0.0501:
                print("    ⚠️ 参数不可靠，不建议把 N_opt 当成定量结论；"
                      "应补更大规模（d8/d12）或更长训练后再拟合。")
            ratio = d_opt / n_opt
            if ratio > 30:
                print("    趋势：同等算力下更偏向增加数据（D/N 偏大）。")
            elif ratio < 10:
                print("    趋势：同等算力下更偏向增大模型（D/N 偏小）。")
            else:
                print("    趋势：模型与数据大致均衡（Chinchilla 经验值约 D/N≈20）。")
            return 0

    # 只有一种 D：固定 D 下拟合 L = a * N^-alpha
    good = L > 0
    x, y = np.log(N[good]), np.log(L[good])
    slope, intercept = np.polyfit(x, y, 1)
    alpha = float(-slope)
    a = float(np.exp(intercept))
    pred = a * N ** (-alpha)
    ss_res = float(((L - pred) ** 2).sum())
    ss_tot = float(((L - L.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"\n  拟合 L = a·N^-α（D 固定）")
    print(f"    a={a:.6f}  α={alpha:.4f}  R²={r2:.4f}")

    # 留一验证 + 去掉最大规模后的 alpha
    loo = []
    for i in range(len(N)):
        mask = np.ones(len(N), dtype=bool)
        mask[i] = False
        if mask.sum() < 2:
            continue
        s, ic = np.polyfit(np.log(N[mask]), np.log(L[mask]), 1)
        p = np.exp(ic) * N[i] ** (-s)
        loo.append(abs(p - L[i]))
    if loo:
        print(f"    留一验证 max|Δ|={max(loo):.4f}  mean|Δ|={np.mean(loo):.4f}")
    if len(N) >= 3:
        s2, ic2 = np.polyfit(np.log(N[:-1]), np.log(L[:-1]), 1)
        print(f"    去掉最大规模后 α={-float(s2):.4f}"
              f"（与全量 α={alpha:.4f} 的差 = {abs(-float(s2) - alpha):.4f}）")

    print("\n  ⚠️ 所有点只在同一个 D 上，无法识别数据项；固定 D 时 L 随 N 单调下降，"
          "\n     最优 N 就是预算允许的最大值。要回答“模型 vs 数据”，"
          "\n     至少跑两个不同 epochs（例如 --epochs 1 和 --epochs 4）再 fit。")
    return 0


# ===========================================================================
# sweep
# ===========================================================================

def cmd_sweep(args):
    for d in args.depths:
        print(f"\n{'=' * 68}\n  训练 depth={d}\n{'=' * 68}")
        args.depth = d
        r = cmd_train(args)
        if r:
            print(f"  depth={d} 失败（返回码 {r}），继续下一个")
    print("\n  全部完成，现在跑 `fit`：")
    print(f"    {sys.executable} {os.path.relpath(__file__)} fit")
    return 0


# ===========================================================================

def _config_for_depth(depth: int, seq: int = 2048, vocab: int = 16384):
    """
    构造 depth 对应的模型配置。

    宽度公式与 nanochat 一致：model_dim = ceil(64 * depth / 128) * 128，
    并且使用固定 head_dim=128（与 base_train 的 n_head = model_dim // 128
    一致，否则深度 2 的注意力 FLOPs 会和实际训练对不上）。
    """
    from nanochat.gpt import GPTConfig
    n_embd = int(-(-(64 * depth) // 128) * 128)
    n_head = max(1, n_embd // 128)
    return GPTConfig(sequence_len=seq, vocab_size=vocab, n_layer=depth,
                     n_head=n_head, n_kv_head=n_head, n_embd=n_embd,
                     window_pattern="SSSL")


def _corpus_bytes():
    """语料字节数（用于估算每 epoch token 数）。

    NANOCHAT_DATA_DIR 指向 parquet 时，parquet 是压缩格式，直接拿字节数
    会低估 token 数；因此优先用随包的 raw corpus（同一批文本）估算。
    """
    default = os.path.join(REPO, "data", "corpus")
    d = os.environ.get("NANOCHAT_DATA_DIR") or default

    def _sum(dirname, exts):
        total = 0
        for root, _dirs, files in os.walk(dirname):
            for fn in files:
                if fn.endswith(exts):
                    total += os.path.getsize(os.path.join(root, fn))
        return total

    total = _sum(d, (".txt", ".jsonl"))
    if total == 0 and os.path.isdir(default):
        total = _sum(default, (".txt", ".jsonl"))
    if total == 0:
        total = _sum(d, (".parquet",))
    return total or 1


def main() -> int:
    ap = argparse.ArgumentParser(description="进阶挑战 2：模型规模对比与 scaling law")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--depths", type=int, nargs="*",
                       default=[4, 6, 8])
        p.add_argument("--epochs", type=float, default=1.0)
        p.add_argument("--seq", type=int, default=2048)
        p.add_argument("--total-batch", type=int, default=32768,
                       help="起手语料小，默认用较小的 batch 以保证步数足够")
        p.add_argument("--device-batch", type=int, default=16)
        p.add_argument("--mfu", type=float, default=0.20)
        p.add_argument("--tokens-per-byte", type=float, default=0.25)
        p.add_argument("--data-dir", default=None)

    p = sub.add_parser("plan", help="只估算算力，不训练")
    common(p)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("train", help="训练一个规模")
    common(p)
    p.add_argument("--depth", type=int, required=True)
    p.add_argument("--iters", type=int, default=None,
                   help="显式步数；不填则按 --epochs 与语料大小估算")
    p.add_argument("--eval-every", type=int, default=50)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("sweep", help="依次训练多个规模")
    common(p)
    p.add_argument("--iters", type=int, default=None,
                   help="显式步数；不填则按 --epochs 与语料大小估算")
    p.add_argument("--eval-every", type=int, default=50)
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("fit", help="汇总并拟合")
    p.set_defaults(func=cmd_fit)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
