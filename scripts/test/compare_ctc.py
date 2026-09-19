# -*- coding: utf-8 -*-
"""
CTC 前向算法对拍验收 —— 手写实现 vs torch.nn.CTCLoss

╔════════════════════════════════════════════════════════════════════════════════╗
║  接口契约：scripts/test/CTC.py 里的实现函数（名字叫 ctc_forward 或 my_ctc_forward 都认）║
╚════════════════════════════════════════════════════════════════════════════════╝

    def my_ctc_forward(log_probs, targets, input_lengths, target_lengths, blank=0):
        '''
        log_probs      : (T, N, C) float32，**已经是 log 概率**（log_softmax 之后）
        targets        : (N, S) int64，标签 id；blank=0 时标签取值必须是 1..C-1
        input_lengths  : (N,)   int64，每个样本的有效帧数（<= T）
        target_lengths : (N,)   int64，每个样本的有效标签数（<= S）
        blank          : blank 的 id，默认 0 —— **必须与 torch.nn.CTCLoss 默认一致**

        返回            : (N,) float32 —— 每个样本的 CTC loss = -log P(target | log_probs)
                          对齐 torch.nn.functional.ctc_loss(..., reduction='none',
                                                            zero_infinity=False)
                          不可达样本（帧数不够）返回 +inf
        '''

验收判据
--------
1. 与 torch.nn.CTCLoss 逐样本 |Δ| < 1e-3（T=100 也必须达到；log 域递推的误差应远小于此）
2. T=100 / L=20 不出现 -inf / NaN —— 数值稳定性（概率域累乘会下溢）
3. 边界：L=0、L=1、T==L、T 不足、相邻重复标签
4. 自检：内置暴力枚举（**纯定义，非 DP**）与 torch.nn.CTCLoss 一致，证明对拍基线本身可信

用法
----
    set PYTHONIOENCODING=utf-8
    F:\\embedded\\prepare\\.venv\\Scripts\\python.exe scripts\\test\\compare_ctc.py             # 跑 CTC.py
    F:\\embedded\\prepare\\.venv\\Scripts\\python.exe scripts\\test\\compare_ctc.py --impl brute  # 只跑自检
"""
import argparse
import importlib.util
import itertools
import math
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
CTC_PATH = os.path.join(HERE, "CTC.py")
TOL = 1e-3          # 逐样本 loss 容差
NEG_INF = float("-inf")
POS_INF = float("inf")


# ─────────────────────────────────────────────────────────────── 实现对加载
def load_ctc_forward():
    """从 CTC.py 加载实现；文件为空 / 缺函数时给出明确提示"""
    if not os.path.exists(CTC_PATH):
        print(f"  ❌ 找不到 {CTC_PATH}")
        return None
    if os.path.getsize(CTC_PATH) == 0:
        print(f"  ⚠️  {CTC_PATH} 是**空文件（0 字节）** —— 没有实现对拍。")
        print("      请把 CTC 前向实现写进去（接口契约见本文件顶部），或先跑 --impl brute 自检。")
        return None

    spec = importlib.util.spec_from_file_location("ctc_impl", CTC_PATH)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:                                  # noqa: BLE001
        print(f"  ❌ 导入 CTC.py 失败：{type(e).__name__}: {e}")
        return None

    for name in ("ctc_forward", "my_ctc_forward"):          # 两种命名都接受
        fn = getattr(mod, name, None)
        if callable(fn):
            print(f"  已加载实现：CTC.py::{name}")
            return fn
    print("  ❌ CTC.py 里找不到 `ctc_forward` / `my_ctc_forward`（签名见本文件顶部契约）")
    return None


# ─────────────────────────────────────────────── 暴力枚举（纯定义，非 DP）
def _collapse(seq, blank):
    """CTC 折叠规则：先合并相邻重复，再删掉 blank"""
    out, prev = [], None
    for s in seq:
        if s != prev and s != blank:
            out.append(s)
        prev = s
    return tuple(out)


def brute_force_logprob(log_probs_one, target, blank=0):
    """
    【纯定义】枚举全部 V^T 条帧级路径，把折叠后等于 target 的概率加起来。
    只用于极小规模（V^T 必须能枚举完）。它**不是** α 递推，是独立的真值来源。
    log_probs_one: (T, V)
    """
    T, V = log_probs_one.shape
    total = torch.tensor(NEG_INF)
    idx_t = torch.arange(T)
    for seq in itertools.product(range(V), repeat=T):
        if _collapse(seq, blank) == tuple(target.tolist()):
            total = torch.logaddexp(
                total, log_probs_one[idx_t, torch.tensor(seq)].sum())
    return total


def brute_forward(log_probs, targets, input_lengths, target_lengths, blank=0):
    """把暴力枚举包成与 my_ctc_forward 相同的接口（返回 loss），只支持极小规模"""
    N = log_probs.size(1)
    out = []
    for n in range(N):
        t_n, s_n = int(input_lengths[n]), int(target_lengths[n])
        out.append(-brute_force_logprob(log_probs[:t_n, n, :], targets[n, :s_n], blank))
    return torch.stack(out).float().reshape(-1)


# ──────────────────────────────────────────────────────────────── 用例工厂
def make_case(T, N, C, lens, seed=0, blank=0):
    """随机生成一组 (log_probs, targets, input_lengths, target_lengths)。
       注意标签取值落在 1..C-1（避开 blank=0）"""
    g = torch.Generator().manual_seed(seed)
    log_probs = torch.log_softmax(torch.randn(T, N, C, generator=g), dim=-1)
    S = max(lens) if lens else 1
    targets = torch.zeros(N, S, dtype=torch.long)
    for n, L in enumerate(lens):
        if L > 0:
            targets[n, :L] = torch.randint(1, C, (L,), generator=g)
    return (log_probs,
            targets,
            torch.full((N,), T, dtype=torch.long),
            torch.tensor(lens, dtype=torch.long))


def ref_loss(log_probs, targets, input_lengths, target_lengths, blank=0):
    """官方参照：nn.CTCLoss(reduction='none', zero_infinity=False) → (N,) loss"""
    loss = nn.CTCLoss(blank=blank, reduction="none", zero_infinity=False)
    return loss(log_probs, targets, input_lengths, target_lengths)


# ─────────────────────────────────────────────────────────────────── 比较器
def cmp_one(name, case, fn, note=""):
    log_probs, targets, ilen, tlen = case
    ref = ref_loss(log_probs, targets, ilen, tlen, 0)

    try:
        mine = fn(log_probs, targets, ilen, tlen, 0)
    except NotImplementedError as e:
        print(f"  {name:<32}{'SKIP':<7}尚未实现: {e}")
        return None
    except TypeError:
        # 实现没写 blank 参数 → 退回 4 参数调用
        try:
            mine = fn(log_probs, targets, ilen, tlen)
        except Exception as e:                              # noqa: BLE001
            print(f"  {name:<32}{'ERROR':<7}{type(e).__name__}: {e}")
            return False
    except Exception as e:                                  # noqa: BLE001
        print(f"  {name:<32}{'ERROR':<7}{type(e).__name__}: {e}")
        return False

    mine = mine.detach().float().reshape(-1)
    if mine.numel() != ref.numel():
        print(f"  {name:<32}{'FAIL':<7}返回 {tuple(mine.shape)}（{mine.numel()} 个），"
              f"应为 (N,) = {tuple(ref.shape)}；是否忘了 reduction='none' 的对齐语义？")
        return False

    # ---- 语义诊断：是不是把 log P 当 loss 返回了（两者互为相反数）----
    fin = torch.isfinite(mine) & torch.isfinite(ref)
    sign_note = ""
    if fin.any():
        d_same = (mine[fin] - ref[fin]).abs().mean().item()
        d_flip = (mine[fin] + ref[fin]).abs().mean().item()
        if d_flip < d_same * 1e-3:
            mine = -mine
            sign_note = "  ⚠️检测到返回的是 log P，已取反后比较（语义差异，非算法错误）"

    # ---- 不可达样本：双方必须同为 +inf ----
    inf_m, inf_r = torch.isinf(mine), torch.isinf(ref)
    if not torch.equal(inf_m, inf_r):
        print(f"  {name:<32}{'FAIL':<7}不可达样本判定不一致")
        print(f"       mine 中 inf 的样本 = {inf_m.nonzero().flatten().tolist()}")
        print(f"       ref  中 inf 的样本 = {inf_r.nonzero().flatten().tolist()}")
        return False
    if (mine[inf_m] < 0).any():
        print(f"  {name:<32}{'FAIL':<7}不可达样本返回了 -inf，应为 +inf（loss 语义）")
        return False

    ok_mask = ~inf_m
    if ok_mask.any():
        d = (mine[ok_mask] - ref[ok_mask]).abs()
        rel = d / ref[ok_mask].abs().clamp(min=1e-6)
        max_d, mean_d, max_rel = d.max().item(), d.mean().item(), rel.max().item()
        passed = max_d < TOL
    else:
        max_d = mean_d = max_rel = 0.0
        passed = True

    print(f"  {name:<32}{'PASS' if passed else 'FAIL':<7}max|Δ|={max_d:.2e}  "
          f"mean={mean_d:.2e}  maxrel={max_rel:.2e}"
          f"{'  (含不可达样本)' if inf_m.any() else ''}{sign_note}{note}")
    return passed


# ──────────────────────────────────────────────────────────────────── 各段
def section_selfcheck():
    print("=" * 96)
    print("A) 对拍基线自检：暴力枚举（纯定义，非 DP）vs torch.nn.CTCLoss")
    print("=" * 96)
    cases = [
        ("T=3 N=1 C=4  L=[2]",  make_case(3, 1, 4, [2], seed=1)),
        ("T=4 N=2 C=3  L=[2,1]", make_case(4, 2, 3, [2, 1], seed=2)),
        ("T=2 N=1 C=3  L=[0]",  make_case(2, 1, 3, [0], seed=3)),
        ("T=5 N=1 C=3  L=[2] 重复", None),
    ]
    g = torch.Generator().manual_seed(4)
    cases[3] = ("T=5 N=1 C=3  L=[2] 重复",
                (torch.log_softmax(torch.randn(5, 1, 3, generator=g), dim=-1),
                 torch.tensor([[1, 1]]), torch.tensor([5]), torch.tensor([2])))
    res = [cmp_one(n, c, brute_forward) for n, c in cases]
    ok = all(res)
    print(f"  → 基线自检 {'通过 ✅' if ok else '失败 ❌'}"
          f"（说明对拍参照本身可信，可用于评判手写实现）")
    return ok


def section_main(fn):
    print()
    print("=" * 96)
    print("B) 主对拍：CTC.py vs torch.nn.CTCLoss")
    print("=" * 96)
    if fn is None:
        print("  跳过（无实现）")
        return None
    cases = [
        ("小规模 T=20 N=4 C=12 L=5",   make_case(20, 4, 12, [5] * 4, seed=11)),
        ("中规模 T=50 N=8 C=30 L=10",  make_case(50, 8, 30, [10] * 8, seed=12)),
        ("变长标签 T=30 N=4 C=15",     make_case(30, 4, 15, [1, 4, 7, 9], seed=13)),
        ("长序列 T=100 N=3 C=25 L=15", make_case(100, 3, 25, [15] * 3, seed=14)),
    ]
    return all(cmp_one(n, c, fn) for n, c in cases)


def section_stability(fn):
    print()
    print("=" * 96)
    print("C) 数值稳定性：T=100 / L=20 不得出现 -inf / NaN")
    print("=" * 96)
    if fn is None:
        print("  跳过（无实现）")
        return None

    case = make_case(100, 2, 20, [20, 20], seed=21)
    log_probs, targets, ilen, tlen = case
    try:
        out = fn(log_probs, targets, ilen, tlen, 0).detach().float().reshape(-1)
    except NotImplementedError as e:
        print(f"  SKIP  尚未实现: {e}")
        return None
    except Exception as e:                                  # noqa: BLE001
        print(f"  ERROR {type(e).__name__}: {e}")
        return False

    ref = ref_loss(log_probs, targets, ilen, tlen, 0)
    bad = ~torch.isfinite(out)
    d = (out - ref).abs().max().item()
    print(f"  手写: {[round(v, 5) for v in out.tolist()]}")
    print(f"  参照: {[round(v, 5) for v in ref.tolist()]}")
    print(f"  非有限值个数 = {int(bad.sum())}   max|Δ| = {d:.2e}")
    ok = bool(bad.sum() == 0) and d < TOL
    print(f"  → {'PASS ✅ 无下溢、无 NaN' if ok else 'FAIL ❌ 出现下溢/NaN 或偏差超限'}")
    if not ok:
        print("     提示：log 域递推必须全程用 logaddexp；概率域累乘在 T=100 时必然下溢到 0 → log = -inf")
    return ok


def section_edges(fn):
    print()
    print("=" * 96)
    print("D) 边界用例")
    print("=" * 96)
    if fn is None:
        print("  跳过（无实现）")
        return None

    def manual(T, N, C, tgt, lens, seed):
        g = torch.Generator().manual_seed(seed)
        lp = torch.log_softmax(torch.randn(T, N, C, generator=g), dim=-1)
        t = torch.zeros(N, max(lens) if lens else 1, dtype=torch.long)
        for n, L in enumerate(lens):
            if L:
                t[n, :L] = torch.tensor(tgt[n][:L])
        return (lp, t, torch.full((N,), T, dtype=torch.long), torch.tensor(lens))

    cases = [
        ("L=0（空标签）T=5",         make_case(5, 1, 6, [0], seed=31)),
        ("L=1 T=5",                  make_case(5, 1, 6, [1], seed=32)),
        ("T==L 刚好够 T=5 L=5",       manual(5, 1, 8, [[1, 2, 3, 4, 5]], [5], 33)),
        ("相邻重复标签 T=9 L=4",      manual(9, 1, 8, [[1, 1, 2, 2]], [4], 34)),
        ("帧数不足（应 +inf）T=3 L=4", manual(3, 1, 8, [[1, 2, 3, 4]], [4], 35)),
        ("单帧 T=1 L=0",             make_case(1, 1, 4, [0], seed=36)),
        ("单帧 T=1 L=1",             make_case(1, 1, 4, [1], seed=37)),
    ]
    res = [cmp_one(n, c, fn) for n, c in cases]

    print("  " + "-" * 92)
    print("  相邻重复标签的可行性边界（[1,1,2,2] 最少 6 帧：1 blank 1 2 blank 2）")
    for T in (5, 6, 7):
        c = manual(T, 1, 8, [[1, 1, 2, 2]], [4], 40 + T)
        res.append(cmp_one(f"T={T} L=4 相邻重复", c, fn,
                           note="  期望 " + ("+inf（不可达）" if T < 6 else "有限值")))
    return all(r for r in res if r is not None)


# ───────────────────────────────────────────────────────────────────── main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["ctc", "brute"], default="ctc",
                    help="ctc=跑 CTC.py（默认）；brute=只用暴力枚举跑自检")
    ap.add_argument("--selfcheck-only", action="store_true")
    args = ap.parse_args()

    print(f"torch {torch.__version__}   容差 |Δ| < {TOL:g}   （实现返回 loss，含 blank 参数）")

    ok_base = section_selfcheck()

    if args.impl == "brute" or args.selfcheck_only:
        print("\n" + "=" * 96)
        print("（--impl brute 模式：跳过 CTC.py，仅验证对拍基线）")
        print(f"总结果: {'✅ 对拍基线自检通过' if ok_base else '❌ 对拍基线自检失败'}")
        return

    fn = load_ctc_forward()
    if fn:
        print()
    r_main = section_main(fn)
    r_stab = section_stability(fn)
    r_edge = section_edges(fn)

    print()
    print("=" * 96)
    print("总结果")
    print("=" * 96)
    if fn is None:
        print("  ⛔ 未实现对拍（CTC.py 为空或缺少实现函数）—— 只有自检段可用")
        return
    for name, r in (("主对拍", r_main), ("数值稳定性", r_stab), ("边界用例", r_edge)):
        print(f"  {name:<12} {'SKIP' if r is None else ('PASS ✅' if r else 'FAIL ❌')}")
    if all(r for r in (r_main, r_stab, r_edge) if r is not None) and ok_base:
        print("\n  🎉 全部通过 —— 前向算法与 torch.nn.CTCLoss 一致")


if __name__ == "__main__":
    sys.exit(main())
