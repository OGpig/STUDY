# -*- coding: utf-8 -*-
"""
CTC 前向算法（手写）—— log 域 α 递推，严格对齐 torch.nn.CTCLoss

    手工正确性判据：scripts/test/compare_ctc.py
        对拍 torch.nn.CTCLoss（含边界用例 + T=100 数值稳定性 + 暴力枚举基线自检）

D1 修正版。原实现的 α 递推逻辑经消融验证是**完全正确**的
（把 blank 换成与自己硬编码一致的值后 max|Δ| = 0），
真正的问题只有 1 处致命 + 3 处工程缺陷，下面用 【修正N】 标出。
"""
import torch


def my_ctc_forward(log_probs, targets, input_lengths, target_lengths, blank=0):
    """
    手写 CTC 前向算法，与 torch.nn.CTCLoss 对齐

    log_probs      : (T, N, C) float32 —— **已取对数的概率**（log_softmax 之后）
    targets        : (N, S)    int64   —— 标签 id；blank=0 时取值必须落在 1..C-1
    input_lengths  : (N,)      int64   —— 每个样本的有效帧数（<= T）
    target_lengths : (N,)      int64   —— 每个样本的有效标签数（<= S）
    blank          : int       —— blank 的类别 id，**必须与参照实现保持一致**

    返回: (N,) float32 —— 每个样本的 CTC loss = -log P(target | log_probs)
          对齐 torch.nn.functional.ctc_loss(..., reduction='none', zero_infinity=False)
          帧数不足以容纳标签时返回 +inf
    """
    # ==================== 0. 输入检查 ====================
    assert log_probs.dim() == 3 and targets.dim() == 2, "log_probs 应为 (T,N,C)，targets 应为 (N,S)"
    T, N, C = log_probs.size()
    assert targets.size(0) == N, "targets 的 batch 维与 log_probs 不一致"
    assert input_lengths.size(0) == N and target_lengths.size(0) == N

    # 【修正1】🔴 致命：blank 必须是**参数**，不能硬编码成 C-1。
    #   torch.nn.CTCLoss 的默认是 blank=0（blank 占 0 号类别），
    #   而不少教科书/老实现把 blank 放在最后一类 C-1。两者混用时，
    #   递推会"看起来完全正确、结果却全错"——因为它在解一道不同的题。
    #   原版硬编码 `blank_idx = C-1`，实测与 torch 偏差 max|Δ| ≈ 4.0。
    assert 0 <= blank < C, f"blank={blank} 越界（C={C}）"

    # ==================== 1. 构造扩展序列 l' ====================
    # l' = [blank, l1, blank, l2, ..., lL, blank]，长度 S = 2L+1
    # 奇数位 (2u+1) 放真实标签，偶数位放 blank —— blank 的作用是把相邻重复字符隔开
    S = 2 * int(target_lengths.max().item()) + 1

    ext = torch.full((N, S), blank, dtype=torch.long, device=log_probs.device)
    for n in range(N):
        L = int(target_lengths[n].item())
        if L > 0:
            ext[n, 1:2 * L + 1:2] = targets[n, :L]      # s=1,3,5,... 依次填标签

    # ==================== 2. α 表 ====================
    # α[t, n, s] = 「前 t 帧走完扩展序列前 s 个位置」的**对数总概率**
    #   取值 -inf 表示该状态不可达
    alpha = torch.full((T, N, S), float("-inf"),
                       dtype=log_probs.dtype, device=log_probs.device)

    # --- 初始化 t = 0：第 0 帧只能停在 s=0（发 blank）或 s=1（发第一个标签）---
    alpha[0, :, 0] = log_probs[0, :, blank]

    has_label = target_lengths >= 1                    # L==0 的样本没有 s=1 这个合法状态
    if S > 1:
        alpha[0, has_label, 1] = log_probs[0, has_label, ext[has_label, 1]]

    # ==================== 3. 递推 t = 1 .. T-1 ====================
    # α[t][s] = P(第 t 帧发 ext[s]) × Σ(能一步走到 s 的上一状态 α[t-1][·])
    #   三种合法转移：
    #     ① 原地不动  s → s   （重复发同一符号，或持续发 blank）
    #     ② 前进一格  s-1 → s （发下一个符号）
    #     ③ 跳过一格  s-2 → s （只有当 s 落在真实标签位、且该标签与 s-2 的标签不同时才允许）
    #   ⚠️ ③ 的约束是关键：若 s-2 与 s 是同一个标签（相邻重复字符），
    #      必须经过中间的 blank 才能区分，因此禁止跳跃。
    for t in range(1, T):
        for n in range(N):
            for s in range(S):
                cur = ext[n, s]
                lp = log_probs[t, n, cur]              # 第 t 帧发符号 cur 的对数概率

                if s == 0:
                    # s=0 没有"前一个状态"，只能原地不动
                    alpha[t, n, s] = alpha[t - 1, n, s] + lp
                else:
                    # 转移 ①②：log 域相加用 logaddexp，不能用乘法
                    alpha[t, n, s] = torch.logsumexp(torch.stack([
                        alpha[t - 1, n, s],            # ① 原地不动
                        alpha[t - 1, n, s - 1],        # ② 前进一格
                    ]), dim=0) + lp

                    # 转移 ③：跳过中间那个符号
                    if s > 1 and cur != blank and cur != ext[n, s - 2]:
                        alpha[t, n, s] = torch.logsumexp(torch.stack([
                            alpha[t, n, s],            # 已含 lp
                            alpha[t - 1, n, s - 2] + lp,
                        ]), dim=0)

    # ==================== 4. 终止 ====================
    # 【修正2】🟡 原版把这一步写在 t 循环**内部**，有两个后果：
    #   (a) 同一个值被重复计算 T-1 次（T=100 时白跑 99 遍）
    #   (b) **T==1 时循环体一次都不执行 → `losses` 从未定义 → UnboundLocalError**
    #   正确做法是循环结束后统一算一次。
    #
    # 【修正3】🟡 原版写 `last_index - 1`，当 L==0 时 last_index=0 → 索引 -1，
    #   Python 负索引会**绕回扩展序列最后一列**。当前恰好那列是 -inf 才没暴露错误，
    #   但这是实打实的隐患，必须显式排除。
    losses = torch.empty(N, dtype=log_probs.dtype, device=log_probs.device)

    for n in range(N):
        t_last = int(input_lengths[n].item()) - 1
        L = int(target_lengths[n].item())
        if t_last < 0 or t_last >= T:
            losses[n] = float("inf")                   # 没有有效帧 → 不可达
            continue

        s_end = 2 * L                                  # 末位标签位（与 torch 定义一致）
        acc = alpha[t_last, n, s_end]
        if s_end - 1 >= 0:                             # L==0 时不存在这个状态，跳过
            acc = torch.logaddexp(acc, alpha[t_last, n, s_end - 1])
        losses[n] = -acc                               # loss = -log P

    return losses


if __name__ == "__main__":
    # 最小自测：完整对拍请跑 scripts/test/compare_ctc.py
    torch.manual_seed(0)
    T, N, C = 5, 2, 4
    log_probs = torch.randn(T, N, C).log_softmax(-1)

    # 【修正0】原自测的两个坑：
    #   (a) targets 里含 0，而 blank=0 —— 0 号类别既当 blank 又当标签，语义自相矛盾；
    #       blank=0 时标签必须落在 1..C-1。
    #   (b) 手写返回 (N,) 的逐样本 loss，而 ctc_loss 默认 reduction='mean' 返回标量，
    #       两者不能直接比大小；要加 reduction='none'。
    targets = torch.tensor([[1, 2], [2, 3]], dtype=torch.long)
    input_lengths = torch.tensor([5, 5], dtype=torch.long)
    target_lengths = torch.tensor([2, 2], dtype=torch.long)

    mine = my_ctc_forward(log_probs, targets, input_lengths, target_lengths, blank=0)
    ref = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                                       blank=0, reduction="none", zero_infinity=False)

    print("手写 :", [round(v, 6) for v in mine.tolist()])
    print("官方 :", [round(v, 6) for v in ref.tolist()])
    print(f"max|Δ| = {(mine - ref).abs().max().item():.3e}")
