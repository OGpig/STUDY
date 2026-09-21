#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
把 TensorBoard 的 tfevents 画成训练曲线图（matplotlib）。

【用法】
    python scripts/plot_tensorboard.py --logdir work/aishell/tensorboard/smoke
    python scripts/plot_tensorboard.py --logdir <dir> --out <file.png> --smooth 28
    python scripts/plot_tensorboard.py --logdir <dir> --list        # 只列数值

【画法要点（都是踩过坑才定的）】
1. **按 step 去重**：同一个 logdir 下堆了多次运行的 event 文件时，
   TensorBoard / EventAccumulator 会把它们**合并**，同一 step 出现两次 → 画成锯齿
   （线降到最低又跳回起点重画一遍）。
2. **平滑窗口按「每 epoch 步数」给**（默认 28）。窗口太小（如 10）会被 per-batch 噪声淹没，
   看不出「总体在降」。
3. **per-batch 噪声的来源不是模型**：CTC 的 -logP 随帧数累积（实测 T 翻倍 loss 翻倍），
   且数据管线按长度排序组 batch → 相邻 batch 长短不同 → 损失起伏。
   所以图里同时画「原始(淡) + 平滑(粗)」。
4. 中文用 Windows 自带的 Microsoft YaHei（回退 SimHei / Noto Sans CJK SC / DejaVu Sans）。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

COLORS = ["#185FA5", "#993C1D", "#0F6E56", "#854F0B", "#A32D2D", "#3B6D11"]


def say(*a, **kw) -> None:
    print(*a, file=sys.stderr, **kw)


def read_scalars(logdir: Path) -> tuple[dict[str, list[tuple[int, float]]],
                                        list[str], list[str]]:
    """读 tfevents，并按 step 去重（同一 step 保留最后一次）。"""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    files = sorted(str(p) for p in logdir.rglob("events.out.tfevents.*"))
    ea = EventAccumulator(str(logdir), size_guidance={"scalars": 0})
    ea.Reload()

    out: dict[str, list[tuple[int, float]]] = {}
    dup: list[str] = []
    for t in ea.Tags()["scalars"]:
        raw = [(int(e.step), float(e.value)) for e in ea.Scalars(t)]
        d: dict[int, float] = {}
        for step, val in raw:
            d[step] = val
        if len(d) != len(raw):
            dup.append(f"{t}: {len(raw)} 点 → {len(d)} 个唯一 step")
        out[t] = sorted(d.items())
    return out, files, dup


def smooth(vals: list[float], win: int) -> list[float]:
    if win <= 1 or not vals:
        return list(vals)
    out, acc, n = [], 0.0, 0
    for i, v in enumerate(vals):
        acc += v
        n += 1
        if n > win:
            acc -= vals[i - win]
            n = win
        out.append(acc / n)
    return out


def setup_font() -> str:
    import matplotlib
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"):
        if name in installed:
            matplotlib.rcParams["font.sans-serif"] = [name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    return "default"


def main() -> int:
    ap = argparse.ArgumentParser(description="tfevents → 训练曲线图（matplotlib）")
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default="WeNet Conformer 冒烟训练（AISHELL-1 200 条子集）")
    ap.add_argument("--smooth", type=int, default=28,
                    help="滑动平均窗口，默认 28（= 一个 epoch 的步数）")
    ap.add_argument("--show-batch", action="store_true",
                    help="额外画一张 per-batch 面板（默认不画：噪声主导，没人拿它做判据）")
    ap.add_argument("--list", action="store_true", help="只列数值")
    args = ap.parse_args()

    logdir = Path(args.logdir)
    if not logdir.is_dir():
        say(f"❌ 目录不存在: {logdir}")
        return 1

    data, ev_files, dup = read_scalars(logdir)

    if len(ev_files) > 1:
        say(f"⚠️ logdir 下有 {len(ev_files)} 个 event 文件（多次运行堆在一起）：")
        for f in ev_files:
            say(f"     {Path(f).name}")
        say("   → 已按 step 去重，否则曲线会出现「降到最低又跳回起点」的锯齿。")
        say("   → 规范：--tensorboard_dir 带运行标识，或重跑前清目录。")
    if dup:
        say(f"   去重明细（{len(dup)} 条）: " + "; ".join(dup[:3])
            + (" …" if len(dup) > 3 else ""))

    if args.list:
        say(f"标量 tag 数: {len(data)}")
        for t in sorted(data):
            v = data[t]
            say(f"  {t:24s} {len(v):5d} 点  首={v[0][1]:12.4f}  末={v[-1][1]:12.4f}")
        return 0

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    say(f"字体: {setup_font()}")

    train_tags = [("train/train_loss", "train_loss"),
                  ("train/loss_att", "loss_att"),
                  ("train/loss_ctc", "loss_ctc")]
    cv_tags = [("epoch/loss", "loss"),
               ("epoch/loss_att", "loss_att"),
               ("epoch/loss_ctc", "loss_ctc")]
    nsteps = len(data.get(train_tags[0][0], []))
    n_ep = len(data.get(cv_tags[0][0], [])) or 5

    def train_epoch_means(tag: str) -> list[float]:
        """把 per-batch 序列按 epoch 聚合（边界 = 总步数 / epoch 数）。"""
        if tag not in data:
            return []
        ys = [y for _, y in data[tag]]
        if not ys or n_ep <= 0 or len(ys) % n_ep:
            return []
        p = len(ys) // n_ep
        return [sum(ys[e * p:(e + 1) * p]) / p for e in range(n_ep)]

    def cv_series(tag: str) -> list[float]:
        return [y for _, y in data.get(tag, [])]

    n = 4 + (1 if args.show_batch else 0)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.7 * n + 1.0),
                             constrained_layout=True)
    fig.suptitle(args.title, fontsize=14)
    i = 0

    # ---- （可选）per-batch 面板：默认不画。
    #      per-batch 损失被「batch 里的音频长度」主导（CTC 的 -logP 随帧数累积），
    #      噪声远大于趋势，实务上没人拿它做判据。
    if args.show_batch:
        ax = axes[i]; i += 1
        for tag, label in train_tags:
            if tag not in data:
                continue
            xs = [x for x, _ in data[tag]]
            ys = [y for _, y in data[tag]]
            ax.plot(xs, ys, alpha=0.25, linewidth=0.7)
            sm = smooth(ys, args.smooth)
            ax.plot(xs, sm, linewidth=2.2,
                    label=f"{label} 平滑后 {sm[0]:.0f} → {sm[-1]:.0f}")
        ax.set_title(f"训练损失（每 batch，共 {nsteps} 步；"
                     f"淡=原始，粗={args.smooth} 步滑动平均）")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    # ---- 面板：总损失 train vs 验证（每 epoch）—— 主图
    ax = axes[i]; i += 1
    m = train_epoch_means(train_tags[0][0])
    if m:
        ax.plot(range(1, len(m) + 1), m, "--s", markersize=6, color="#185FA5",
                linewidth=1.8, label=f"train loss  {m[0]:.1f} → {m[-1]:.1f}")
    c = cv_series(cv_tags[0][0])
    if c:
        ax.plot(range(1, len(c) + 1), c, "-o", markersize=6, color="#993C1D",
                linewidth=2, label=f"cv loss  {c[0]:.1f} → {c[-1]:.1f}")
    ax.set_title("总损失：train vs 验证（每 epoch）")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_xticks(range(n_ep))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ---- 面板：分支损失 attention / CTC
    ax = axes[i]; i += 1
    for ttag, cvtag, label, col in (
            ("train/loss_att", "epoch/loss_att", "loss_att", "#0F6E56"),
            ("train/loss_ctc", "epoch/loss_ctc", "loss_ctc", "#854F0B")):
        m = train_epoch_means(ttag)
        if m:
            ax.plot(range(1, len(m) + 1), m, "--s", markersize=5, color=col,
                    linewidth=1.6, label=f"train {label}  {m[0]:.1f} → {m[-1]:.1f}")
        c = cv_series(cvtag)
        if c:
            ax.plot(range(1, len(c) + 1), c, "-o", markersize=5, color=col,
                    linewidth=2.2, label=f"cv {label}  {c[0]:.1f} → {c[-1]:.1f}")
    ax.set_title("分支损失：attention / CTC（每 epoch）")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_xticks(range(n_ep))
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # ---- 面板：帧准确率
    #      ⚠️ x 轴统一用 epoch：train 的 step 要除以「每 epoch 步数」，
    #         否则 train（0~139 步）和 cv（0~4 epoch）根本没法放同一条轴上比。
    ax = axes[i]; i += 1
    if "train/th_accuracy" in data and n_ep and nsteps:
        per = nsteps // n_ep
        xs = [(k + 1) / per for k, _ in data["train/th_accuracy"]]
        ys = smooth([y for _, y in data["train/th_accuracy"]], args.smooth)
        ax.plot(xs, ys, color="#3B6D11", linewidth=2,
                label=f"train th_accuracy（{args.smooth} 步平均，x 已换算成 epoch）")
    c = cv_series("epoch/acc")
    if c:
        ax.plot([e + 1 for e in range(len(c))], c, "-o", markersize=6,
                color="#A32D2D", linewidth=2,
                label=f"cv acc  {c[0]:.4f} → {c[-1]:.4f}")
    ax.set_title("帧准确率（x 轴统一为 epoch）")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_xticks(range(1, n_ep + 1))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ---- 面板：学习率
    ax = axes[i]; i += 1
    if "train/lr_0" in data:
        s = data["train/lr_0"]
        ax.plot([x for x, _ in s], [y for _, y in s], linewidth=2, color="#0F6E56")
    ax.set_title("学习率（warmup 生效的证据）")
    ax.set_xlabel("step")
    ax.set_ylabel("lr")
    ax.grid(alpha=0.3)


    out = Path(args.out) if args.out else logdir / "curves.png"
    fig.savefig(out, dpi=150)
    say(f"✅ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
