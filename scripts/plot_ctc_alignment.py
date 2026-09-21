#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
实验①：CTC 对齐可视化 —— 亲眼看到模型「听」到了什么。

做法：取验证集里的一条真实语音，跑 encoder + CTC，得到
    probs (T', V)   # 每一帧在 3591 个符号上的概率
然后画一张热力图：
    x 轴 = 帧（每 10ms 一帧，经 subsampling ÷4）
    y 轴 = blank + 这句话里出现的每个字
    颜色 = 概率

这样一眼就能看出：
  · 模型把哪个字「听」到了哪几帧（对齐）
  · 每帧的 argmax 是什么（去重去 blank 后就是解码结果）
  · 它错在哪：概率分散还是集中

输出：work/aishell/exp/mid/ctc_alignment_<key>.png
"""

import json
import os
import sys
from types import SimpleNamespace

import torch
import yaml

sys.path.insert(0, r"F:\embedded\prepare\examples\wenet-main")
ROOT = r"F:\embedded\prepare\work\aishell"
os.chdir(ROOT)

CKPT = "exp/mid/final.pt"          # E02 的模型（CER 72.7%）
CFG = "exp/mid/train.yaml"
LIST = "data/dev_mid/data.list"
IDX = 0                            # 取第几条

import matplotlib
matplotlib.use("Agg")
# ★ 中文字形：必须在 import pyplot 之前设置全局字体，
#   否则刻度标签回退到 DejaVu Sans → 中文变方框
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    from wenet.utils.init_dataset import init_dataset
    from wenet.utils.init_model import init_model
    from wenet.utils.init_tokenizer import init_tokenizer

    cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
    args = SimpleNamespace(device="cpu", use_amp=False, dtype="fp32", lora_rank=None,
                           init_lora=False, load_lora=None, load_lora_ckpt=None,
                           lora_alpha=None, lora_rank_enc=None, lora_rank_dec=None,
                           lora_alpha_enc=None, lora_alpha_dec=None, task=None)
    model, cfg = init_model(args, cfg)
    # ★ init_model 只做随机初始化，必须单独加载权重
    #   （recognize.py 里也是这两步分开；漏了就会拿到随机模型 → 解码乱码）
    from wenet.utils.checkpoint import load_checkpoint
    load_checkpoint(model, CKPT)
    model.eval()
    say = lambda *a: print(*a, file=sys.stderr, flush=True)

    tok = init_tokenizer(cfg)
    id2char = {v: k for k, v in tok.symbol_table.items()}
    blank = cfg.get("ctc_conf", {}).get("ctc_blank_id", 0)

    # ---- 取一条真实语音（走 WeNet 自己的数据管线，保证 fbank/CMVN 与训练一致）
    ds = init_dataset("asr", "raw", LIST, tok, cfg["dataset_conf"],
                      partition=True, split="dev")
    batch = next(iter(ds))
    keys = batch["keys"]
    idx = min(IDX, len(keys) - 1)
    key = keys[idx]

    ref = {}
    for l in open(LIST, encoding="utf-8"):
        if l.strip():
            d = json.loads(l)
            ref[d["key"]] = d["txt"]
    text_ref = ref.get(key, "")

    # ---- 前向：非流式整句
    feats = batch["feats"]
    flen = batch["feats_lengths"]
    with torch.no_grad():
        enc_out, enc_mask = model.encoder(feats, flen)
        enc_len = enc_mask.squeeze(1).sum(1).long()
        probs = model.ctc.log_softmax(enc_out)      # (B, T', V)
        ids = probs.argmax(-1)                      # (B, T')

    T = int(enc_len[idx].item())
    P = probs[idx, :T].numpy()                      # (T', V)
    ids_b = ids[idx, :T].tolist()

    # ---- 折叠：去重复、去 blank → 解码结果
    dec = []
    prev = -1
    for i in ids_b:
        if i != prev and i != blank:
            dec.append(id2char.get(i, "?"))
        prev = i
    decoded = "".join(dec)

    # ---- 组矩阵：y = blank + 参考句里按出现顺序的唯一字
    chars = []
    for ch in text_ref:
        if ch not in chars and ch in tok.symbol_table:
            chars.append(ch)
    rows = ["<blank>"] + chars
    row_ids = [blank] + [tok.symbol_table[c] for c in chars]

    M = np.zeros((len(rows), T))
    for r, rid in enumerate(row_ids):
        M[r] = P[:, rid]

    # ★ 全词表概率分散在 3591 类上，单符号概率都很低（~0.2），
    #   直接用 0~1 色标会整张图全黑。改成「每帧在所选符号间归一化」：
    #   颜色只表达「这一帧里这些符号谁占优」，对齐关系一目了然。
    Mn = M / (M.sum(axis=0, keepdims=True) + 1e-9)

    # ---- argmax 时间线（供对比：每帧实际输出什么）
    argmax_chars = ["_" if i == blank else id2char.get(i, "?") for i in ids_b]

    # ---- 画图
    fig, (ax2, ax1, ax3) = plt.subplots(
        3, 1, figsize=(14, 3.2 + 0.28 * len(rows) + 1.6),
        gridspec_kw={"height_ratios": [1.0, len(rows), 1.4]},
        constrained_layout=True)

    fig.suptitle(f"实验① CTC 对齐可视化   key={key}", fontsize=13)

    # 上：参考文本
    ax2.axis("off")
    ax2.text(0.0, 0.5, f"标注：{text_ref}", fontsize=13, va="center",
             fontname="Microsoft YaHei")

    # 中：热力图（行内归一化）
    im = ax1.imshow(Mn, aspect="auto", cmap="magma", interpolation="nearest",
                    vmin=0.0, vmax=Mn.max())
    ax1.set_yticks(range(len(rows)))
    ax1.set_yticklabels(rows, fontsize=10)
    ax1.set_xticks(range(0, T, 5))
    ax1.set_xticklabels([str(t) for t in range(0, T, 5)], fontsize=8)
    ax1.set_ylabel("符号（blank + 参考字）")
    ax1.set_xlabel(f"帧（encoder 输出，共 {T} 帧 ≈ {T*40/1000:.1f} s）")
    fig.colorbar(im, ax=ax1, pad=0.01, label="相对概率（每帧归一化）")

    # ★ 标出每个参考字的「峰值帧」—— 这就是模型认为该字出现的位置
    for r in range(1, len(rows)):            # 跳过 blank 行
        t_peak = int(np.argmax(M[r]))
        ax1.plot(t_peak, r, marker="*", markersize=13,
                 color="#00E5FF", markeredgecolor="black", markeredgewidth=0.6,
                 linestyle="none")

    # 下：每帧 argmax 的符号时间线
    ax3.axis("off")
    for i, ch in enumerate(argmax_chars):
        ax3.text((i + 0.5) / T, 0.62, ch, ha="center", va="center",
                 fontsize=11 if ch != "_" else 9,
                 fontname="Microsoft YaHei",
                 color="#A32D2D" if ch == "_" else "#085041")
    ax3.text(0.0, 0.15, "每帧 argmax（折叠后 = 解码结果）：", fontsize=11,
             va="center", fontname="Microsoft YaHei")
    ax3.text(0.0, -0.05, f"识别：{decoded if decoded else '(空串)'}",
             fontsize=12, va="center", fontname="Microsoft YaHei",
             color="#A32D2D" if not decoded else "#085041")
    ax3.set_xlim(0, 1)
    ax3.set_ylim(-0.1, 1.0)

    out = f"exp/mid/ctc_alignment_{key}.png"
    fig.savefig(out, dpi=140)
    print(f"✅ {out}", file=sys.stderr)
    print(f"   标注: {text_ref}", file=sys.stderr)
    print(f"   识别: {decoded if decoded else '(空串)'}", file=sys.stderr)
    print(f"   帧数 T' = {T}", file=sys.stderr)


if __name__ == "__main__":
    main()
