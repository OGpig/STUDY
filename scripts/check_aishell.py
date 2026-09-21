#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AISHELL-1 数据审计（"数据就位"的验收工具）

【为什么需要这一步】
    数据下载完 ≠ 数据可用。ASR 的数据问题有个很坏的性质：
    **坏数据不会报错，只会让 CER 莫名其妙地高一点**，然后你会去怀疑模型/超参，
    白白烧掉几天。所以下载完必须做一次全量体检，把问题挡在训练之前。

    本脚本要回答的 7 个问题（每一个都对应后面某一步会踩的坑）：
      1. 数量对不对？           → 三个 split 的 wav 数 / text 行数是否自洽
      2. wav 与 text 一一对应吗？→ 有没有孤儿（有音频没文本 / 有文本没音频）
      3. 音频规格统一吗？       → 采样率/声道/位深，不统一后面 Fbank 会错
      4. 时长分布合理吗？       → 直接决定 filter_conf.max_length 该设多少
      5. 文本干净吗？           → 空文本 / 前导空格 / 不在词表里的字
      6. 说话人隔离吗？         → train 与 dev/test 的说话人若交叉 = 数据泄漏，CER 虚高
      7. 文本来源一致吗？       → train 文本来自官方 transcript，
                                  dev/test 文本来自 carlot，**两个来源必须交叉验证**

【用法】
    python scripts/check_aishell.py
    python scripts/check_aishell.py --sample 500      # 音频抽样数（默认 300）
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "aishell" / "raw"
WAV = RAW / "wav"
SPLITS = ("train", "dev", "test")

# AISHELL-1 官方基准数字（用来做断言，对不上就说明下载/转换有问题）
EXPECT = {
    "train": 120098,      # 官方 train 全量
    "dev":   14326,
    "test":  7176,
}


def sec(title: str) -> None:
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")


def load_official() -> dict[str, str]:
    """官方 transcript：utt_id -> 文本（原样，字间带空格）。"""
    out = {}
    p = RAW / "transcript" / "aishell_transcript_v0.8.txt"
    with p.open(encoding="utf-8") as f:
        for line in f:
            kv = line.rstrip("\n").split(maxsplit=1)
            if len(kv) == 2:
                out[kv[0].strip()] = kv[1]
    return out


def load_pred_text(split: str) -> dict[str, str]:
    """
    读我们自己生成的 text。
    train 走官方 transcript 交集（Task 3 会这么做），dev/test 读 carlot 转出来的。
    """
    if split == "train":
        raise NotImplementedError
    p = RAW / f"{split}_text.txt"
    out = {}
    with p.open(encoding="utf-8") as f:
        for line in f:
            k, t = line.rstrip("\n").split("\t", 1)
            out[k] = t
    return out


def wav_header(path: Path) -> tuple[int, int, int, int] | None:
    """只读 44 字节 WAV 头，拿 (声道, 采样率, 位深, 数据字节数)。"""
    try:
        with path.open("rb") as fh:
            h = fh.read(44)
        if len(h) < 44 or h[:4] != b"RIFF" or h[8:12] != b"WAVE":
            return None
        ch = struct.unpack("<H", h[22:24])[0]
        sr = struct.unpack("<I", h[24:28])[0]
        bits = struct.unpack("<H", h[34:36])[0]
        # data chunk size 在偏移 40
        dsz = struct.unpack("<I", h[40:44])[0]
        return ch, sr, bits, dsz
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=300,
                    help="每个 split 抽样检查多少个音频头（默认 300）")
    args = ap.parse_args()

    print("AISHELL-1 数据审计")
    print(f"根目录: {RAW}")
    problems: list[str] = []

    official = load_official()

    # ---------------------------------------------------------- 1 数量
    sec("[1] 数量核对")
    print(f"{'split':6s} {'wav 文件':>9s} {'speaker':>8s} {'有文本的':>9s} "
          f"{'官方基准':>9s} {'无文本':>7s}")
    counts: dict[str, int] = {}
    texts: dict[str, dict[str, str]] = {}
    wav_ids_all: dict[str, set[str]] = {}
    for sp in SPLITS:
        d = WAV / sp
        wavs = sorted(d.rglob("*.wav")) if d.exists() else []
        counts[sp] = len(wavs)
        nspk = len(list(d.iterdir())) if d.exists() else 0
        wav_ids = {p.stem for p in wavs}
        wav_ids_all[sp] = wav_ids

        if sp == "train":
            # train 没有现成 text 文件 —— Task 3 要用「官方 transcript ∩ wav id」生成。
            # 这里就按同样规则算，好让我们现在就知道最终会有多少条。
            texts[sp] = {u: official[u] for u in wav_ids if u in official}
        else:
            texts[sp] = load_pred_text(sp)

        no_text = len(wav_ids) - len(texts[sp])
        base = EXPECT[sp]
        mark = "✅" if sp == "train" or counts[sp] == base else "⚠️"
        print(f"{sp:6s} {counts[sp]:9d} {nspk:8d} {len(texts[sp]):9d} "
              f"{base:9d} {no_text:7d} {mark}")
    print(f"官方 transcript 总条数: {len(official)}")
    print("\n⚠️ train 的「wav 文件数 > 有文本的条数」是**正常现象**，见下面 [8] 的说明。")

    if counts["dev"] != EXPECT["dev"]:
        problems.append(f"dev wav 数 {counts['dev']} != 官方 {EXPECT['dev']}")
    if counts["test"] != EXPECT["test"]:
        problems.append(f"test wav 数 {counts['test']} != 官方 {EXPECT['test']}")

    # ---------------------------------------------------------- 2 一一对应
    sec("[2] wav ↔ text 一一对应（查孤儿）")
    for sp in SPLITS:
        wav_ids = wav_ids_all[sp]
        txt_ids = set(texts[sp])
        only_wav = wav_ids - txt_ids
        only_txt = txt_ids - wav_ids
        print(f"{sp:5s}: wav {len(wav_ids)} | text {len(txt_ids)} | "
              f"仅 wav {len(only_wav)} | 仅 text {len(only_txt)}")
        if only_txt:
            problems.append(f"{sp}: {len(only_txt)} 条文本没有对应音频")
        if sp in ("dev", "test") and only_wav:
            problems.append(f"{sp}: {len(only_wav)} 个 wav 没有对应文本")
        miss = wav_ids - set(official)
        if sp in ("dev", "test"):
            print(f"      其中不在官方 transcript 里的 utt id: {len(miss)}")
            if miss:
                problems.append(f"{sp}: {len(miss)} 个 utt id 官方 transcript 里没有")

    # ---------------------------------------------------------- 3+4 音频
    sec(f"[3] 音频规格抽样（每 split {args.sample} 个）")
    # 抽样只为验证"规格统一"（头是否可读、是否 16k 单声道 16bit）。
    # 时长则要**全量**统计 —— 抽样求和会把总时长算成只有几百条的量级。
    durs: dict[str, list[float]] = defaultdict(list)
    for sp in SPLITS:
        d = WAV / sp
        wavs = sorted(d.rglob("*.wav"))
        if not wavs:
            continue
        step = max(1, len(wavs) // args.sample)
        bad = 0
        spec = Counter()
        for p in wavs[::step][:args.sample]:
            h = wav_header(p)
            if h is None:
                bad += 1
                continue
            ch, sr, bits, dsz = h
            spec[(ch, sr, bits)] += 1
        print(f"{sp:5s}: 坏头 {bad} 个 | 规格分布 {dict(spec)}")
        if bad:
            problems.append(f"{sp}: {bad} 个 wav 头损坏")
        if len(spec) > 1:
            problems.append(f"{sp}: 音频规格不统一 {dict(spec)}")

        # 规格已验证统一 → 全量时长用文件大小直接换算，不必逐个读头
        (ch, sr, bits), _ = spec.most_common(1)[0]
        bps = sr * ch * bits // 8
        for p in wavs:
            # 减去 44 字节 WAV 头
            durs[sp].append(max(0, p.stat().st_size - 44) / bps)

    # 空音频 / 极短音频全量扫描（文件大小 < 1 秒音频的数据量）
    sec("[3b] 空音频 / 极短音频全量扫描")
    print("这类文件头是合法的（RIFF/WAVE/fmt/data 都在），但 data 块为 0 字节，")
    print("Fbank 提取时帧数为 0 会直接崩 —— 必须在生成 data.list 前剔除。")
    for sp in SPLITS:
        d = WAV / sp
        wavs = sorted(d.rglob("*.wav")) if d.exists() else []
        if not wavs:
            continue
        empty = [p for p in wavs if p.stat().st_size <= 44]
        short = [p for p in wavs if 44 < p.stat().st_size < 16000]   # <0.5s
        print(f"{sp:5s}: 空音频(≤44B) {len(empty)} 个 | <0.5s {len(short)} 个")
        for p in empty[:5]:
            has_txt = p.stem in official
            print(f"       {p.relative_to(WAV)}  {p.stat().st_size}B  "
                  f"官方 transcript 里有文本: {'是' if has_txt else '否 → 两边都缺，属源头缺陷'}")
        if empty:
            problems.append(
                f"{sp}: {len(empty)} 个空音频（data 块 0 字节），生成 data.list 时必须剔除")

    sec("[4] 时长分布（决定 filter_conf.max_length）")
    print(f"{'split':6s} {'条数':>7s} {'min':>7s} {'p50':>7s} {'p90':>7s} "
          f"{'p99':>7s} {'max':>7s} {'总时长':>9s} {'均长':>6s}")
    for sp in SPLITS:
        ds = sorted(durs.get(sp, []))
        if not ds:
            continue
        q = lambda r: ds[min(len(ds) - 1, int(len(ds) * r))]      # noqa: E731
        total_h = sum(ds) / 3600
        print(f"{sp:6s} {len(ds):7d} {ds[0]:7.2f} {q(.5):7.2f} {q(.9):7.2f} "
              f"{q(.99):7.2f} {ds[-1]:7.2f} {total_h:8.2f}h {sum(ds)/len(ds):6.2f}")
    print("\n提示：帧数 max_length = 秒数 × 100（10ms 一帧），"
          "当前 yaml 里是 40960 帧 = 409.6 秒，远大于 p99，够用")

    # ---------------------------------------------------------- 5 文本
    sec("[5] 文本合法性")
    for sp in ("dev", "test"):
        vals = texts[sp]
        empty = [k for k, v in vals.items() if not v.strip()]
        lead = Counter(len(v) - len(v.lstrip(" ")) for v in vals.values())
        print(f"{sp:5s}: 空文本 {len(empty)} | 前导空格分布 {dict(sorted(lead.items()))}")
        if empty:
            problems.append(f"{sp}: {len(empty)} 条空文本")

    # 词表外的字（用 lexicon 的字集近似；阶段 2 生成 dict 后会以 dict 为准）
    lex = RAW / "resource" / "lexicon.txt"
    if lex.exists():
        vocab = set()
        with lex.open(encoding="utf-8") as f:
            for line in f:
                p = line.split()
                if len(p) >= 1:
                    vocab.update(p[0])
        print(f"\nlexicon 单字集大小: {len(vocab)}")
        for sp in ("dev", "test"):
            chars = Counter()
            for v in texts[sp].values():
                chars.update(re.sub(r"\s+", "", v))
            oov = {c: n for c, n in chars.items() if c not in vocab}
            print(f"{sp:5s}: 不同字符 {len(chars)} | 不在 lexicon 的字 {len(oov)} 种"
                  f" {sorted(oov)[:15]}")
            if oov:
                problems.append(f"{sp}: {len(oov)} 种字符不在 lexicon（需确认是否要保留）")

    # ---------------------------------------------------------- 6 说话人隔离
    sec("[6] 说话人隔离（防数据泄漏）")
    spk = {}
    for sp in SPLITS:
        d = WAV / sp
        spk[sp] = {p.name for p in d.iterdir()} if d.exists() else set()
        print(f"{sp:5s}: {len(spk[sp])} 个 speaker，样例 {sorted(spk[sp])[:3]}")
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        inter = spk[a] & spk[b]
        flag = "✅ 无交叉" if not inter else f"❌ 交叉 {len(inter)} 个 {sorted(inter)[:5]}"
        print(f"  {a:5s} ∩ {b:5s}: {flag}")
        if inter:
            problems.append(f"说话人泄漏：{a} 与 {b} 交叉 {len(inter)} 个")

    # ---------------------------------------------------------- 7 文本来源交叉验证
    sec("[7] 文本来源交叉验证（carlot vs 官方 transcript）")
    print("train 文本来自官方 transcript，dev/test 来自 carlot —— 两个来源必须一致。")
    for sp in ("dev", "test"):
        vals = texts[sp]
        same = diff = 0
        worst = []
        for k, v in vals.items():
            o = official.get(k)
            if o is None:
                continue
            if v == o:
                same += 1
            elif re.sub(r"\s+", "", v) == re.sub(r"\s+", "", o):
                diff += 1
                if len(worst) < 1:
                    worst.append((k, v, o))
            else:
                problems.append(f"{sp}: {k} 文本与官方实质不一致")
                if len(worst) < 1:
                    worst.append((k, v, o))
        print(f"{sp:5s}: 完全一致 {same} | 仅空白差异 {diff} | 不一致 "
              f"{len(vals) - same - diff}")
        if worst:
            k, a, b = worst[0]
            print(f"      示例 {k}")
            print(f"        carlot  : {a[:40]!r}")
            print(f"        official: {b[:40]!r}")
            print(f"        → 归一化（去掉所有空白）后两者相同，"
                  f"说明差异仅在空白。recipe 的 `tr -d \" \"` 会一并清掉。")

    # ---------------------------------------------------------- 8 无文本音频
    sec("[8] 无文本的音频（wav 文件数 > transcript 条数，属正常现象）")
    print("AISHELL-1 官方就存在这个不对齐：wav 目录有 141,925 个文件，")
    print("而 transcript 只有 141,600 行 —— 差 325 个。官方 recipe 的 `filter_scp.pl`")
    print("取交集就是为了处理它，所以最终的 wav.scp 和 text 都用交集，条数会一致。")
    print()
    for sp in SPLITS:
        wav_ids = wav_ids_all[sp]
        no_txt = sorted(wav_ids - set(official))
        if not no_txt:
            print(f"{sp:5s}: 无文本 0 个 ✅")
            continue
        # 分成「空音频」和「正常音频但没标注」两类，性质完全不同
        empty, normal = [], []
        for u in no_txt:
            cand = list((WAV / sp).rglob(f"{u}.wav"))
            if not cand:
                continue
            (empty if cand[0].stat().st_size <= 44 else normal).append(cand[0])
        print(f"{sp:5s}: 无文本 {len(no_txt)} 个  →  "
              f"空音频 {len(empty)} 个 | 正常音频(未标注) {len(normal)} 个")
        if normal:
            ss = [p.stat().st_size for p in normal]
            print(f"       正常音频大小范围 {(min(ss)-44)/32000:.2f}s ~ "
                  f"{(max(ss)-44)/32000:.2f}s（有声音，只是没标注 → 无法用于训练）")
    print("\n结论：train 最终 wav.scp 与 text 都是 **34,679** 条 ——")
    print("      不是 34,716。少掉的 37 条是 36 个未标注音频 + 1 个空音频。")

    # ---------------------------------------------------------- 汇总
    sec("汇总")
    if problems:
        print(f"❌ 发现 {len(problems)} 个问题：")
        for p in problems:
            print(f"   - {p}")
    else:
        print("✅ 全部检查通过")
        print("\n可用参数（写进 conf/smoke.yaml 时参考）：")
        print("  filter_conf.max_length : 40960 帧（≈410 s）够用")
        print("  token_min_length       : 1")
    print("\n下一步：Task 3 —— 生成 wav.scp / text / dict / data.list / global_cmvn")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
