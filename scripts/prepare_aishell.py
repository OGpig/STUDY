#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AISHELL-1 数据准备：把 WeNet 官方 shell recipe（stage 0~3）移植成 Python。

【为什么不能直接用官方 recipe】
    `examples/wenet-main/examples/aishell/s0/run.sh` 是 bash 脚本，依赖：
      · `. ./path.sh`（source 一个 shell 文件）
      · `local/*.sh`（子 shell）
      · `tools` 与 `wenet` 两个**符号链接**
    而这台机器上：tar.exe 把符号链接解压成了普通文本文件（内容就是路径字符串），
    Windows 也没有可用的 bash 环境来跑 `source`。所以整条 recipe 流程不可用，
    只能把它的四个 stage 翻译成 Python。

【翻译自哪些文件】
    stage 0  wav.scp / text          ← local/aishell_data_prep.sh
    stage 1  文本去空格 + global_cmvn ← run.sh 第 88~98 行
    stage 2  dict/lang_char.txt       ← run.sh 第 101~110 行
    stage 3  data.list                ← run.sh 第 112~124 行

【与官方 recipe 的三处有意不同（都是审计查出来的实际数据问题）】
    1. 文本统一用**官方 transcript** 作为唯一来源。
       官方 recipe 里 train 的文本来自 transcript、dev/test 也来自 transcript；
       但我们的 dev/test 音频是从 carlot parquet 转出来的，parquet 里也带 transcription。
       两套文本必须交叉验证（实测 100% 一致，仅空白差异），
       然后统一用官方 transcript 生成 —— 少一个来源就少一类 bug。
    2. 归一化用「删掉所有空白」而不是只 strip 首尾。
       实测 carlot 的 dev/test 文本带**统一 4 个前导空格**、官方 transcript 带尾部空格。
       若只 strip，4 个空格会变成 4 个空字符混进标签序列。
    3. 显式剔除空音频。
       train 里有 1 个 44 字节的 wav（WAV 头合法但 data 块为 0），
       它会让 Fbank 提出 0 帧、**在训练中途**才崩。取交集能排除它，但我们要显式记一笔。

【输出布局】（刻意对齐 WeNet 自己的 recipe 布局，这样 yaml 里的相对路径直接可用）
    work/aishell/
    ├── conf/                      ← 训练配置（下一步生成）
    └── data/
        ├── train/  {wav.scp, text, data.list, global_cmvn}
        ├── dev/    {wav.scp, text, data.list}
        ├── test/   {wav.scp, text, data.list}
        ├── train_smoke/ {wav.scp, text, data.list}   ← 冒烟测试用的小子集
        ├── dev_smoke/   {wav.scp, text, data.list}
        └── dict/lang_char.txt

    ⚠️ 为什么 yaml 里的路径能直接用：WeNet 的 conf 写的是 `data/train/global_cmvn`
       这类**相对路径**，相对于 **CWD**。所以训练时 CWD 必须是 `work/aishell/`。

【用法】
    # 全部做完（推荐）
    python scripts/prepare_aishell.py --stage all

    # 分步
    python scripts/prepare_aishell.py --stage env      # 先探环境（torchaudio 能不能读音频）
    python scripts/prepare_aishell.py --stage scp
    python scripts/prepare_aishell.py --stage dict
    python scripts/prepare_aishell.py --stage cmvn
    python scripts/prepare_aishell.py --stage list
    python scripts/prepare_aishell.py --stage smoke
    python scripts/prepare_aishell.py --stage verify

【可重复运行】各阶段都是幂等的；cmvn 若已生成会跳过（除非加 --force）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------- 路径

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "aishell" / "raw"          # 下载来的原始数据
WORK = ROOT / "work" / "aishell"                 # 训练工作目录（CWD）
PREP = WORK / "data"                             # 准备好的数据（对齐 recipe 布局）

SPLITS = ("train", "dev", "test")
# 官方基准（用来断言；数字来自 data/aishell/raw 的审计结论）
EXPECT_TRAIN_UTTS = 34679        # 不是 34716：37 个 wav 没有标注（36 未标注 + 1 空文件）

SMOKE_TRAIN = 200                # 冒烟训练用的条数
SMOKE_DEV = 50

FEAT_DIM = 80                    # fbank 维数（与 conf 里 fbank_conf.num_mel_bins 一致）
SAMPLE_RATE = 16000


def say(*a, **kw) -> None:
    """脚本信息一律走 stderr，不污染 stdout。"""
    print(*a, file=sys.stderr, **kw)


def sec(t: str) -> None:
    say(f"\n{'=' * 64}\n{t}\n{'=' * 64}")


# ---------------------------------------------------------------- 基础工具

def wav_size(p: Path) -> int:
    return p.stat().st_size


def is_empty_wav(p: Path) -> bool:
    """44 字节 = 只有 WAV 头、data 块为 0 的空音频。"""
    return wav_size(p) <= 44


def norm_text(t: str) -> str:
    """
    文本归一化：删掉**所有**空白（不只是首尾）。

    ⚠️ 这是本脚本最关键的一行。实测 carlot 的 dev/test 文本带统一 4 个前导空格，
    官方 transcript 带尾部空格；只 strip() 首尾会把中间/首部的空格留成空标签。
    """
    return re.sub(r"\s+", "", t)


def load_official_transcript() -> dict[str, str]:
    """官方 transcript：utt_id -> 原始文本（保留空白，后续统一归一化）。"""
    p = RAW / "transcript" / "aishell_transcript_v0.8.txt"
    out: dict[str, str] = {}
    with p.open(encoding="utf-8") as f:
        for line in f:
            kv = line.rstrip("\n").split(maxsplit=1)
            if len(kv) == 2:
                out[kv[0].strip()] = kv[1]
    return out


def load_carlot_text(split: str) -> dict[str, str]:
    """carlot 转出来的文本（仅 dev/test 有），用于交叉验证。"""
    p = RAW / f"{split}_text.txt"
    if not p.exists():
        return {}
    out = {}
    with p.open(encoding="utf-8") as f:
        for line in f:
            k, _, t = line.rstrip("\n").partition("\t")
            out[k] = t
    return out


def list_wavs(split: str) -> list[Path]:
    d = RAW / "wav" / split
    return sorted(d.rglob("*.wav")) if d.exists() else []


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for ln in lines:
            f.write(ln + "\n")


# ---------------------------------------------------------------- stage: env

# 训练链路依赖的 FFmpeg 共享库（torchcodec 用）。装法与理由见 docs/ENV.md。
FFMPEG_CANDIDATES = [
    Path(r"F:\data\toolchains\ffmpeg-shared\ffmpeg-master-latest-win64-gpl-shared\bin"),
    Path(r"F:\data\toolchains\ffmpeg-shared\bin"),
]


def find_ffmpeg_bin() -> Path | None:
    for p in FFMPEG_CANDIDATES:
        if p.is_dir() and any(p.glob("avcodec-*.dll")):
            return p
    return None


def ensure_ffmpeg_on_path() -> Path | None:
    """
    把 FFmpeg 共享库目录塞进本进程的 PATH。

    为什么要在脚本里做而不是让用户 export：
      训练链路（torchaudio.load → torchcodec → FFmpeg）离不开它，
      而"每次开新终端都要记得设 PATH"是必然会忘的事。
      在这里做掉，调用方就只需要一条 python 命令。
    必须在 import torchaudio 之前调用（torchcodec 在首次解码时才 dlopen，
    所以只要在真正读音频前设好就行，但越早越安全）。
    """
    ff = find_ffmpeg_bin()
    if ff is None:
        return None
    cur = os.environ.get("PATH", "")
    if str(ff).lower() not in cur.lower():
        os.environ["PATH"] = str(ff) + os.pathsep + cur
    return ff


def stage_env() -> None:
    """
    检查运行训练所需的关键能力（torchaudio 读音频）。

    ★ 这是本任务最容易卡住的地方：
      torchaudio ≥ 2.9 把音频解码改成依赖 torchcodec，
      而 torchcodec 自己没有捆绑 FFmpeg 的**共享库**（avcodec/avformat…），
      缺了它 `torchaudio.load()` 会报 "Could not load libtorchcodec"，
      而 WeNet 的 `decode_wav` 正是用它读 wav —— **训练会直接起不来**。
      修法：把 FFmpeg shared 构建的 bin 目录加到 PATH。
    """
    sec("[env] 环境自检")

    ff = find_ffmpeg_bin()
    if ff is None:
        say("❌ 没找到 FFmpeg 共享库（avcodec-*.dll）")
        say("   见 docs/ENV.md 的「FFmpeg」一节")
    else:
        on_path = any(str(ff).lower() == p.lower()
                      for p in os.environ.get("PATH", "").split(os.pathsep))
        say(f"FFmpeg 共享库: {ff}")
        say(f"  已在本进程 PATH 中: {'是 ✅' if on_path else '否 ← 需要手动加'}")

    try:
        import torchaudio
    except ImportError as e:
        say(f"❌ torchaudio 未安装: {e}")
        return
    say(f"torchaudio {torchaudio.__version__}")
    say(f"有 torchaudio.info 吗: {hasattr(torchaudio, 'info')}"
        f"（2.11 已移除；只有 segments 分支用得到，我们不用）")

    probe = next(iter(list_wavs("train")), None)
    if probe is None:
        say("❌ 找不到任何 train wav，先跑下载")
        return

    try:
        w, sr = torchaudio.load(str(probe))
        say(f"  ✅ torchaudio.load -> {tuple(w.shape)} @ {sr}Hz")
    except Exception as e:                                        # noqa: BLE001
        say(f"  ❌ torchaudio.load: {type(e).__name__}: {str(e)[:130]}")
        say("\n⚠️ 音频读取不可用 —— 这会**直接卡住训练**。修法（二选一）：")
        if ff is not None:
            say(f'   PowerShell:  $env:PATH = "{ff};$env:PATH"')
        say("   然后再跑本脚本；或直接用 scripts/run_prepare.ps1（它已设好 PATH）")
        return

    if ff is not None:
        say("\n✅ 音频读取正常，训练链路具备条件")


# ---------------------------------------------------------------- stage: scp

def stage_scp() -> None:
    sec("[scp] 生成 wav.scp + text")
    official = load_official_transcript()
    say(f"官方 transcript 条数: {len(official)}")

    for split in SPLITS:
        wavs = list_wavs(split)
        if not wavs:
            say(f"[{split}] ⚠️ 没有 wav，跳过")
            continue

        empty = [p for p in wavs if is_empty_wav(p)]
        usable = [p for p in wavs if not is_empty_wav(p)]

        # 只在「官方 transcript 里有标注」的音频上做 —— 这一步同时排除了未标注样本
        rows = [(p.stem, p) for p in usable if p.stem in official]
        dropped_no_text = len(usable) - len(rows)

        # ── 交叉验证：dev/test 的 carlot 文本必须与官方 transcript 一致（仅空白差异）
        if split in ("dev", "test"):
            carlot = load_carlot_text(split)
            if carlot:
                bad = []
                for utt, _ in rows:
                    a, b = carlot.get(utt), official.get(utt)
                    if a is None or b is None:
                        continue
                    if norm_text(a) != norm_text(b):
                        bad.append(utt)
                flag = "✅ 一致" if not bad else f"❌ {len(bad)} 条不一致 {bad[:3]}"
                say(f"[{split}] 交叉验证 carlot vs 官方 transcript: {flag}（比对 {len(carlot)} 条）")
                if bad:
                    raise SystemExit(
                        f"[{split}] 文本来源不一致，停止。先查清是归一化差异还是真的标注不同。")

        d = PREP / split
        write_lines(d / "wav.scp",
                    [f"{utt} {'/'.join(str(p.resolve()).split(chr(92)))}" for utt, p in rows])
        write_lines(d / "text",
                    [f"{utt} {norm_text(official[utt])}" for utt, _ in rows])

        say(f"[{split}] wav 文件 {len(wavs)} → 空音频 {len(empty)} → "
            f"无标注 {dropped_no_text} → **最终 {len(rows)} 条**")
        if empty:
            for p in empty:
                say(f"        剔除空音频: {p.relative_to(RAW)}  {wav_size(p)}B")

    say("\n[scp] wav.scp 与 text 行数必须相同（取交集后天然一致）")


# ---------------------------------------------------------------- stage: dict

def stage_dict() -> None:
    sec("[dict] 生成 lang_char.txt")
    text = PREP / "train" / "text"
    if not text.exists():
        say("❌ 先跑 --stage scp")
        return

    chars: set[str] = set()
    with text.open(encoding="utf-8") as f:
        for line in f:
            body = line.rstrip("\n").split(" ", 1)[1] if " " in line else ""
            chars.update(body)
    chars.discard("")

    # 与 run.sh 第 104~109 行一致：0/1/2 是特殊符号，其余按码点序从 3 开始编号
    lines = ["<blank> 0", "<unk> 1", "<sos/eos> 2"]
    for i, c in enumerate(sorted(chars), start=3):
        lines.append(f"{c} {i}")

    out = PREP / "dict" / "lang_char.txt"
    write_lines(out, lines)

    say(f"train 文本去重后字符数: {len(chars)}")
    say(f"词表总大小: {len(lines)} 行（含 3 个特殊符号）→ {out}")
    say(f"样例: {lines[:3]} … {lines[-2:]}")
    if len(chars) < 1000:
        say("⚠️ 字符数偏少（中文常用字应有数千）—— 检查 text 是不是空的")


# ---------------------------------------------------------------- stage: cmvn

def _cmvn_worker(args: tuple[str, str]) -> tuple[int, list, list] | None:
    """
    单个 wav 的 CMVN 统计（在每个子进程里跑）。

    ⚠️ 为什么不用 WeNet 自带的 tools/compute_cmvn_stats.py：
       它用 `torchaudio.info()` + `torchaudio.load()`，
       在 torchaudio 2.11 上前者已被删除、后者需要 torchcodec。
       这里改用 soundfile 读音频（纯 WAV 原生支持、无外部依赖），
       但 **Fbank 仍用 torchaudio.compliance.kaldi.fbank** ——
       与 WeNet 训练时用的完全是同一个函数，统计结果可比。
    """
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio.compliance.kaldi as kaldi

    utt, path = args
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data[:, 0]).unsqueeze(0)
        waveform = waveform * (1 << 15)          # 与 WeNet 一致：整型幅度还原
        mat = kaldi.fbank(waveform,
                          num_mel_bins=80,
                          dither=0.0,                # 统计时关掉抖动
                          energy_floor=0.0,
                          sample_frequency=sr)
        return int(mat.shape[0]), mat.sum(0).tolist(), (mat ** 2).sum(0).tolist()
    except Exception:                            # noqa: BLE001
        return None


def stage_cmvn(force: bool = False, workers: int = 0) -> None:
    sec("[cmvn] 计算 global_cmvn（train 上统计）")
    scp = PREP / "train" / "wav.scp"
    out = PREP / "train" / "global_cmvn"
    if not scp.exists():
        say("❌ 先跑 --stage scp")
        return
    if out.exists() and not force:
        say(f"已存在，跳过（要重算加 --force）：{out}")
        return

    tasks = []
    with scp.open(encoding="utf-8") as f:
        for line in f:
            utt, _, p = line.rstrip("\n").partition(" ")
            tasks.append((utt, p))
    say(f"待处理 {len(tasks)} 条 wav")

    nproc = workers or 12
    say(f"并行进程数 {nproc}（Windows 用 spawn，每个子进程会 import torch，首次约 5~10s）")

    t0 = time.time()
    total_frames = 0
    mean_stat = [0.0] * FEAT_DIM
    var_stat = [0.0] * FEAT_DIM
    done = failed = 0

    with ProcessPoolExecutor(max_workers=nproc) as pool:
        futs = [pool.submit(_cmvn_worker, t) for t in tasks]
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            if r is None:
                failed += 1
                continue
            n, m, v = r
            total_frames += n
            for i in range(FEAT_DIM):
                mean_stat[i] += m[i]
                var_stat[i] += v[i]
            if done % 2000 == 0 or done == len(tasks):
                el = time.time() - t0
                say(f"\r  已处理 {done}/{len(tasks)}  "
                    f"帧数 {total_frames}  {el:.0f}s  ({done/el:.0f} 条/秒)", end="")

    say("")
    info = {"mean_stat": mean_stat, "var_stat": var_stat, "frame_num": total_frames}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fo:
        fo.write(json.dumps(info))

    say(f"✅ {out}")
    say(f"   frame_num = {total_frames}（总帧数，约 {total_frames/100/3600:.2f} 小时）")
    say(f"   elapsed {time.time()-t0:.1f}s, 失败 {failed} 条")
    say(f"   载入时换算方式（wenet/utils/cmvn.py::_load_json_cmvn）："
        f"mean = mean_stat/N, istd = 1/sqrt(var_stat/N − mean²)")
    # 快速合理性检查
    m0 = [x / total_frames for x in mean_stat[:3]]
    say(f"   前 3 维均值 = {[round(x,2) for x in m0]}（fbank log 域，量级应在 ±20 内）")


# ---------------------------------------------------------------- stage: list

def stage_list() -> None:
    sec("[list] 生成 data.list（jsonl）")
    for split in list(SPLITS) + ["train_smoke", "dev_smoke"]:
        d = PREP / split
        if not (d / "wav.scp").exists():
            continue
        wav_tbl: dict[str, str] = {}
        with (d / "wav.scp").open(encoding="utf-8") as f:
            for line in f:
                utt, _, p = line.rstrip("\n").partition(" ")
                wav_tbl[utt] = p

        n = 0
        lines = []
        with (d / "text").open(encoding="utf-8") as f:
            for line in f:
                utt, _, txt = line.rstrip("\n").partition(" ")
                assert utt in wav_tbl, f"{split}: {utt} 在 text 里但不在 wav.scp"
                # 字段与 WeNet tools/make_raw_list.py 完全一致：key / wav / txt
                lines.append(json.dumps({"key": utt, "wav": wav_tbl[utt], "txt": txt},
                                        ensure_ascii=False))
                n += 1
        write_lines(d / "data.list", lines)
        say(f"[{split}] data.list {n} 行")


# ---------------------------------------------------------------- stage: smoke

def stage_smoke() -> None:
    sec("[smoke] 生成冒烟子集（train_smoke / dev_smoke）")
    plans = (("train", "train_smoke", SMOKE_TRAIN), ("dev", "dev_smoke", SMOKE_DEV))
    for src, dst, k in plans:
        s = PREP / src
        if not (s / "wav.scp").exists():
            say(f"[{dst}] 源 {src} 不存在，跳过")
            continue
        scp = (s / "wav.scp").read_text(encoding="utf-8").splitlines()
        txt = {ln.split(" ", 1)[0]: ln for ln in
               (s / "text").read_text(encoding="utf-8").splitlines()}
        # 等间隔抽样，避免只取到少数几个说话人
        step = max(1, len(scp) // k)
        picked = scp[::step][:k]
        d = PREP / dst
        write_lines(d / "wav.scp", picked)
        write_lines(d / "text", [txt[ln.split(" ", 1)[0]] for ln in picked])
        spk = {ln.split(" ")[1].split("/")[-2] for ln in picked}
        say(f"[{dst}] {len(picked)} 条，覆盖 {len(spk)} 个说话人")


# ---------------------------------------------------------------- stage: conf

# 冒烟配置相对官方 train_unified_conformer.yaml 的覆盖项。
# 每一项都写清了"为什么"，因为**不改会踩坑**（尤其 warmup_steps）。
SMOKE_OVERRIDES: dict[str, object] = {
    "encoder_conf.num_blocks": 4,          # 12 → 4：少 2/3 计算量，5 分钟内能跑完
    "dataset_conf.speed_perturb": False,   # 关掉扰动，让 loss 曲线干净可解释
    "dataset_conf.spec_aug": False,        # 同上
    "dataset_conf.batch_conf.batch_size": 8,
    "max_epoch": 5,
    "log_interval": 10,                    # 100 → 10：否则 5 个 epoch 只打几条日志
    "scheduler_conf.warmup_steps": 500,    # ★ 25000 → 500，见下
}


def _set_nested(cfg: dict, dotted: str, val: object) -> None:
    keys = dotted.split(".")
    cur = cfg
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = val


def stage_conf() -> None:
    """
    生成 work/aishell/conf/smoke.yaml（从官方流式配置派生）。

    ★ 为什么必须改 warmup_steps：
      200 条 / batch 8 = 每 epoch 只有 25 步，5 epoch 一共 125 步。
      而官方 warmup_steps=25000 —— 整个训练期间学习率都停在爬升的最底部，
      **loss 会是一条平线**，极易被误判成"模型/数据有问题"。
      本质：scheduler 的时间尺度必须和数据规模匹配。
    """
    sec("[conf] 生成 smoke.yaml")
    import yaml

    src = (ROOT / "examples" / "wenet-main" / "examples" / "aishell" / "s0"
           / "conf" / "train_unified_conformer.yaml")
    if not src.exists():
        say(f"❌ 找不到官方配置模板: {src}")
        return

    with src.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k, v in SMOKE_OVERRIDES.items():
        _set_nested(cfg, k, v)

    header = [
        "# 由 scripts/prepare_aishell.py --stage conf 自动生成，请勿手改（改脚本）",
        f"# 派生自: {src.relative_to(ROOT)}",
        "#",
        "# 相对官方配置的覆盖项（冒烟训练用，目的是「先证明链路通」而不是「训出好模型」）:",
    ]
    for k, v in SMOKE_OVERRIDES.items():
        header.append(f"#   {k}: {v}")
    header += [
        "#",
        "# ★ 最关键的是 warmup_steps: 25000→500。",
        "#   200 条/batch 8 = 每 epoch 25 步，5 epoch 共 125 步；",
        "#   若沿用 25000，学习率全程停在爬升底部，loss 会是平线，会被误判成模型坏了。",
        "#",
        "# 路径说明：cmvn_file / symbol_table_path 是**相对 CWD** 的，",
        "#   所以训练时 CWD 必须是 work/aishell/。",
        "",
    ]
    out = WORK / "conf" / "smoke.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(header))
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    say(f"✅ {out}")
    say(f"   覆盖 {len(SMOKE_OVERRIDES)} 项；num_blocks={cfg['encoder_conf']['num_blocks']}, "
        f"max_epoch={cfg['max_epoch']}, warmup_steps={cfg['scheduler_conf']['warmup_steps']}")
    say(f"   流式相关保持官方默认: causal={cfg['encoder_conf']['causal']}, "
        f"use_dynamic_chunk={cfg['encoder_conf']['use_dynamic_chunk']}")


# ---------------------------------------------------------------- stage: verify

def stage_verify() -> None:
    sec("[verify] 校验产物")
    problems = []
    expect = {"train": EXPECT_TRAIN_UTTS, "dev": 14326, "test": 7176}

    for split in SPLITS:
        d = PREP / split
        for fn in ("wav.scp", "text", "data.list"):
            if not (d / fn).exists():
                problems.append(f"{split}/{fn} 缺失")
        if not (d / "wav.scp").exists():
            continue
        n_scp = len((d / "wav.scp").read_text(encoding="utf-8").splitlines())
        n_txt = len((d / "text").read_text(encoding="utf-8").splitlines())
        n_lst = len((d / "data.list").read_text(encoding="utf-8").splitlines())
        e = expect.get(split)
        mark = "" if e is None or n_scp == e else f"  ⚠️ 期望 {e}"
        say(f"[{split}] wav.scp {n_scp} | text {n_txt} | data.list {n_lst}{mark}")
        if not (n_scp == n_txt == n_lst):
            problems.append(f"{split}: 三个文件行数不一致 {n_scp}/{n_txt}/{n_lst}")
        if e is not None and n_scp != e:
            problems.append(f"{split}: 条数 {n_scp} != 期望 {e}")

    dct = PREP / "dict" / "lang_char.txt"
    if dct.exists():
        n = len(dct.read_text(encoding="utf-8").splitlines())
        say(f"[dict] lang_char.txt {n} 行" + ("" if n > 1000 else "  ⚠️ 偏少"))
        head = dct.read_text(encoding="utf-8").splitlines()[:3]
        if head != ["<blank> 0", "<unk> 1", "<sos/eos> 2"]:
            problems.append(f"dict 前 3 行不对: {head}")
    else:
        problems.append("dict/lang_char.txt 缺失")

    cmvn = PREP / "train" / "global_cmvn"
    if cmvn.exists():
        info = json.loads(cmvn.read_text(encoding="utf-8"))
        ok = (len(info["mean_stat"]) == FEAT_DIM and len(info["var_stat"]) == FEAT_DIM
              and info["frame_num"] > 0)
        say(f"[cmvn] {FEAT_DIM} 维, frame_num={info['frame_num']} " + ("✅" if ok else "❌"))
        if not ok:
            problems.append("global_cmvn 维度或 frame_num 异常")
    else:
        problems.append("train/global_cmvn 缺失")

    for s in ("train_smoke", "dev_smoke"):
        d = PREP / s
        if (d / "data.list").exists():
            say(f"[{s}] {len((d/'data.list').read_text(encoding='utf-8').splitlines())} 条")

    say("")
    if problems:
        say(f"❌ {len(problems)} 个问题：")
        for p in problems:
            say(f"   - {p}")
        raise SystemExit(1)
    say("✅ 全部通过")


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description="AISHELL-1 数据准备（WeNet recipe 的 Python 移植）")
    ap.add_argument("--stage",
                    choices=["all", "env", "dirs", "scp", "dict", "cmvn",
                             "list", "smoke", "conf", "verify"],
                    default="all")
    ap.add_argument("--workers", type=int, default=0, help="cmvn 并行进程数（默认 12）")
    ap.add_argument("--force", action="store_true", help="重算已存在的产物")
    args = ap.parse_args()

    # ★ 第一件事：把 FFmpeg 共享库挂上 PATH。
    #   必须在任何 torchaudio 解码动作之前（cmvn 阶段用 soundfile，
    #   但 env 阶段与后续训练要用 torchaudio.load）。
    ff = ensure_ffmpeg_on_path()
    say(f"FFmpeg 共享库: {ff if ff else '未找到（torchaudio.load 可能失败）'}")

    say(f"原始数据: {RAW}")
    say(f"工作目录: {WORK}")
    if not RAW.exists():
        say("❌ 原始数据目录不存在，先跑 scripts/download_aishell.py")
        return 1

    if args.stage in ("all", "dirs"):
        sec("[dirs] 建立目录")
        for s in list(SPLITS) + ["train_smoke", "dev_smoke", "dict"]:
            (PREP / s).mkdir(parents=True, exist_ok=True)
        (WORK / "conf").mkdir(parents=True, exist_ok=True)
        say(f"✅ {PREP}")

    if args.stage in ("all", "env"):
        stage_env()
    if args.stage in ("all", "scp"):
        stage_scp()
    if args.stage in ("all", "dict"):
        stage_dict()
    if args.stage in ("all", "cmvn"):
        stage_cmvn(force=args.force, workers=args.workers)
    # ⚠️ smoke 必须在 list 之前：list 要为 train_smoke/dev_smoke 也生成 data.list
    if args.stage in ("all", "smoke"):
        stage_smoke()
    if args.stage in ("all", "list"):
        stage_list()
    if args.stage in ("all", "conf"):
        stage_conf()
    if args.stage in ("all", "verify"):
        stage_verify()

    sec("完成")
    say(f"产物目录: {PREP}")
    say("下一步：生成 conf 并启动训练")
    return 0


if __name__ == "__main__":
    sys.exit(main())
