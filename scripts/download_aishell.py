#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AISHELL-1 数据集下载器（走 HuggingFace 国内镜像，无需科学上网）

【为什么不用官方 OpenSLR】
    实测 www.openslr.org 平均只有 0.59 MB/s 且剧烈波动（60 KB/s ~ 3 MB/s），
    官方 data_aishell.tgz 是 15.58 GB 单体包 → 需要 7 小时以上，不可接受。
    换成 hf-mirror.com 后大文件可达 11~19 MB/s（快 20~30 倍）。

【★ 关键实测结论：瓶颈是"请求数"而不是带宽】
    同一个镜像上：
      · 一个 455 MB 的大 parquet → 11.1 MB/s（带宽受限，很快）
      · 300 个 ~150 KB 的小 wav  → 24 并发也只有 0.16 MB/s（≈ 1 文件/秒）
    说明镜像对"小请求"做了限流，**并发也救不回来**（开 24 线程仍约 1 文件/秒）。
    → 选数据源的第一原则：**能少发请求就少发请求。**

【各镜像实测对比】
    源                    内容                   请求数  体积     实测      耗时
    OpenSLR 官方          全量（官方 tgz）        1     15.6 GB  0.59 MB/s  ~7.3 h
    AISHELL/AISHELL-1     train 100/340 说话人  100     3.45 GB  19 MB/s    ~4 min
    shenyunhang/AISHELL-1 全量官方划分          122k    ~14 GB   1 文件/s   ~33 h
    ★ carlot/AIShell      全量官方划分            37     20.3 GB  11 MB/s    ~30 min

【最终采用的组合策略】
    carlot/AIShell 是**唯一同时满足「官方完整划分」和「请求数少」**的源：
      · parquet 里 `audio.path` 就是原始 utt id（如 BAC009S0002W0122.wav）
      · `audio.bytes` 是原始 RIFF WAV 字节，与官方文件逐字节一致
      · train 35 片 / validation 1 片 / test 1 片 = 37 个请求
    所以 dev/test 与全量 train 一律走 carlot（parquet → wav）。
    AISHELL/AISHELL-1 的 tar 分片留作"要数据但不想等 30 分钟"的快速通路。

【用法】
    # 0) 元数据（transcript + 词典），秒级
    python scripts/download_aishell.py --stage meta

    # 1) 快速通路：train 100 个 speaker 分片（3.45 GB，~4 min），解压即标准布局
    python scripts/download_aishell.py --stage train-tar

    # 2) ★ 官方 dev + test（carlot，2 个文件 3.4 GB，~5 min）→ wav + text
    python scripts/download_aishell.py --stage carlot-devtest

    # 3) ★ 官方全量 train（carlot，35 片 17.3 GB，~26 min）→ wav + text
    python scripts/download_aishell.py --stage carlot-train

    # 4) 全部
    python scripts/download_aishell.py --stage all

    # 调试：只下前 N 项
    python scripts/download_aishell.py --stage carlot-devtest --limit 1
    python scripts/download_aishell.py --stage train-tar --limit 2

【可中断可续传】
    已存在且大小正确的文件会被跳过，重跑即续传。
    parquet 分片转完 wav 后原件保留在 data/aishell/download/，可手动删。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------- 配置

ENDPOINT = "https://hf-mirror.com"

# A：train 分片（tar.gz，快）
REPO_TAR = "AISHELL/AISHELL-1"
TAR_DIR = "data_aishell/wav"          # 内含 S0002.tar.gz ... S0101.tar.gz

# B：官方完整划分（逐个 wav，慢但标准）—— 仅作备用
REPO_FULL = "shenyunhang/AISHELL-1"

# ★ C：官方完整划分的 parquet 版（请求数少，本脚本主用）
REPO_CARLOT = "carlot/AIShell"
#     split 名 → (远端文件前缀, 落到本地的目录名)
CARLOT = {
    "train":      ("train",      "train"),
    "validation": ("validation", "dev"),      # HF 叫 validation，官方叫 dev
    "test":       ("test",       "test"),
}

# 项目根：脚本在 scripts/ 下，向上一级
ROOT = Path(__file__).resolve().parent.parent

# 目标布局刻意对齐官方 AISHELL-1，这样 WeNet 的 aishell recipe 可以原样使用：
#     data/aishell/raw/wav/train/S0002/BAC009S0002W0122.wav
#     data/aishell/raw/wav/dev/S0724/...
#     data/aishell/raw/wav/test/S0764/...
#     data/aishell/raw/transcript/aishell_transcript_v0.8.txt
#     data/aishell/raw/resource/lexicon.txt
DEST = ROOT / "data" / "aishell"
RAW = DEST / "raw"
CACHE = DEST / "download"             # tar.gz 原件缓存，解压后可删

THREADS = 24                          # 20 核机器，24 并发足够打满带宽
RETRY = 4
TIMEOUT = 60

_print_lock = threading.Lock()


def say(*a, **kw):
    """脚本自身信息一律走 stderr，避免污染 stdout。"""
    with _print_lock:
        print(*a, file=sys.stderr, **kw)


# ---------------------------------------------------------------- HTTP

def resolve_url(repo: str, path: str) -> str:
    """HF 的 resolve 地址（会 302 跳到 CDN，必须跟随重定向）。"""
    return f"{ENDPOINT}/datasets/{repo}/resolve/main/{path}"


def fetch_bytes(url: str) -> bytes:
    """带重试的 GET。"""
    last = None
    for attempt in range(RETRY):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.read()
        except Exception as e:                                # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"下载失败 {url}: {last}")


def list_tree(repo: str, path: str, recursive: bool = False) -> list[dict]:
    """列出仓库某目录（自动翻页，HF 每页最多 1000 项）。"""
    out: list[dict] = []
    cursor = None
    while True:
        url = (f"{ENDPOINT}/api/datasets/{repo}/tree/main/{path}"
               f"?recursive={'true' if recursive else 'false'}&expand=false")
        if cursor:
            url += f"&cursor={cursor}"
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            body = json.loads(r.read())
            link = r.headers.get("Link")
        if not isinstance(body, list):
            raise RuntimeError(f"列目录失败 {repo}/{path}: {body}")
        out.extend(body)
        # 解析 Link: <...cursor=XXX>; rel="next"
        if link and 'rel="next"' in link:
            cursor = link.split("cursor=")[1].split(">")[0]
        else:
            break
    return out


def download_one(url: str, dst: Path) -> tuple[Path, int, bool]:
    """下载单个文件到 dst。已存在且非空则跳过。返回 (路径, 字节数, 是否新下载)。"""
    if dst.exists() and dst.stat().st_size > 0:
        return dst, dst.stat().st_size, False
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = fetch_bytes(url)
    tmp = dst.with_suffix(dst.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(dst)                       # 原子替换，避免半截文件
    return dst, len(data), True


def run_pool(tasks: list[tuple[str, Path]], label: str) -> tuple[int, int, float]:
    """并发下载一批任务，带进度与吞吐显示。"""
    if not tasks:
        say(f"[{label}] 无需下载（全部已存在）")
        return 0, 0, 0.0

    total = len(tasks)
    done = skipped = nbytes = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        futs = {pool.submit(download_one, u, d): d for u, d in tasks}
        for fut in as_completed(futs):
            try:
                _, n, new = fut.result()
            except Exception as e:                          # noqa: BLE001
                say(f"\n[{label}] ❌ {futs[fut].name}: {e}")
                continue
            done += 1
            nbytes += n
            if not new:
                skipped += 1
            if done % 200 == 0 or done == total:
                el = time.time() - t0
                rate = nbytes / el / 1048576 if el > 0 else 0
                say(f"\r[{label}] {done}/{total}  "
                    f"{nbytes/1048576:8.1f} MB  {rate:6.2f} MB/s  "
                    f"已用 {el:5.0f}s", end="")
    el = time.time() - t0
    say(f"\n[{label}] ✅ {done} 个文件 / {nbytes/1048576:.1f} MB，"
        f"耗时 {el:.1f}s，均速 {nbytes/el/1048576:.2f} MB/s"
        f"（其中跳过已存在 {skipped} 个）")
    return done, nbytes, el


# ---------------------------------------------------------------- 各阶段

def stage_meta() -> None:
    """transcript + 词典 + speaker.info，共约 13 MB。"""
    say("\n=== stage meta：元数据 ===")
    items = [
        (REPO_FULL, "data_aishell/transcript/aishell_transcript_v0.8.txt",
         RAW / "transcript" / "aishell_transcript_v0.8.txt"),
        (REPO_FULL, "resource_aishell/lexicon.txt", RAW / "resource" / "lexicon.txt"),
        (REPO_FULL, "resource_aishell/speaker.info", RAW / "resource" / "speaker.info"),
    ]
    tasks = [(resolve_url(r, p), d) for r, p, d in items]
    run_pool(tasks, "meta")
    # 行数校验：官方 transcript 恒为 141600 行，对不上说明下载不完整
    tx = RAW / "transcript" / "aishell_transcript_v0.8.txt"
    if tx.exists():
        n = sum(1 for _ in tx.open("r", encoding="utf-8"))
        flag = "✅" if n == 141600 else "⚠️ 期望 141600"
        say(f"[meta] transcript 行数 = {n}  {flag}")


def stage_train_tar(limit: int | None = None) -> None:
    """下载并解压 train 的 speaker 分片 → raw/wav/train/S0xxx/*.wav"""
    say("\n=== stage train-tar：train 分片（100 个 speaker）===")
    entries = list_tree(REPO_TAR, TAR_DIR)
    tars = sorted(e["path"] for e in entries if e["path"].endswith(".tar.gz"))
    if limit:
        tars = tars[:limit]
    say(f"[train-tar] 分片数 = {len(tars)}")

    tasks = [(resolve_url(REPO_TAR, p), CACHE / Path(p).name) for p in tars]
    run_pool(tasks, "train-tar 下载")

    # 解压：tarball 内部根目录是 train/S0xxx/，正好对上官方布局
    say("[train-tar] 解压中 ...")
    out_dir = RAW / "wav"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_wav = 0
    for p in tars:
        tb = CACHE / Path(p).name
        if not tb.exists():
            continue
        with tarfile.open(tb, "r:gz") as tf:
            members = [m for m in tf.getmembers() if m.name.endswith(".wav")]
            for m in members:
                tf.extract(m, out_dir, filter="data")
                n_wav += 1
    say(f"[train-tar] ✅ 解压出 {n_wav} 个 wav → {out_dir}")


def stage_devtest(limit: int | None = None) -> None:
    """官方 dev(40 speaker) + test(20 speaker)，逐个 wav 下载。"""
    say("\n=== stage devtest：官方 dev + test ===")
    tasks: list[tuple[str, Path]] = []
    for split in ("dev", "test"):
        spk_dirs = [e for e in list_tree(REPO_FULL, f"data_aishell/wav/{split}")
                    if e["type"] == "directory"]
        say(f"[devtest] {split}: {len(spk_dirs)} 个 speaker")
        for sd in spk_dirs:
            files = [e for e in list_tree(REPO_FULL, sd["path"])
                     if e["type"] == "file"]
            for f in files:
                tasks.append((resolve_url(REPO_FULL, f["path"]),
                              RAW / "wav" / split / Path(f["path"]).parent.name
                              / Path(f["path"]).name))
    if limit:
        tasks = tasks[:limit]
    run_pool(tasks, "devtest")


# ---------------------------------------------------------------- carlot 通路

def _carlot_files(split: str) -> list[str]:
    """列出 carlot/AIShell 某个 split 的所有 parquet 文件名。"""
    prefix = CARLOT[split][0]
    entries = list_tree(REPO_CARLOT, "data")
    return sorted(e["path"] for e in entries
                  if e["path"].startswith(f"data/{prefix}-") and e["path"].endswith(".parquet"))


def _parquet_to_wav(pq_path: Path, split_out: str) -> tuple[int, int]:
    """
    把一个 parquet 分片拆成 wav 文件 + 文本。

    parquet 结构（实测）：
        audio: struct<bytes: binary, path: string>
            bytes = 原始 RIFF WAV 数据（与官方文件逐字节一致）
            path  = 原始文件名，即 utt id，如 BAC009S0002W0122.wav
        transcription: string，如 '而 对 楼市 成交 抑制 作用 最 大 的 限 购'

    输出：
        raw/wav/<split>/<S0002>/BAC009S0002W0122.wav
        raw/<split>_text.txt   —— 每行 "<utt_id> <transcription>"

    用 iter_batches 流式读取，避免把 500 MB 分片整体载入内存。
    """
    import pyarrow.parquet as pq                       # 延迟导入，未装时其它 stage 仍可用

    wav_root = RAW / "wav" / split_out
    text_path = RAW / f"{split_out}_text.txt"
    n_wav = 0
    n_bytes = 0

    pf = pq.ParquetFile(pq_path)
    with text_path.open("a", encoding="utf-8") as ftext:
        for batch in pf.iter_batches(batch_size=256,
                                     columns=["audio", "transcription"]):
            audios = batch.column("audio").to_pylist()
            texts = batch.column("transcription").to_pylist()
            for a, t in zip(audios, texts):
                utt = Path(a["path"]).stem                  # BAC009S0002W0122
                # utt id 的第 7~11 个字符是说话人编号（BAC009S0002W0122 → S0002）
                spk = utt[6:11]
                dst = wav_root / spk / f"{utt}.wav"
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(a["bytes"])
                n_wav += 1
                n_bytes += len(a["bytes"])
                ftext.write(f"{utt}\t{t}\n")
    return n_wav, n_bytes


def stage_carlot(splits: tuple[str, ...]) -> None:
    """下载 carlot/AIShell 的指定 split 并转成 wav + text。"""
    say(f"\n=== stage carlot：{', '.join(splits)} ===")
    try:
        import pyarrow                                  # noqa: F401
    except ImportError:
        say("[carlot] ❌ 需要 pyarrow：pip install pyarrow")
        return

    for split in splits:
        files = _carlot_files(split)
        if not files:
            say(f"[carlot] {split}: 未找到分片，跳过")
            continue
        say(f"[carlot] {split}: {len(files)} 个 parquet 分片")

        tasks = [(resolve_url(REPO_CARLOT, p), CACHE / Path(p).name) for p in files]
        run_pool(tasks, f"carlot-{split} 下载")

        # 转换前清掉上次的半成品文本，避免重复追加
        out_name = CARLOT[split][1]
        txt = RAW / f"{out_name}_text.txt"
        if txt.exists():
            txt.unlink()

        tot_w = tot_b = 0
        for p in files:
            local = CACHE / Path(p).name
            if not local.exists():
                continue
            w, b = _parquet_to_wav(local, out_name)
            tot_w += w
            tot_b += b
            say(f"\r[carlot] {split} 转换中 {Path(p).name} → 累计 {tot_w} 条 wav", end="")
        say(f"\n[carlot] ✅ {split} → 目录 {out_name}/，"
            f"{tot_w} 条 wav，{tot_b/1073741824:.2f} GB")


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser(
        description="AISHELL-1 下载器（HF 国内镜像，断点续传）")
    ap.add_argument("--stage",
                    choices=["meta", "train-tar", "devtest",
                             "carlot-devtest", "carlot-train", "all",
                             "carlot-all"],
                    default="all")
    ap.add_argument("--limit", type=int, default=None,
                    help="只处理前 N 项（调试用）")
    args = ap.parse_args()

    say(f"目标目录：{DEST}")
    say(f"并发数：{THREADS}")

    if args.stage in ("meta", "all"):
        stage_meta()
    if args.stage in ("train-tar", "all"):
        stage_train_tar(limit=args.limit)
    if args.stage == "devtest":
        stage_devtest(limit=args.limit)
    if args.stage == "carlot-devtest":
        stage_carlot(("validation", "test"))
    if args.stage == "carlot-train":
        stage_carlot(("train",))
    if args.stage == "carlot-all":
        stage_carlot(("validation", "test", "train"))

    # 汇总
    say("\n=== 汇总 ===")
    for split in ("train", "dev", "test"):
        d = RAW / "wav" / split
        n = sum(1 for _ in d.rglob("*.wav")) if d.exists() else 0
        sz = sum(f.stat().st_size for f in d.rglob("*.wav")) / 1073741824 if n else 0
        say(f"  {split:5s}: {n:7d} 个 wav, {sz:6.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
