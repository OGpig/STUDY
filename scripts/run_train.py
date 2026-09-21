#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
WeNet 训练启动器（Windows 适配版）

把「手动跑 WeNet 训练」在 Windows 上必踩的 5 件事一次做掉，调用方只需要给
真正有意义的几个参数。

【为什么需要它 —— 官方 run.sh 在 Windows 上不可用，而手动敲要补 5 件事】

  1. **FFmpeg 共享库必须挂 PATH**
     torchaudio ≥2.9 靠 torchcodec 解码音频，torchcodec 需要 avcodec/avformat 等 DLL。
     ⚠️ 关键：**PATH 必须在这里用绝对 Windows 路径设置**。
        实测在 Git Bash 里 `export PATH="F:/data/...:$PATH"` 会被 shim 改写成
        `...PortableGit\versions\...\data\toolchains\...`，于是主进程和 DataLoader
        子进程都找不到 FFmpeg，报 "Could not load libtorchcodec"。
        在 Python 里设就不会被改写，而且**子进程会继承**。

  2. **PYTHONPATH 要指向 WeNet 源码根目录**（`import wenet.*` 才找得到）。

  3. **model_dir / tensorboard_dir 要先建好**
     官方 run.sh 的 stage 4 开头有 `mkdir -p $dir`；手动跑容易漏，
     然后报 `FileNotFoundError: exp/smoke\train.yaml`（train_utils 要往里写配置）。

  4. **rdzv_backend 用 `static`，不要用 c10d**
     本机 torch 2.11.0+cu128 的 Windows 轮子有 libuv bug（见 patch_torch_libuv.py），
     c10d/standalone 会在建 TCPStore 时直接崩。static 走另一条路径。

  5. **CWD 必须是 `work/aishell`**
     yaml 里的 `data/train/global_cmvn`、`data/dict/lang_char.txt` 都是**相对 CWD** 的。

  另外：torch 的 Windows 轮子也不认 `--ddp.dist_backend nccl`（没有 NCCL），必须 `gloo`。
  官方 recipe 写的是 nccl，这里替它改掉。

【用法】
    # 冒烟测试（200 条 / 5 epoch）
    python scripts/run_train.py --preset smoke

    # 自定义
    python scripts/run_train.py \
        --config conf/smoke.yaml \
        --train_data data/train_smoke/data.list \
        --cv_data  data/dev_smoke/data.list \
        --model_dir exp/smoke \
        --tensorboard_dir tensorboard \
        --num_workers 4

    # 透传额外参数给 wenet/bin/train.py
    python scripts/run_train.py --preset smoke -- --log_interval 5

【退出码】训练本身的退出码；启动前的检查失败返回 1。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WENET = ROOT / "examples" / "wenet-main"
WORK = ROOT / "work" / "aishell"
PY = ROOT / ".venv" / "Scripts" / "python.exe"

FFMPEG_CANDIDATES = [
    Path(r"F:\data\toolchains\ffmpeg-shared\ffmpeg-master-latest-win64-gpl-shared\bin"),
    Path(r"F:\data\toolchains\ffmpeg-shared\bin"),
]

PRESETS = {
    "smoke": dict(
        config="conf/smoke.yaml",
        train_data="data/train_smoke/data.list",
        cv_data="data/dev_smoke/data.list",
        model_dir="exp/smoke",
    ),
}


def say(*a, **kw) -> None:
    print(*a, file=sys.stderr, **kw)


def find_ffmpeg() -> Path | None:
    for p in FFMPEG_CANDIDATES:
        if p.is_dir() and any(p.glob("avcodec-*.dll")):
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="WeNet 训练启动器（Windows 适配）")
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None)
    ap.add_argument("--config", default=None, help="相对 work/aishell 的 yaml")
    ap.add_argument("--train_data", default=None)
    ap.add_argument("--cv_data", default=None)
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--tensorboard_dir", default="tensorboard")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--prefetch", type=int, default=10)
    ap.add_argument("--device", default=None,
                    help="不填则自动（有 CUDA 用 cuda，否则 cpu）")
    ap.add_argument("--port", type=int, default=29560, help="static rendezvous 端口")
    ap.add_argument("rest", nargs="*", help="透传给 wenet/bin/train.py 的参数（放在 -- 之后）")
    args = ap.parse_args()

    # ---------- 1. 合并 preset
    cfg = dict(config=None, train_data=None, cv_data=None, model_dir=None)
    if args.preset:
        cfg.update(PRESETS[args.preset])
    for k in ("config", "train_data", "cv_data", "model_dir"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    missing = [k for k, v in cfg.items() if not v]
    if missing:
        say(f"❌ 缺少参数: {', '.join(missing)}（要么用 --preset，要么显式给全）")
        return 1

    # ---------- 2. 检查
    if not PY.exists():
        say(f"❌ 找不到 venv python: {PY}")
        return 1
    train_py = WENET / "wenet" / "bin" / "train.py"
    if not train_py.exists():
        say(f"❌ 找不到 {train_py}")
        return 1
    if not (WORK / cfg["config"]).exists():
        say(f"❌ 配置不存在: {WORK / cfg['config']}")
        say("   先生成：python scripts/prepare_aishell.py --stage conf")
        return 1

    # ---------- 3. 环境（★ 关键：在 Python 里设 PATH，不会被 Git Bash shim 改写）
    ff = find_ffmpeg()
    if ff is None:
        say("⚠️ 没找到 FFmpeg 共享库 —— torchaudio 解码会失败（训练起不来）")
        say("   见 docs/ENV.md 风险 6")
    else:
        os.environ["PATH"] = str(ff) + os.pathsep + os.environ.get("PATH", "")
        say(f"FFmpeg : {ff}")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(WENET) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")

    # 设备
    if args.device is None:
        try:
            import torch  # noqa: PLC0415
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:                                        # noqa: BLE001
            args.device = "cpu"

    # ---------- 4. 建目录（官方 run.sh 里的 mkdir -p，手动跑最容易漏）
    model_dir = WORK / cfg["model_dir"]
    tb_dir = WORK / args.tensorboard_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 5. 拼命令
    cmd = [
        str(PY), "-m", "torch.distributed.run",
        "--nnodes=1", "--nproc_per_node=1",
        # ★ rdzv 用 static：c10d / --standalone 会踩 torch 的 libuv bug
        "--rdzv_backend=static",
        f"--rdzv_endpoint=127.0.0.1:{args.port}",
        str(train_py),
        "--train_engine", "torch_ddp",
        "--ddp.dist_backend", "gloo",          # Windows 没有 NCCL
        "--config", cfg["config"],
        "--data_type", "raw",
        "--train_data", cfg["train_data"],
        "--cv_data", cfg["cv_data"],
        "--model_dir", cfg["model_dir"],
        "--tensorboard_dir", args.tensorboard_dir,
        "--num_workers", str(args.num_workers),
        "--prefetch", str(args.prefetch),
        "--device", args.device,
    ] + list(args.rest)

    say(f"CWD    : {WORK}")
    say(f"device : {args.device}")
    say(f"model  : {cfg['model_dir']}")
    say(f"tensor : {args.tensorboard_dir}")
    say("")
    say("命令: " + " ".join(cmd))
    say("")

    # ---------- 6. 执行（CWD 必须是 work/aishell）
    proc = subprocess.run(cmd, cwd=str(WORK), env=env)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
