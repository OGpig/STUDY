#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
让 torchaudio 能解码音频：把 FFmpeg 共享库目录注册进**进程的 DLL 搜索路径**。

【为什么需要】
    torchaudio ≥ 2.9 把音频解码改为依赖 torchcodec，而 torchcodec 自己不捆绑
    FFmpeg 共享库（avcodec/avformat/avutil/swresample/swscale）。
    缺了它们，`torchaudio.load()` 会报 "Could not load libtorchcodec"。

    而 WeNet 的数据管线 `decode_wav` 就用 `torchaudio.load`，并且外面包了
    `map_ignore_error` —— **样本级异常会被静默丢弃**，于是全部样本被丢掉，
    最后表现成一个毫不相干的
        ZeroDivisionError: division by zero   （在 executor.cv 里）
    这个报错跟"FFmpeg 没装"看起来毫无关系，极难排查。

【为什么不用「把目录加到 PATH」】
    两种方式都能让 torchcodec 找到 FFmpeg，但 PATH 有两个坑：
      1. **Git Bash 会改写 PATH 里的 `F:/...`**（变成 `...PortableGit\versions\...\data\...`），
         于是主进程和 spawn 出来的 DataLoader worker 都找不到 → 又变成静默丢样本。
      2. PATH 是**每个 shell 都要记得设**的东西，忘了就复现上面的假错误。
    `os.add_dll_directory()` 是 CPython 3.8+ 的官方 API，
    直接把目录注册进当前进程的 DLL 搜索路径，**不依赖 PATH，也不受 shell 影响**。

【怎么自动生效】
    由 `scripts/install_env_fix.py` 在 venv 的 site-packages 放一个 `.pth`，
    内容是「把 scripts/pyfix 加进 sys.path」+「import wenet_dll_fix」。
    Python 启动时 `site.py` 会执行 `.pth` 里的 import 行 →
    每个进程（**包括 DataLoader 的 spawn worker**）启动时自动注册。

    ⚠️ 刻意不做任何重量级 import（不碰 torch），对启动开销几乎无影响。
"""

from __future__ import annotations

import os
import sys

# FFmpeg shared 构建的 bin 目录（含 avcodec-63.dll 等）
FFMPEG_CANDIDATES = [
    r"F:\data\toolchains\ffmpeg-shared\ffmpeg-master-latest-win64-gpl-shared\bin",
    r"F:\data\toolchains\ffmpeg-shared\bin",
]

_applied_dir: str | None = None
_handle = None          # 必须保持存活，否则 Windows 会注销该目录
_done = False


def find_ffmpeg() -> str | None:
    for d in FFMPEG_CANDIDATES:
        if os.path.isdir(d) and any(
                f.startswith("avcodec-") and f.endswith(".dll")
                for f in os.listdir(d)):
            return d
    return None


def apply() -> str | None:
    """把 FFmpeg 目录注册进本进程的 DLL 搜索路径。返回生效目录，None 表示没找到。"""
    global _applied_dir, _handle, _done
    if _done:
        return _applied_dir
    _done = True

    if sys.platform != "win32":
        return None
    d = find_ffmpeg()
    if d is None:
        return None
    try:
        _handle = os.add_dll_directory(d)
    except (OSError, AttributeError):
        return None
    _applied_dir = d
    return d


def _self_test() -> None:
    got = apply()
    print(f"FFmpeg DLL 目录: {got or '未找到'}")
    try:
        import torchaudio
    except ImportError:
        print("（torchaudio 未安装）")
        return
    probe = r"F:\embedded\prepare\data\aishell\raw\wav\train\S0002\BAC009S0002W0122.wav"
    if not os.path.exists(probe):
        print("（没有测试音频，跳过实际解码）")
        return
    try:
        w, sr = torchaudio.load(probe)
        print(f"✅ torchaudio.load OK: {tuple(w.shape)} @{sr}Hz")
    except Exception as e:                                        # noqa: BLE001
        print(f"❌ torchaudio.load 失败: {str(e)[:180]}")


if __name__ == "__main__":
    _self_test()
else:
    # 被 .pth 自动 import 时立刻生效
    apply()
