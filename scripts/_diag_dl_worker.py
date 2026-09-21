#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断脚本：DataLoader 的 spawn worker 能否读到音频。

为什么需要单独写成文件：Windows 的 multiprocessing 用 spawn，
子进程会重新 import 主模块 —— 所以主模块必须是真实文件，
用 `python - <<EOF`（stdin）会直接 OSError: Invalid argument '<stdin>'。

用法：python scripts/_diag_dl_worker.py
"""

import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset

WAV = r"F:\embedded\prepare\data\aishell\raw\wav\train\S0002\BAC009S0002W0122.wav"


def _has_ffmpeg() -> bool:
    return any("ffmpeg-shared" in p for p in os.environ.get("PATH", "").split(os.pathsep))


class D(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, i: int) -> int:
        import torchaudio
        has = _has_ffmpeg()
        try:
            w, sr = torchaudio.load(WAV)
            print(f"[worker {os.getpid()}] PATH有FFmpeg={has} OK {tuple(w.shape)} @{sr}",
                  file=sys.stderr, flush=True)
        except Exception as e:                                    # noqa: BLE001
            print(f"[worker {os.getpid()}] PATH有FFmpeg={has} FAIL {str(e)[:120]}",
                  file=sys.stderr, flush=True)
        return i


if __name__ == "__main__":
    print(f"[main {os.getpid()}] PATH有FFmpeg={_has_ffmpeg()}", file=sys.stderr, flush=True)
    for nw in (0, 2):
        print(f"\n===== num_workers={nw} =====", file=sys.stderr, flush=True)
        try:
            kw = {"num_workers": nw}
            if nw > 0:
                kw["prefetch_factor"] = 2
            list(DataLoader(D(), batch_size=1, **kw))
            print(f"[main] num_workers={nw} done", file=sys.stderr, flush=True)
        except Exception as e:                                    # noqa: BLE001
            print(f"[main] num_workers={nw} {type(e).__name__}: {str(e)[:160]}",
                  file=sys.stderr, flush=True)
