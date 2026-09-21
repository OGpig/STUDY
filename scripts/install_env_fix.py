#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
安装/卸载「让 torchaudio 找得到 FFmpeg」的环境补丁。

做的事：在 venv 的 site-packages 里放一个 `.pth` 文件，内容是

    <项目>/scripts/pyfix                 ← 把它加进 sys.path
    import wenet_dll_fix                 ← 导入即生效（见该模块的 docstring）

Python 启动时 `site.py` 会执行 `.pth` 里的 import 行，所以
**每一个** Python 进程（含 DataLoader 的 spawn worker）都会自动把 FFmpeg
目录注册进 DLL 搜索路径 —— 不用设 PATH，也不受 Git Bash 路径改写影响。

【为什么需要这个】
    torchaudio ≥2.9 靠 torchcodec 解码音频，torchcodec 需要 FFmpeg 共享库。
    而 WeNet 的 `decode_wav` 外面包了 `map_ignore_error`，
    FFmpeg 缺失时**样本会被静默丢弃**，最后表现成：
        ZeroDivisionError: division by zero   （在 executor.cv 里）
    —— 报错跟真正原因毫无关系。这个补丁从根上消除它。

【用法】
    python scripts/install_env_fix.py            # 安装 + 自检
    python scripts/install_env_fix.py --check    # 只看状态
    python scripts/install_env_fix.py --uninstall
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / ".venv" / "Lib" / "site-packages"
PYFIX = ROOT / "scripts" / "pyfix"
PTH_NAME = "wenet_ffmpeg_dll.pth"


def say(*a, **kw) -> None:
    print(*a, file=sys.stderr, **kw)


def pth_content() -> str:
    # .pth 里每个 import 行都会被 site.py 执行；不带 import 的行会被当作路径
    return f"{PYFIX.as_posix()}\nimport wenet_dll_fix\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="安装 torchaudio↔FFmpeg 的 DLL 修复")
    ap.add_argument("--check", action="store_true", help="只检查，不修改")
    ap.add_argument("--uninstall", action="store_true", help="移除 .pth")
    args = ap.parse_args()

    if not SITE.is_dir():
        say(f"❌ 找不到 site-packages: {SITE}")
        return 1
    pth = SITE / PTH_NAME

    if args.uninstall:
        if pth.exists():
            pth.unlink()
            say(f"🗑  已移除 {pth}")
        else:
            say("（本来就没装）")
        return 0

    mod = PYFIX / "wenet_dll_fix.py"
    if not mod.exists():
        say(f"❌ 缺少修复模块: {mod}")
        return 1

    want = pth_content()
    cur = pth.read_text(encoding="utf-8") if pth.exists() else None

    if cur == want:
        say(f"✅ 已安装且内容正确：{pth}")
    elif args.check:
        say(f"⚠️ 未安装或内容不符：{pth}")
        return 1
    else:
        pth.write_text(want, encoding="utf-8", newline="\n")
        say(f"✅ 已写入 {pth}")
        say(f"   内容:\n      {want.strip().replace(chr(10), chr(10) + '      ')}")

    # ---- 自检：在一个全新进程里验证（.pth 只在启动时生效，本进程不重读）
    say("\n[自检] 起一个全新 python 进程，不设任何环境变量，直接解码音频…")
    import subprocess
    code = (
        "import torchaudio,sys;"
        r"p=r'F:\embedded\prepare\data\aishell\raw\wav\train\S0002\BAC009S0002W0122.wav';"
        "w,sr=torchaudio.load(p);print('OK',tuple(w.shape),sr)"
    )
    r = subprocess.run([sys.executable, "-c", code],
                       capture_output=True, text=True)
    if r.returncode == 0 and "OK" in r.stdout:
        say(f"   ✅ 通过：{r.stdout.strip()}")
        say("\n   → 现在 torchaudio.load 在任何进程里都能用，**不需要设 PATH**。")
        return 0
    say(f"   ❌ 失败（returncode={r.returncode}）")
    say(f"   stdout: {r.stdout.strip()[:200]}")
    say(f"   stderr: {r.stderr.strip()[-300:]}")
    say("\n   排查：F:\\data\\toolchains\\ffmpeg-shared\\...\\bin 下是否有 avcodec-*.dll")
    return 1


if __name__ == "__main__":
    sys.exit(main())
