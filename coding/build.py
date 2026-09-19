# -*- coding: utf-8 -*-
"""
C++ 算法练习：一键编译 + 运行 + 计时
=====================================

本机有两套编译器，脚本会自动探测并配好环境（不需要 Developer Command Prompt、
不经过 cmd.exe，也不依赖系统 PATH 里有没有编译器）：

  gcc  (默认)  MinGW-W64 g++ 16.2.0   F:\\data\\toolchains\\mingw64
  msvc         MSVC 14.51 + Win SDK    VS 18 BuildTools

用法
----
  # 编译 + 运行（自动喂 tests/<同名>_1.in，没有则读空输入）
  python coding/build.py coding/solutions/P01_LIS.cpp

  # 只编译不运行
  python coding/build.py coding/solutions/P01_LIS.cpp --no-run

  # 换编译器 / 换标准 / 调试版
  python coding/build.py coding/solutions/P01_LIS.cpp --cc msvc
  python coding/build.py coding/solutions/P01_LIS.cpp --std c++20
  python coding/build.py coding/solutions/P01_LIS.cpp --dbg

  # 指定输入文件 / 手动输入（不喂文件，直接交互）
  python coding/build.py coding/solutions/P01_LIS.cpp --in myinput.txt
  python coding/build.py coding/solutions/P01_LIS.cpp --manual

输出纪律（重要）
----------------
  **stdout 只放被测程序的 stdout，一个字节都不多。**
  脚本自己的横幅、编译命令、耗时、[stderr] 回显，全部走 stderr。
  所以下面这种用法是干净的，可以直接和期望输出 diff：

      python coding/build.py xxx.cpp > out.txt
      python coding/build.py xxx.cpp 2> diag.txt

产物
----
  可执行文件统一放 coding/build/<源文件名>.exe（已在 .gitignore 里忽略）
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

CODING_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODING_DIR.parent
BUILD_DIR = CODING_DIR / "build"

MINGW_BIN = Path(r"F:\data\toolchains\mingw64\bin")
VS_ROOT = Path(r"C:\Program Files (x86)\Microsoft Visual Studio")
SDK_ROOT = Path(r"C:\Program Files (x86)\Windows Kits\10")


def say(*args, **kwargs) -> None:
    """脚本自己的信息一律走 stderr —— 保证 stdout 只有被测程序的输出"""
    print(*args, file=sys.stderr, **kwargs)


# ──────────────────────────────────────────────── 工具链探测
def find_gcc() -> Path | None:
    gxx = MINGW_BIN / "g++.exe"
    return gxx if gxx.exists() else None


def find_msvc() -> tuple[Path, str, str] | None:
    """返回 (cl.exe, VC工具根, SDK版本号)；版本号目录自动探测，不硬编码"""
    cl_list = glob.glob(str(VS_ROOT / "*" / "BuildTools" / "VC" / "Tools" / "MSVC" / "*"
                          / "bin" / "Hostx64" / "x64" / "cl.exe"))
    if not cl_list:
        return None
    cl = Path(sorted(cl_list)[-1])                       # 取版本号最大的
    vc_tools = cl.parents[3]                             # .../VC/Tools/MSVC/<ver>
    sdk_vers = sorted(p.name for p in (SDK_ROOT / "Include").glob("10.*"))
    if not sdk_vers:
        return None
    return cl, str(vc_tools), sdk_vers[-1]


def build_env_gcc() -> dict:
    env = os.environ.copy()
    env["PATH"] = f"{MINGW_BIN}{os.pathsep}" + env.get("PATH", "")
    return env


def build_env_msvc(vc_tools: str, sdk_ver: str) -> dict:
    """手工拼 INCLUDE / LIB / PATH —— 绕开 cmd.exe 和 vcvars64.bat"""
    env = os.environ.copy()
    inc = [rf"{vc_tools}\include",
           rf"{SDK_ROOT}\Include\{sdk_ver}\ucrt",
           rf"{SDK_ROOT}\Include\{sdk_ver}\shared",
           rf"{SDK_ROOT}\Include\{sdk_ver}\um"]
    lib = [rf"{vc_tools}\lib\x64",
           rf"{SDK_ROOT}\Lib\{sdk_ver}\ucrt\x64",
           rf"{SDK_ROOT}\Lib\{sdk_ver}\um\x64"]
    env["INCLUDE"] = ";".join(inc)                       # Windows 用 ; 分隔，别用 :
    env["LIB"] = ";".join(lib)
    env["PATH"] = os.pathsep.join([
        rf"{vc_tools}\bin\Hostx64\x64",                  # link.exe 靠它才能被找到
        rf"{SDK_ROOT}\bin\{sdk_ver}\x64",
        env.get("PATH", ""),
    ])
    return env


# ──────────────────────────────────────────────── 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="C++ 算法练习一键编译运行")
    ap.add_argument("source", help="要编译的 .cpp 文件")
    ap.add_argument("--cc", choices=["gcc", "msvc"], default="gcc", help="编译器（默认 gcc）")
    ap.add_argument("--std", default="c++17", help="C++ 标准（默认 c++17）")
    ap.add_argument("--dbg", action="store_true", help="调试版：-O0 -g（gcc 还会关掉 -Wall 的优化相关提示）")
    ap.add_argument("--no-run", action="store_true", help="只编译，不运行")
    ap.add_argument("--in", dest="infile", default=None, help="指定输入文件")
    ap.add_argument("--manual", action="store_true", help="不喂输入文件，直接交互输入")
    ap.add_argument("--timeout", type=float, default=10.0, help="运行超时秒数（默认 10）")
    args = ap.parse_args()

    src = Path(args.source)
    if not src.is_absolute():
        src = (Path.cwd() / src).resolve()
    if not src.exists():
        say(f"[错误] 找不到源文件：{src}")
        return 2

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    exe = BUILD_DIR / (src.stem + ".exe")
    opt = "-O0" if args.dbg else "-O2"          # gcc 用 -O2，msvc 下面单独换算成 /O2

    say("=" * 72)
    say(f"源文件 : {src}")
    say(f"编译器 : {args.cc}")
    say(f"输出   : {exe}")

    if args.cc == "gcc":
        gxx = find_gcc()
        if gxx is None:
            say(f"[错误] 找不到 g++：{MINGW_BIN / 'g++.exe'}")
            return 2
        cmd = [str(gxx), f"-std={args.std}", opt, "-Wall", "-Wextra",
               "-fno-omit-frame-pointer", str(src), "-o", str(exe)]
        if args.dbg:
            cmd.insert(1, "-g")
        env = build_env_gcc()
        say(f"命令   : g++.exe -std={args.std} {opt} -Wall -Wextra -o {exe.name}")
    else:
        found = find_msvc()
        if found is None:
            say("[错误] 找不到 MSVC（cl.exe）或 Windows SDK")
            return 2
        cl, vc_tools, sdk_ver = found
        msvc_opt = "/Od" if args.dbg else "/O2"
        # /utf-8 必加（源码含中文注释会报 C4819）
        # NOMINMAX 作为编译宏传入，避免 windows.h 污染 std::max / std::min
        msvc_std = "/std:" + args.std          # c++17 -> /std:c++17
        cmd = [str(cl), "/nologo", "/utf-8", "/EHsc", msvc_std,
               "/DNOMINMAX", "/DWIN32_LEAN_AND_MEAN", msvc_opt,
               f"/Fe:{exe}", str(src)]
        if args.dbg:
            cmd += ["/Zi"]
        env = build_env_msvc(vc_tools, sdk_ver)
        say(f"命令   : cl.exe /utf-8 {msvc_std} {msvc_opt} /Fe:{exe.name}"
            f"  （env 已手工拼好 INCLUDE/LIB/PATH）")

    say("=" * 72)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    compile_ms = (time.perf_counter() - t0) * 1000

    out = (proc.stdout or "") + (proc.stderr or "")
    if out.strip():
        say(out.rstrip())                       # 编译器诊断 -> stderr
    if proc.returncode != 0:
        say(f"\n[编译失败] 退出码 {proc.returncode}（耗时 {compile_ms:.0f} ms）")
        return proc.returncode
    say(f"[编译成功] 耗时 {compile_ms:.0f} ms → {exe.name}")

    if args.no_run:
        return 0

    # ---------------- 运行 ----------------
    if args.manual:
        say("\n" + "-" * 72)
        say("[运行] 手动输入模式（Ctrl+Z 回车 结束输入）")
        say("-" * 72)
        proc = subprocess.run([str(exe)], env=env, timeout=args.timeout)   # stdout 直接继承终端
        say(f"\n[退出码] {proc.returncode}")
        return proc.returncode

    infile = args.infile
    if infile is None:
        cand = CODING_DIR / "tests" / f"{src.stem}_1.in"
        infile = str(cand) if cand.exists() else None

    say("-" * 72)
    say(f"[运行] 输入 <- {infile}" if infile else "[运行] 无输入文件（stdin 为空）")
    say("-" * 72)

    t0 = time.perf_counter()
    try:
        if infile:
            with open(infile, "rb") as fin:
                proc = subprocess.run([str(exe)], env=env, stdin=fin,
                                      capture_output=True, timeout=args.timeout)
        else:
            proc = subprocess.run([str(exe)], env=env, stdin=subprocess.DEVNULL,
                                  capture_output=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        say(f"[运行超时] 超过 {args.timeout:g} 秒 —— 大概率是死循环或复杂度过高")
        return 124
    run_ms = (time.perf_counter() - t0) * 1000

    # 被测程序的 stdout -> 原样透传到脚本 stdout（写原始字节，避免 Windows 上
    # Python 再把 \n 翻译成 \r\n，变成 \r\r\n）
    if proc.stdout:
        sys.stdout.buffer.write(proc.stdout)
        sys.stdout.buffer.flush()
    # 被测程序的 stderr -> 加前缀后走脚本 stderr
    if proc.stderr:
        for line in proc.stderr.decode("utf-8", errors="replace").rstrip().splitlines():
            say(f"[stderr] {line}")

    say("-" * 72)
    say(f"[运行结束] 退出码 {proc.returncode}   耗时 {run_ms:.1f} ms")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
