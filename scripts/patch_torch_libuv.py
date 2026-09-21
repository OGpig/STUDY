#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
给 PyTorch 打一个 Windows 必需的补丁：TCPStore(..., use_libuv=False)

【问题】
    torch 2.11.0+cu128 的 Windows 轮子在创建 TCPStore 时有**三处都漏传 use_libuv**
    （c10d_rendezvous_backend / static_tcp_rendezvous / rendezvous），
    而 C++ 侧 TCPStore 的默认值是 **True**（libuv）——
    可这个轮子是在**没有 libuv 支持**的情况下编译的。于是任何分布式初始化都直接失败：

        torch.distributed.DistStoreError: use_libuv was requested but PyTorch
        was built without libuv support, run with USE_LIBUV=0 to disable it.

【为什么报了错还是修不了】
    报错信息让你设 USE_LIBUV=0。**但实测这个环境变量完全无效** ——
    C++ 层的 TCPStore 根本不读它，只有 `rendezvous.py::_get_use_libuv_from_query_dict`
    （只覆盖 `init_method="env://"` 且只在 URL 里带参数时）会看它一眼，
    而 elastic 的 TCPStore 创建路径压根不经过那里。
    实测（本机 torch 2.11.0+cu128）：
        USE_LIBUV 未设 → 默认参数建 TCPStore ❌
        USE_LIBUV=0    → 默认参数建 TCPStore ❌   ← 设了也没用
        显式 use_libuv=False           → ✅
    所以**唯一的解法就是给那三处调用显式补上参数**。

【这个脚本做什么】
    在 torch 的 3 个文件里，给每个 `TCPStore(...)` 调用的右括号前插入 `use_libuv=False`。
    做法是**括号配对扫描**，不依赖具体换行/缩进，也不改动文件其它任何字节。
    首次修改前会把原文件备份成 `xxx.py.orig`。

【用法】
    python scripts/patch_torch_libuv.py --check      # 只看状态（不写）
    python scripts/patch_torch_libuv.py             # 打补丁
    python scripts/patch_torch_libuv.py --revert    # 从 .orig 还原

    # 换机器/重建 venv 后重新执行一次即可；已打过会报"无需修改"
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / ".venv" / "Lib" / "site-packages"

TARGETS = [
    # 分布式 rendezvous 的三条路径（c10d / static / dynamic）—— 训练必经
    "torch/distributed/rendezvous.py",
    "torch/distributed/elastic/rendezvous/c10d_rendezvous_backend.py",
    "torch/distributed/elastic/rendezvous/static_tcp_rendezvous.py",
    "torch/distributed/elastic/rendezvous/dynamic_rendezvous.py",
    # 其余会创建 TCPStore 的路径
    "torch/distributed/elastic/utils/distributed.py",
    "torch/distributed/checkpoint/_async_process_executor.py",
    "torch/distributed/checkpoint/_experimental/barriers.py",
    "torch/distributed/debug/_store.py",
]

NEEDLE = "TCPStore("
INSERT = "use_libuv=False, "


def say(*a, **kw) -> None:
    print(*a, file=sys.stderr, **kw)


def docstring_line_ranges(src: str) -> list[tuple[int, int]]:
    """
    返回文件里所有**字符串字面量**的行号区间。

    为什么要这个：有些 docstring 里带示例代码 —— 例如
    `torch/distributed/elastic/rendezvous/__init__.py` 的模块 docstring 里就有
    `store = TCPStore("localhost")`。往那里插参数会污染文档，运行期毫无作用，必须跳过。
    """
    import ast

    ranges: list[tuple[int, int]] = []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ranges
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.end_lineno:
            ranges.append((node.lineno, node.end_lineno))
    return ranges


def find_calls(src: str, skip_lines: list[tuple[int, int]] | None = None) -> list[int]:
    """
    返回每个 `TCPStore(...)` 调用中**右括号**的下标。

    用括号配对扫描定位（不依赖换行/缩进），并跳过两类：
      · 函数定义 `def TCPStore(`
      · 落在字符串字面量（docstring 示例）里的代码
    """
    out: list[int] = []
    skip_lines = skip_lines or []
    i = 0
    while True:
        j = src.find(NEEDLE, i)
        if j < 0:
            break
        line = src.count("\n", 0, j) + 1
        if any(a <= line <= b for a, b in skip_lines):
            i = j + len(NEEDLE)
            continue
        if src[max(0, j - 4):j] == "def ":
            i = j + len(NEEDLE)
            continue

        open_paren = j + len(NEEDLE) - 1
        depth = 0
        p = open_paren
        in_str: str | None = None
        while p < len(src):
            c = src[p]
            if in_str:
                if c == in_str and src[p - 1] != "\\":
                    in_str = None
            elif c in "\"'":
                in_str = c
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
            p += 1
        if p >= len(src):
            raise RuntimeError("括号配对失败，文件可能不是预期版本")
        out.append(p)
        i = p + 1
    return out


def patch_file(path: Path, apply: bool) -> tuple[int, int]:
    """返回 (找到的调用数, 已带 use_libuv 的调用数)。"""
    src = path.read_text(encoding="utf-8")
    closes = find_calls(src, skip_lines=docstring_line_ranges(src))
    already = 0
    todo: list[int] = []
    for c in closes:
        # 往前找该调用的起点，检查参数里是否已有 use_libuv
        start = src.rfind(NEEDLE, 0, c)
        if "use_libuv" in src[start:c]:
            already += 1
        else:
            todo.append(c)

    if apply and todo:
        backup = path.with_suffix(path.suffix + ".orig")
        if not backup.exists():
            shutil.copy2(path, backup)
        # 从后往前插，保证前面的下标不失效
        buf = src
        for c in sorted(todo, reverse=True):
            buf = buf[:c] + INSERT + buf[c:]
        path.write_text(buf, encoding="utf-8", newline="")
    return len(closes), already


def main() -> int:
    ap = argparse.ArgumentParser(description="给 PyTorch 补 use_libuv=False（Windows 必需）")
    ap.add_argument("--check", action="store_true", help="只检查，不修改")
    ap.add_argument("--revert", action="store_true", help="从 .orig 还原")
    args = ap.parse_args()

    if not SITE.is_dir():
        say(f"❌ 找不到 site-packages: {SITE}")
        return 1

    total_calls = total_already = 0
    missing = []

    for rel in TARGETS:
        p = SITE / rel
        if not p.exists():
            missing.append(rel)
            say(f"⚠️ 文件不存在（torch 版本可能变了）: {rel}")
            continue

        if args.revert:
            backup = p.with_suffix(p.suffix + ".orig")
            if backup.exists():
                shutil.copy2(backup, p)
                say(f"↩️  已还原 {rel}")
            else:
                say(f"（无备份，跳过）{rel}")
            continue

        calls, already = patch_file(p, apply=not args.check)
        total_calls += calls
        total_already += already
        state = "已打补丁" if already == calls and calls else "**待补丁**"
        say(f"{rel}")
        say(f"    TCPStore 调用 {calls} 处，其中已带 use_libuv 的 {already} 处 → {state}")

    if args.revert:
        return 0

    say("")
    if missing:
        say(f"⚠️ 有 {len(missing)} 个目标文件缺失，补丁可能不完整")
    if total_calls == 0:
        say("❌ 一处 TCPStore 调用都没找到 —— torch 结构变了，请手动检查")
        return 1
    if total_already == total_calls:
        say(f"✅ 已全部打好（{total_calls} 处），无需修改")
        return 0
    if args.check:
        say(f"⚠️ 还有 {total_calls - total_already} 处未打补丁 —— 去掉 --check 执行")
        return 1
    say(f"✅ 补丁完成：新增 {total_calls - total_already} 处，共 {total_calls} 处")
    say("   备份在各自同目录的 *.py.orig（可用 --revert 还原）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
