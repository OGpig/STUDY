# -*- coding: utf-8 -*-
"""
论文 PDF 工具 —— 读论文时的必备辅助（全文提取 / 关键词定位 / 单页查看）

用法:
    python paper.py extract <pdf>                  # 提取全文到 <pdf>.txt
    python paper.py grep <pdf> <kw1> [<kw2> ...]   # 多关键词检索, 打印页码 + 上下文
    python paper.py page <pdf> <n>                 # 打印第 n 页全文
    python paper.py toc <pdf>                      # 每页首 120 字, 快速摸结构

例:
    python paper.py grep papers/00_survey/2111.01690_e2e-asr-survey.pdf hybrid HMM conventional
"""
import os
import re
import sys

from pypdf import PdfReader


def load(path):
    reader = PdfReader(path)
    out = []
    for i, p in enumerate(reader.pages, 1):
        try:
            t = p.extract_text() or ""
        except Exception as e:
            t = f"[extract failed: {e}]"
        out.append((i, t))
    return out


def flat(s):
    return re.sub(r"\s+", " ", s)


def cmd_extract(path):
    pgs = load(path)
    dst = os.path.splitext(path)[0] + ".txt"
    with open(dst, "w", encoding="utf-8") as f:
        for i, t in pgs:
            f.write(f"\n===== page {i} =====\n{t}\n")
    print(f"{len(pgs)} pages -> {dst}")


def cmd_grep(path, kws):
    pgs = load(path)
    for kw in kws:
        print(f"\n{'='*70}\n### 关键词: {kw}\n{'='*70}")
        total = 0
        for i, t in pgs:
            for m in re.finditer(re.escape(kw), t, re.I):
                total += 1
                a = max(0, m.start() - 90)
                b = min(len(t), m.end() + 130)
                print(f"[p{i:>2}] ...{flat(t[a:b])}...")
                if total >= 14:
                    print(f"  (更多命中已省略)")
                    break
            if total >= 14:
                break
        print(f"  小计: {total}{'+' if total >= 14 else ''} 处")


def cmd_page(path, n):
    for i, t in load(path):
        if i == n:
            print(f"===== page {n} =====\n{t}")
            return
    print(f"page {n} not found")


def cmd_toc(path):
    for i, t in load(path):
        print(f"[p{i:>2}] {flat(t)[:120]}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    cmd, pdf = sys.argv[1], sys.argv[2]
    if cmd == "extract":
        cmd_extract(pdf)
    elif cmd == "grep":
        cmd_grep(pdf, sys.argv[3:])
    elif cmd == "page":
        cmd_page(pdf, int(sys.argv[3]))
    elif cmd == "toc":
        cmd_toc(pdf)
    else:
        print(__doc__)
