# -*- coding: utf-8 -*-
"""只保留 download_papers.py 定义的标准文件名，删除重复/旧命名的 PDF"""
import os, re, sys
BASE = r"F:/embedded/prepare/papers"

keep = set()
src = open("F:/embedded/prepare/scripts/download_papers.py", encoding="utf-8").read()
for m in re.finditer(r'\("(\d{2}_\w+)","(core|ext)","([\d.]+)","([\w-]+)"', src):
    keep.add(f"{m.group(3)}_{m.group(4)}.pdf")
print("白名单", len(keep), "个")

remove = []
for root, _, files in os.walk(BASE):
    for f in files:
        if f.lower().endswith(".pdf") and f not in keep:
            remove.append(os.path.join(root, f))

for p in remove:
    print("  删除:", os.path.relpath(p, BASE))
if "--do" in sys.argv:
    for p in remove:
        os.remove(p)
    print(f"\n已删除 {len(remove)} 个重复文件")
else:
    print(f"\n待删除 {len(remove)} 个（加 --do 执行）")
