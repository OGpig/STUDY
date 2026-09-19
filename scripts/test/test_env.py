# -*- coding: utf-8 -*-
"""
环境自检 —— 依赖版本 + GPU 真实算力 + venv 独立性

以后任何时候想知道「我这个环境到底行不行」，跑这一条：
    F:\\embedded\\prepare\\.venv\\Scripts\\python.exe F:\\embedded\\prepare\\scripts\\test\\test_env.py

判读要点见 docs/ENV.md 第二节。
"""
import sys
import importlib.metadata as md

PKGS = ["torch", "torchaudio", "numpy", "pandas", "scipy",
        "soundfile", "librosa", "numba", "tensorboard", "scikit-learn"]


def check_python():
    print("=" * 64)
    print("Python     :", sys.version.split()[0])
    print("executable :", sys.executable)
    print("base_prefix:", sys.base_prefix)
    independent = "anaconda" not in sys.base_prefix.lower()
    print("venv 独立性:", "OK  (与 conda 脱钩)" if independent
          else "FAIL (仍绑定 anaconda —— 见 docs/ENV.md 风险 1)")
    return independent


def check_pkgs():
    print("-" * 64)
    missing = []
    for p in PKGS:
        try:
            print(f"  {p:<14} {md.version(p)}")
        except Exception:
            print(f"  {p:<14} MISSING")
            missing.append(p)
    return missing


def check_gpu():
    print("-" * 64)
    try:
        import torch
    except Exception as e:
        print("import torch 失败:", e)
        return False
    print("  torch      :", torch.__version__)
    if not torch.cuda.is_available():
        print("  cuda avail : False  <<< 需要检查驱动/轮子")
        return False
    print("  cuda avail : True")
    print("  device name:", torch.cuda.get_device_name(0))
    cap = torch.cuda.get_device_capability(0)
    print("  capability :", cap, "(期望 (12, 0) —— 否则落在兼容路径)")
    print("  cuda ver   :", torch.version.cuda)
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        a = torch.randn(1024, 1024, device="cuda", dtype=dt)
        b = torch.randn(1024, 1024, device="cuda", dtype=dt)
        print(f"    matmul {str(dt):16s} ok   sum={float((a @ b).sum()):.1f}")
    torch.cuda.synchronize()
    print("  synchronize: ok   mem=%.1f MB" % (torch.cuda.memory_allocated() / 1024 ** 2))
    return True


def main():
    independent = check_python()
    missing = check_pkgs()
    gpu_ok = check_gpu()
    print("=" * 64)
    if independent and not missing and gpu_ok:
        print("结论：环境就绪 ✅")
        return 0
    if missing:
        print("结论：缺少依赖 ->", ", ".join(missing))
    if not independent:
        print("结论：venv 仍绑在 conda 上，建议按 docs/ENV.md 风险 1 重建")
    return 1


if __name__ == "__main__":
    sys.exit(main())
