# -*- coding: utf-8 -*-
"""
Fbank 对拍诊断：手写实现 vs torchaudio.compliance.kaldi.fbank（Kaldi 的官方 Python 移植）

两部分：
  A) 单独对比 mel 滤波器组矩阵 —— 隔离出「滤波器设计」这一项
  B) 端到端对比 + 参数消融 —— 用 torchaudio 的开关逐个隔离差异来源
     （round_to_power_of_two 控制 FFT 点数；remove_dc_offset 控制是否去直流）

用法:
    set PYTHONIOENCODING=utf-8
    F:\\embedded\\prepare\\.venv\\Scripts\\python.exe scripts\\test\\compare_fbank.py
"""
import math
import os
import sys

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Fbank import my_fbank, _get_mel_banks

SR = 16000
N_MELS = 80
WAV_CANDIDATES = [
    r"F:\embedded\rknn_model_zoo-main\examples\whisper\model\test_zh.wav",
    r"F:\embedded\rknn_model_zoo-main\examples\zipformer\model\test.wav",
]


def load_wav(path, target_sr=SR):
    """torchaudio>=2.9 的 load() 需要 torchcodec，这里改用 soundfile 读（已装）"""
    import numpy as np
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32")
    wav = torch.from_numpy(np.asarray(data))
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    else:
        wav = wav.transpose(0, 1)          # (n, c) -> (c, n)
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def stats(name, mine, ref):
    if mine.shape != ref.shape:
        print(f"  {name:<32} 形状不同  mine{tuple(mine.shape)} vs ref{tuple(ref.shape)}")
        return None
    d = (mine - ref).abs()
    rel = (d / ref.abs().clamp(min=1e-6)).mean().item() * 100
    print(f"  {name:<32} max={d.max():7.4f}  mean={d.mean():7.4f}  rel={rel:6.2f}%")
    return d


# ---------------------------------------------------------------- A) mel banks
def cmp_mel_banks():
    print("=" * 82)
    print("A) mel 滤波器组矩阵对比（window_length = 400）")
    print("=" * 82)
    mine = _get_mel_banks(N_MELS, 400, SR, low_freq=20, high_freq=0)
    ref, center = kaldi.get_mel_banks(N_MELS, 400, SR, 20.0, 0.0, 100.0, -500.0, 1.0)
    print(f"  手写形状      : {tuple(mine.shape)}   (应 = window/2 = 200)")
    print(f"  torchaudio    : {tuple(ref.shape)}   (= window/2 = 200)")
    print(f"  两只形状一致  : {mine.shape == ref.shape}")

    m = mine[:, : ref.size(1)]
    d = (m - ref).abs()
    print(f"\n  对齐到 200 列后: max abs diff = {d.max():.6f}   mean = {d.mean():.6f}")
    print(f"  完全相同(±1e-6)的元素占比: {float((d < 1e-6).float().mean())*100:.1f}%")

    per_bin = d.max(dim=1).values
    worst = per_bin.topk(5)
    print("  差异最大的 5 个 mel bin（索引, 该行 max diff）:")
    for i, v in zip(worst.indices.tolist(), worst.values.tolist()):
        print(f"     bin {i:>2} : {v:.6f}")

    # 检测「量化塌陷」—— 手写版把 mel 端点 floor 到 bin 索引后，窄三角会塌缩成空行
    zero_rows = [i for i in range(N_MELS) if float(mine[i].abs().sum()) == 0]
    print(f"\n  ⚠ 手写版权值和为 0（整行塌陷）的 mel bin: {len(zero_rows)} 个")
    print(f"     -> {zero_rows[:15]}{' ...' if len(zero_rows) > 15 else ''}")
    if zero_rows:
        i = zero_rows[0]
        print(f"     例 bin {i}: 手写行和={float(mine[i].sum()):.4f}   "
              f"torchaudio 行和={float(ref[i].sum()):.4f}")
    ratio = float(ref.sum(1).clamp(min=1e-9).div(mine.sum(1).clamp(min=1e-9)).max())
    print(f"     最大「行和比值」= {ratio:.3f}  (手写版为空行时分母近似 0，比值会很大)")

    # 手写版的 bin 边界 vs torchaudio 的三角中心
    import math
    mel = lambda f: 1127.0 * math.log(1.0 + f / 700.0)
    low, high = mel(20.0), mel(SR / 2)
    delta = (high - low) / (N_MELS + 1)
    pts = [700.0 * (math.exp((low + i * delta) / 1127.0) - 1.0) for i in range(3)]
    print(f"\n  提示: Kaldi 在 **mel 域** 均匀取三角端点, 前 3 个端点频率 = "
          f"{[round(p,1) for p in pts]} Hz")
    print(f"        fft_bin_width = {SR}/400 = {SR/400:.2f} Hz  (torchaudio 用 padded=512 时为 {SR/512:.2f} Hz)")


# ------------------------------------------------------------ B) 端到端对比
def cmp_end_to_end(wav, tag):
    print()
    print("=" * 82)
    print(f"B) 端到端对比  ——  音频: {tag}")
    print(f"   {wav.size(1)} 采样点 = {wav.size(1)/SR:.2f}s   (snip_edges=True)")
    print("=" * 82)
    mine = my_fbank(wav.squeeze(0), sample_rate=SR, num_mel_bins=N_MELS)
    print(f"  手写输出: shape={tuple(mine.shape)}  "
          f"range=[{mine.min():.3f}, {mine.max():.3f}]  "
          f"含-inf/NaN: {bool(torch.isinf(mine).any() or torch.isnan(mine).any())}")
    print()

    cfgs = [
        ("① torchaudio 默认(512fft+去DC)", dict()),
        ("② 512fft, 不去DC",              dict(remove_dc_offset=False)),
        ("③ 400fft, 去DC",                dict(round_to_power_of_two=False)),
        ("④ 400fft, 不去DC",              dict(round_to_power_of_two=False,
                                               remove_dc_offset=False)),
    ]
    refs = {}
    for name, kw in cfgs:
        ref = kaldi.fbank(wav, num_mel_bins=N_MELS, sample_frequency=SR,
                          dither=0.0, energy_floor=0.0, use_energy=False, **kw)
        refs[name] = ref
        stats(name, mine, ref)

    print()
    print("  参考实现自身之间的差异（说明这两项各占多少）:")
    stats("  ② 400fft+去DC  vs ① 默认",
          refs["③ 400fft, 去DC"], refs["① torchaudio 默认(512fft+去DC)"])
    print("     ↑ 这是「FFT 点数」单项的贡献（去DC 固定为 True）")

    # 在手写实现最接近的配置下, 看误差落在哪些 mel bin
    best = min(cfgs, key=lambda c: (mine - refs[c[0]]).abs().mean().item())[0]
    d = (mine - refs[best]).abs()
    per_bin = d.mean(dim=0)
    print(f"\n  最接近手写实现的配置: {best}")
    print(f"  逐 mel bin 平均误差 —— 最小的 5 个 bin: "
          f"{per_bin.topk(5, largest=False).indices.flatten().tolist()}")
    print(f"  逐 mel bin 平均误差 —— 最大的 5 个 bin: "
          f"{per_bin.topk(5).indices.flatten().tolist()}")
    print(f"  (低 mel bin 误差大 → 说明直流分量在做怪；"
          f"高 mel bin 误差大 → 说明滤波器/FFT 分辨率在做怪)")


def _variant(wav1d, fft_n, banks, remove_dc=True, log_floor=1e-10):
    """复刻手写实现的流程，但允许替换单个组件（用于误差归因）"""
    x = wav1d.unsqueeze(0).float()
    x = torch.cat([x[:, :1], x[:, 1:] - 0.97 * x[:, :-1]], dim=1)   # 整段预加重（同手写版）
    fl, fs = 400, 160
    nf = (x.size(1) - fl) // fs + 1
    frames = x.unfold(1, fl, fs)[:, :nf, :]
    if remove_dc:
        frames = frames - frames.mean(dim=2, keepdim=True)
    win = torch.hann_window(fl, periodic=False).pow(0.85)            # Povey 窗
    frames = frames * win
    if fft_n != fl:
        frames = torch.nn.functional.pad(frames, (0, fft_n - fl))   # 零填充到 fft_n
    spec = torch.fft.rfft(frames, n=fft_n).abs().pow(2)             # 功率谱
    b = torch.nn.functional.pad(banks[:, : fft_n // 2], (0, 1))     # 补 Nyquist 列=0
    return torch.clamp(spec @ b.T, min=log_floor).log().squeeze(0)


def cmp_ablation(wav, tag):
    print()
    print("=" * 82)
    print(f"C) 误差归因消融 —— 音频: {tag}")
    print("   基准 = torchaudio 默认(512fft + 去DC + Kaldi mel)")
    print("=" * 82)
    ref = kaldi.fbank(wav, num_mel_bins=N_MELS, sample_frequency=SR,
                      dither=0.0, energy_floor=0.0, use_energy=False)

    my400 = _get_mel_banks(N_MELS, 400, SR, 20, 0)
    kd400, _ = kaldi.get_mel_banks(N_MELS, 400, SR, 20.0, 0.0, 100.0, -500.0, 1.0)
    my512 = _get_mel_banks(N_MELS, 512, SR, 20, 0)
    kd512, _ = kaldi.get_mel_banks(N_MELS, 512, SR, 20.0, 0.0, 100.0, -500.0, 1.0)

    print(f"  {'配置':<26}{'max':>9}{'mean':>9}{'rel':>9}")
    print("  " + "-" * 51)
    combos = [
        ("手写 mel + 400fft", my400, 400),
        ("Kaldi mel + 400fft", kd400, 400),
        ("手写 mel + 512fft", my512, 512),
        ("Kaldi mel + 512fft", kd512, 512),
    ]
    for name, banks, n in combos:
        out = _variant(wav.squeeze(0), n, banks, remove_dc=True)
        d = (out - ref).abs()
        rel = (d / ref.abs().clamp(min=1e-6)).mean().item() * 100
        print(f"  {name:<26}{d.max():9.4f}{d.mean():9.4f}{rel:8.2f}%")
    print("  " + "-" * 51)
    print("  读法：固定 fft 比「手写 mel / Kaldi mel」，差值 = mel 滤波器设计的贡献；")
    print("        固定 mel 比 400/512，差值 = FFT 点数的贡献。")


def cmp_sweep():
    """D) 参数 / 音频边界扫描 —— 全部要求 bit-exact"""
    print()
    print("=" * 82)
    print("D) 参数与音频边界扫描（验收标准：torch.equal == True）")
    print("=" * 82)

    files = [p for p in WAV_CANDIDATES if os.path.exists(p)]
    cases = [
        dict(num_mel_bins=80),
        dict(num_mel_bins=40),
        dict(num_mel_bins=23),
        dict(num_mel_bins=80, snip_edges=False),
        dict(num_mel_bins=80, remove_dc_offset=False),
        dict(num_mel_bins=80, round_to_power_of_two=False),
        dict(num_mel_bins=80, frame_length=20, frame_shift=5),
        dict(num_mel_bins=80, low_freq=0.0),
        dict(num_mel_bins=80, low_freq=40.0, high_freq=7600.0),
        dict(num_mel_bins=64, sample_rate=8000),
    ]

    bad, total = 0, 0
    for fp in files:
        w, sr = _read_wav_raw(fp)
        for kw in cases:
            c = dict(kw)
            sr_ = c.pop("sample_rate", sr)
            wav = torchaudio.functional.resample(w, sr, sr_) if sr_ != sr else w
            total += 1
            try:
                mine = my_fbank(wav.squeeze(0), sample_rate=sr_, **c)
                ref = kaldi.fbank(wav, sample_frequency=sr_, dither=0.0, energy_floor=0.0,
                                  use_energy=False, **c)
                ok = (mine.shape == ref.shape) and torch.equal(mine, ref)
                d = (mine - ref).abs().max().item() if mine.shape == ref.shape else float("nan")
            except Exception as e:                                  # noqa: BLE001
                ok, d = False, float("nan")
                print(f"  {os.path.basename(fp)[:10]:<11}{str(c)[:48]:<50}"
                      f"EXC {type(e).__name__}: {e}")
                bad += 1
                continue
            print(f"  {os.path.basename(fp)[:10]:<11}{str(c)[:48]:<50}"
                  f"{str(tuple(mine.shape)):<14}{'PASS' if ok else 'FAIL':<6}{d:.2e}")
            bad += 0 if ok else 1

    print("  " + "-" * 78)
    print(f"  结果: {total - bad}/{total} 用例 bit-exact"
          f"{'  ✅ 全部通过' if bad == 0 else '  ❌ 存在未通过用例'}")


def _read_wav_raw(path):
    """读原始 wav 并统一成 (1, N) float32 —— 不重采样"""
    import numpy as np
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32")
    wav = torch.from_numpy(np.asarray(data))
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    else:
        wav = wav.transpose(0, 1)
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    return wav, sr


def main():
    print(f"torchaudio {torchaudio.__version__} / torch {torch.__version__}")
    cmp_mel_banks()

    found = [p for p in WAV_CANDIDATES if os.path.exists(p)]
    if found:
        wav, tag = load_wav(found[0]), os.path.basename(found[0])
    else:
        print("\n未找到测试音频，改用合成信号")
        torch.manual_seed(0)
        t = torch.arange(SR * 2) / SR
        wav = (0.5 * torch.sin(2 * math.pi * 440 * t)
               + 0.3 * torch.sin(2 * math.pi * 1500 * t)
               + 0.1 * torch.randn_like(t) + 0.15).unsqueeze(0)
        tag = "合成(440+1500Hz+噪声+直流0.15)"

    cmp_ablation(wav, tag)
    cmp_end_to_end(wav, tag)
    cmp_sweep()


if __name__ == "__main__":
    main()
