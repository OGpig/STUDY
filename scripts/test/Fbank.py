#Fbank 的流程是：分帧 → 去直流 → 帧内预加重 → 加窗(Povey) → 零填充+FFT → 功率谱 → Mel滤波器组 → 取Log。
#   ⚠️ 注意：不是"先预加重再分帧"。Kaldi 是**先分帧**，之后所有逐帧操作都在帧内做。
import math

import torch

# Kaldi / torchaudio 的 log 下限 = std::numeric_limits<float>::epsilon()
EPS = torch.finfo(torch.float32).eps          # 1.1920929e-07


def my_fbank(waveform,
             sample_rate=16000, num_mel_bins=80,
             frame_length=25, frame_shift=10,
             preemphasis=0.97, dither=0.0,
             snip_edges=True, remove_dc_offset=True,
             round_to_power_of_two=True,
             low_freq=20.0, high_freq=0.0):
    """
    手写 Fbank 特征提取，严格对齐 torchaudio.compliance.kaldi.fbank

    输入 waveform: (num_samples,) 或 (1, num_samples)
    输出          : (num_frames, num_mel_bins)
    """
    # ==================== 0. 输入检查与预处理 ====================
    was_1d = waveform.dim() == 1
    if was_1d:
        waveform = waveform.unsqueeze(0)
    if waveform.size(0) > 1:                       # 多声道取均值
        waveform = waveform.mean(0, keepdim=True)
    waveform = waveform.float()

    frame_length_samples = int(sample_rate * frame_length / 1000)   # 25ms -> 400点
    frame_shift_samples = int(sample_rate * frame_shift / 1000)     # 10ms -> 160点
    assert 2 <= frame_length_samples <= waveform.size(1), \
        f"音频太短：{waveform.size(1)} 采样点 < 帧长 {frame_length_samples}"

    # 【修正2】FFT 点数：Kaldi 默认 round_to_power_of_two=True，400 要**零填充到 512**。
    #   用 400 会让低频的窄 mel 三角滤波器在 bin 索引上"无地可站"（见【修正1】）。
    padded_len = (1 << (frame_length_samples - 1).bit_length()) if round_to_power_of_two \
        else frame_length_samples
    assert padded_len % 2 == 0

    # ==================== 1. 分帧 (Framing) ====================
    # 【修正3】顺序：先分帧。后面的去直流/预加重/加窗全部在帧内做。
    #   原实现在整段波形上做预加重，等于用了跨帧的"上一帧尾样本"，与 Kaldi 不同。
    x = waveform
    if snip_edges:
        # 丢弃末尾不足一帧的部分
        num_frames = (x.size(1) - frame_length_samples) // frame_shift_samples + 1
        frames = x.unfold(1, frame_length_samples, frame_shift_samples)[:, :num_frames, :]
    else:
        # 帧数只由帧移决定，两端做 reflect 填充（Kaldi _get_strided 的行为）
        m = (x.size(1) + frame_shift_samples // 2) // frame_shift_samples
        pad = frame_length_samples // 2 - frame_shift_samples // 2
        rev = torch.flip(x, [1])
        x = torch.cat([rev[:, -pad:], x, rev], dim=1) if pad > 0 \
            else torch.cat([x[:, -pad:], rev], dim=1)
        frames = x.unfold(1, frame_length_samples, frame_shift_samples)[:, :m, :]

    # 抖动 (Dither)：对拍时必须为 0.0，否则随机噪声会让误差失去意义
    if dither != 0.0:
        frames = frames + dither * torch.randn_like(frames)

    # ==================== 2. 去直流 (Remove DC Offset) ====================
    # 【修正3】原实现缺这一步。每帧减该帧均值，去掉录音设备的直流偏置。
    #   本测试音频影响小，但换一支麦克风就会放大成主要误差。
    if remove_dc_offset:
        frames = frames - frames.mean(dim=2, keepdim=True)

    # ==================== 3. 预加重 (Pre-emphasis) ====================
    # 作用：提升高频能量，使频谱更平坦。帧内公式 y[j] = x[j] - α·x[j-1]
    # 【修正3】j=0 处 Kaldi 用 replicate 填充 → y[0] = x[0] - 0.97·x[0] = 0.03·x[0]
    #   （不是原写法的"直接保留 x[0]"）
    if preemphasis != 0.0:
        prev = torch.nn.functional.pad(frames, (1, 0), mode="replicate")[:, :, :-1]
        frames = frames - preemphasis * prev

    # ==================== 4. 加窗 (Windowing) ====================
    # Kaldi 默认 Povey 窗：w(n) = (0.5 - 0.5·cos(2πn/(N-1)))^0.85
    # torch.hann_window(periodic=False) 就是 (0.5 - 0.5·cos(...))，再 ^0.85 即 Povey 窗
    window = torch.hann_window(frame_length_samples, periodic=False).pow(0.85)
    frames = frames * window

    # ==================== 5. FFT & 功率谱 (Power Spectrum) ====================
    # 【修正2】零填充到 2 的幂，rfft 输出长度 = padded_len//2 + 1（512 -> 257）
    if padded_len != frame_length_samples:
        frames = torch.nn.functional.pad(frames, (0, padded_len - frame_length_samples))
    spectrum = torch.fft.rfft(frames).abs().pow(2.0)     # 功率谱 = 实²+虚²

    # ==================== 6. Mel 滤波器组 (Mel Filter Banks) ====================
    # 【修正1】核心修正：在 **mel 域** 算权重，而不是把 mel 端点 floor 成 bin 索引后
    #   在 **bin 索引域** 插值。索引域插值会让相邻端点量化到同一个 bin 上，
    #   低频三角滤波器直接塌陷成整行 0（原版有 5 个 mel bin 全零）。
    #   注意传入的是 padded_len（512），bin 宽度 = 16000/512 = 31.25 Hz，不是 40 Hz。
    mel_banks = _get_mel_banks(num_mel_bins, padded_len, sample_rate,
                               low_freq=low_freq, high_freq=high_freq)
    # banks 只有 padded_len//2 列，右侧补一列 0 对齐 Nyquist 分量
    mel_banks = torch.nn.functional.pad(mel_banks, (0, 1), mode="constant", value=0.0)

    # 矩阵乘法: [num_frames, freq_bins] x [freq_bins, num_mel_bins] -> [num_frames, num_mel_bins]
    mel_energy = torch.matmul(spectrum, mel_banks.t())

    # ==================== 7. 取对数 (Log) ====================
    # 地板值取 float32 eps（Kaldi 行为），防止 log(0) 出现 -inf
    # 注意 torch.max(Tensor, 标量) 不接受 Python float，所以用 clamp(min=...)
    out = torch.clamp(mel_energy, min=EPS).log()

    # 1D 输入返回 (num_frames, num_mel_bins)，2D 输入保留 batch 维
    return out.squeeze(0) if was_1d else out


def _get_mel_banks(num_mel_bins, padded_len, sample_rate, low_freq=20.0, high_freq=0.0):
    """
    构建 Kaldi 风格的 Mel 滤波器组矩阵（mel 域向量化版本）
    返回 (num_mel_bins, padded_len // 2)，与 torchaudio.compliance.kaldi.get_mel_banks 对齐。
    """
    num_fft_bins = padded_len // 2
    nyquist = sample_rate / 2.0
    if high_freq <= 0:
        high_freq = nyquist + high_freq      # <=0 表示相对于 Nyquist 的偏移

    # Kaldi 使用自然对数 ln（mel 刻度），不是 log10
    def hz_to_mel(f):
        return 1127.0 * math.log(1.0 + f / 700.0)

    mel_low = hz_to_mel(low_freq)
    mel_high = hz_to_mel(high_freq)

    # 在 Mel 刻度上均匀分 num_mel_bins+1 段；最左/最右滤波器只张开一半，所以要多铺一格
    mel_delta = (mel_high - mel_low) / (num_mel_bins + 1)

    k = torch.arange(num_mel_bins).unsqueeze(1).float()          # (num_bins, 1)
    left_mel = mel_low + k * mel_delta
    center_mel = mel_low + (k + 1.0) * mel_delta
    right_mel = mel_low + (k + 2.0) * mel_delta

    # 【修正1】把每个 FFT bin 的**中心频率**转成 mel 值（这才是 Kaldi 的做法）
    fft_bin_width = sample_rate / padded_len                    # 用 padded 长度！
    mel_of_bin = 1127.0 * torch.log(1.0 + fft_bin_width * torch.arange(num_fft_bins).float() / 700.0)
    mel_of_bin = mel_of_bin.unsqueeze(0)                        # (1, num_fft_bins)

    # 三角形的两条边：取 min(上升沿, 下降沿)，区间外的负值 clamp 到 0
    #   这样天然处理左右边界，不再需要原来那套 `if left < center` 的整型判断
    up_slope = (mel_of_bin - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel_of_bin) / (right_mel - center_mel)
    banks = torch.clamp(torch.minimum(up_slope, down_slope), min=0.0)

    return banks
