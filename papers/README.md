# 论文索引

优先级：**core = 必读**（与小米岗位强相关 / 实验直接依赖）；ext = 进阶选读。

建议阅读顺序：**00 → 01 → 02**（02 是岗位1「上下文增强大模型ASR」的命题核心）→ 04（部署前）→ 03（有余力）→ 06（面试补基础）。

## 00_survey

| # | 优先级 | arXiv | 文件名 | 用途 |
|---|---|---|---|---|
| 1 | **core** | [2111.01690](https://arxiv.org/abs/2111.01690) | `2111.01690_e2e-asr-survey.pdf` | 端到端 ASR 综述：CTC/AED/RNN-T 三大范式 |
| 2 | **core** | [2410.03751](https://arxiv.org/abs/2410.03751) | `2410.03751_speech-lm-survey.pdf` | Speech Language Model 综述(2024)：大模型语音最新地图 |
| 3 | ext | [1708.07524](https://arxiv.org/abs/1708.07524) | `1708.07524_supervised-separation-survey.pdf` | 监督式语音分离综述(王德亮组) |
| 4 | ext | [1905.00078](https://arxiv.org/abs/1905.00078) | `1905.00078_dl-audio-survey.pdf` | 音频深度学习综述：前端/增强/ASR 全景 |
| 5 | ext | [2006.05525](https://arxiv.org/abs/2006.05525) | `2006.05525_kd-survey.pdf` | 知识蒸馏综述 |

## 01_asr_architecture

| # | 优先级 | arXiv | 文件名 | 用途 |
|---|---|---|---|---|
| 6 | **core** | [2002.02562](https://arxiv.org/abs/2002.02562) | `2002.02562_transformer-transducer.pdf` | Transformer-Transducer：流式 RNN-T |
| 7 | **core** | [2005.08100](https://arxiv.org/abs/2005.08100) | `2005.08100_conformer.pdf` | Conformer：卷积+Transformer 主干 |
| 8 | **core** | [2203.15455](https://arxiv.org/abs/2203.15455) | `2203.15455_wenet2.pdf` | WeNet 2.0：生产级流式/非流式统一工具链 |
| 9 | **core** | [2212.04356](https://arxiv.org/abs/2212.04356) | `2212.04356_whisper.pdf` | Whisper：弱监督大规模 ASR 原文 |
| 10 | **core** | [2303.01037](https://arxiv.org/abs/2303.01037) | `2303.01037_google-usm.pdf` | Google USM：大模型多语种 ASR(岗位1「大模型ASR」) |
| 11 | ext | [2206.08317](https://arxiv.org/abs/2206.08317) | `2206.08317_paraformer.pdf` | Paraformer：非自回归并行 ASR |
| 12 | ext | [2207.02971](https://arxiv.org/abs/2207.02971) | `2207.02971_branchformer.pdf` | Branchformer：MLP+Attention 并行分支 |
| 13 | ext | [2210.00077](https://arxiv.org/abs/2210.00077) | `2210.00077_e-branchformer.pdf` | E-Branchformer |
| 14 | ext | [2310.11230](https://arxiv.org/abs/2310.11230) | `2310.11230_zipformer.pdf` | Zipformer：k2 高效编码器(板端目标架构) |

## 02_contextual_biasing

| # | 优先级 | arXiv | 文件名 | 用途 |
|---|---|---|---|---|
| 15 | **core** | [1808.02480](https://arxiv.org/abs/1808.02480) | `1808.02480_deep-context-clas.pdf` | Deep Context / CLAS 原版：上下文 LAS 奠基 |
| 16 | **core** | [2006.03411](https://arxiv.org/abs/2006.03411) | `2006.03411_contextual-rnnt.pdf` | Contextual RNN-T：开放域偏置经典 |
| 17 | **core** | [2306.00804](https://arxiv.org/abs/2306.00804) | `2306.00804_adaptive-contextual-biasing.pdf` | Adaptive Contextual Biasing：流式 Transducer 自适应偏置 |
| 18 | **core** | [2512.21828](https://arxiv.org/abs/2512.21828) | `2512.21828_llm-asr-contextual-biasing.pdf` | LLM-based ASR 上下文偏置+热词检索(2025 最新) |
| 19 | ext | [2109.00627](https://arxiv.org/abs/2109.00627) | `2109.00627_tcpgen.pdf` | TCPGen：树约束指针生成器 |
| 20 | ext | [2306.01942](https://arxiv.org/abs/2306.01942) | `2306.01942_cb-whisper-gpt2.pdf` | Whisper+GPT-2 上上下文偏置是否仍有效 |
| 21 | ext | [2409.17603](https://arxiv.org/abs/2409.17603) | `2409.17603_deep-clas.pdf` | Deep CLAS：深度上下文 LAS(2024) |

## 03_speech_enhancement

| # | 优先级 | arXiv | 文件名 | 用途 |
|---|---|---|---|---|
| 22 | **core** | [2107.05429](https://arxiv.org/abs/2107.05429) | `2107.05429_dpcrn.pdf` | DPCRN：双路 CRN 实时降噪 |
| 23 | **core** | [2403.17829](https://arxiv.org/abs/2403.17829) | `2403.17829_gtcrn.pdf` | GTCRN：移动端实时降噪(端侧部署首选) |
| 24 | ext | [1709.08243](https://arxiv.org/abs/1709.08243) | `1709.08243_rnnoise.pdf` | RNNoise：DSP+DL 混合实时降噪 |
| 25 | ext | [1809.07454](https://arxiv.org/abs/1809.07454) | `1809.07454_conv-tasnet.pdf` | Conv-TasNet：时域分离奠基 |
| 26 | ext | [2110.05588](https://arxiv.org/abs/2110.05588) | `2110.05588_deepfilternet.pdf` | DeepFilterNet：低复杂度全带降噪 |
| 27 | ext | [2501.05183](https://arxiv.org/abs/2501.05183) | `2501.05183_zipenhancer.pdf` | ZipEnhancer：Zipformer 做语音增强(2025) |

## 04_distill_quantization

| # | 优先级 | arXiv | 文件名 | 用途 |
|---|---|---|---|---|
| 28 | **core** | [1712.05877](https://arxiv.org/abs/1712.05877) | `1712.05877_int8-quantization.pdf` | INT8 量化奠基(部署必懂) |
| 29 | **core** | [1811.06621](https://arxiv.org/abs/1811.06621) | `1811.06621_streaming-mobile-asr.pdf` | 移动端流式 E2E ASR |
| 30 | **core** | [2011.06110](https://arxiv.org/abs/2011.06110) | `2011.06110_kd-rnnt.pdf` | RNN-T 高效知识蒸馏 |
| 31 | **core** | [2311.00430](https://arxiv.org/abs/2311.00430) | `2311.00430_distil-whisper.pdf` | Distil-Whisper：大规模伪标签蒸馏 |
| 32 | ext | [1902.08153](https://arxiv.org/abs/1902.08153) | `1902.08153_lsq.pdf` | LSQ：可学习量化步长 |
| 33 | ext | [2306.15171](https://arxiv.org/abs/2306.15171) | `2306.15171_streaming-distill.pdf` | 流式/非流式 Transducer 对齐蒸馏 |

## 05_dataset_systems

| # | 优先级 | arXiv | 文件名 | 用途 |
|---|---|---|---|---|
| 34 | **core** | [1709.05522](https://arxiv.org/abs/1709.05522) | `1709.05522_aishell1.pdf` | AISHELL-1：中文 178h 基线 |
| 35 | **core** | [2110.03370](https://arxiv.org/abs/2110.03370) | `2110.03370_wenetspeech.pdf` | WenetSpeech：中文万小时多域 |
| 36 | ext | [1503.06848](https://arxiv.org/abs/1503.06848) | `1503.06848_librispeech.pdf` | LibriSpeech：英文 960h 基线 |
| 37 | ext | [2106.06909](https://arxiv.org/abs/2106.06909) | `2106.06909_gigaspeech.pdf` | GigaSpeech：英文万小时 |

## 06_foundation

| # | 优先级 | arXiv | 文件名 | 用途 |
|---|---|---|---|---|
| 38 | **core** | [1412.6980](https://arxiv.org/abs/1412.6980) | `1412.6980_adam.pdf` | Adam 优化器 |
| 39 | **core** | [1706.03762](https://arxiv.org/abs/1706.03762) | `1706.03762_transformer.pdf` | Transformer 原文 |
| 40 | ext | [1409.0473](https://arxiv.org/abs/1409.0473) | `1409.0473_bahdanau-attention.pdf` | Bahdanau 注意力 |
| 41 | ext | [1607.06450](https://arxiv.org/abs/1607.06450) | `1607.06450_layer-norm.pdf` | Layer Normalization |
| 42 | ext | [1910.07467](https://arxiv.org/abs/1910.07467) | `1910.07467_rmsnorm.pdf` | RMSNorm(大模型标配) |

共 42 篇（计划 42 篇，缺 0 篇）。