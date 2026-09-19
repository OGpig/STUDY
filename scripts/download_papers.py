# -*- coding: utf-8 -*-
"""
用 curl 直接按 arXiv ID 下载 PDF（绕过 arXiv API 限流），生成 papers/README.md 索引。
已存在且 >20KB 的文件自动跳过，可重复运行直到补全。
  python download_papers.py
"""
import os, re, subprocess, time, json

BASE = r"F:/embedded/prepare/papers"
CURL = "curl"

# cat, prio, id, slug(英文短名), 中文用途备注
P = [
 ("00_survey","core","2111.01690","e2e-asr-survey","端到端 ASR 综述：CTC/AED/RNN-T 三大范式"),
 ("00_survey","core","2410.03751","speech-lm-survey","Speech Language Model 综述(2024)：大模型语音最新地图"),
 ("00_survey","ext","2006.05525","kd-survey","知识蒸馏综述"),
 ("00_survey","ext","1708.07524","supervised-separation-survey","监督式语音分离综述(王德亮组)"),
 ("00_survey","ext","1905.00078","dl-audio-survey","音频深度学习综述：前端/增强/ASR 全景"),

 ("01_asr_architecture","core","2005.08100","conformer","Conformer：卷积+Transformer 主干"),
 ("01_asr_architecture","core","2203.15455","wenet2","WeNet 2.0：生产级流式/非流式统一工具链"),
 ("01_asr_architecture","core","2002.02562","transformer-transducer","Transformer-Transducer：流式 RNN-T"),
 ("01_asr_architecture","ext","2310.11230","zipformer","Zipformer：k2 高效编码器(板端目标架构)"),
 ("01_asr_architecture","ext","2206.08317","paraformer","Paraformer：非自回归并行 ASR"),
 ("01_asr_architecture","ext","2207.02971","branchformer","Branchformer：MLP+Attention 并行分支"),
 ("01_asr_architecture","ext","2210.00077","e-branchformer","E-Branchformer"),
 ("01_asr_architecture","core","2212.04356","whisper","Whisper：弱监督大规模 ASR 原文"),
 ("01_asr_architecture","core","2303.01037","google-usm","Google USM：大模型多语种 ASR(岗位1「大模型ASR」)"),

 ("02_contextual_biasing","core","1808.02480","deep-context-clas","Deep Context / CLAS 原版：上下文 LAS 奠基"),
 ("02_contextual_biasing","ext","2409.17603","deep-clas","Deep CLAS：深度上下文 LAS(2024)"),
 ("02_contextual_biasing","core","2306.00804","adaptive-contextual-biasing","Adaptive Contextual Biasing：流式 Transducer 自适应偏置"),
 ("02_contextual_biasing","core","2006.03411","contextual-rnnt","Contextual RNN-T：开放域偏置经典"),
 ("02_contextual_biasing","ext","2109.00627","tcpgen","TCPGen：树约束指针生成器"),
 ("02_contextual_biasing","core","2512.21828","llm-asr-contextual-biasing","LLM-based ASR 上下文偏置+热词检索(2025 最新)"),
 ("02_contextual_biasing","ext","2306.01942","cb-whisper-gpt2","Whisper+GPT-2 上上下文偏置是否仍有效"),

 ("03_speech_enhancement","ext","1809.07454","conv-tasnet","Conv-TasNet：时域分离奠基"),
 ("03_speech_enhancement","core","2107.05429","dpcrn","DPCRN：双路 CRN 实时降噪"),
 ("03_speech_enhancement","ext","2110.05588","deepfilternet","DeepFilterNet：低复杂度全带降噪"),
 ("03_speech_enhancement","core","2403.17829","gtcrn","GTCRN：移动端实时降噪(端侧部署首选)"),
 ("03_speech_enhancement","ext","1709.08243","rnnoise","RNNoise：DSP+DL 混合实时降噪"),
 ("03_speech_enhancement","ext","2501.05183","zipenhancer","ZipEnhancer：Zipformer 做语音增强(2025)"),

 ("04_distill_quantization","core","2311.00430","distil-whisper","Distil-Whisper：大规模伪标签蒸馏"),
 ("04_distill_quantization","core","1712.05877","int8-quantization","INT8 量化奠基(部署必懂)"),
 ("04_distill_quantization","ext","1902.08153","lsq","LSQ：可学习量化步长"),
 ("04_distill_quantization","core","2011.06110","kd-rnnt","RNN-T 高效知识蒸馏"),
 ("04_distill_quantization","ext","2306.15171","streaming-distill","流式/非流式 Transducer 对齐蒸馏"),
 ("04_distill_quantization","core","1811.06621","streaming-mobile-asr","移动端流式 E2E ASR"),

 ("05_dataset_systems","core","2110.03370","wenetspeech","WenetSpeech：中文万小时多域"),
 ("05_dataset_systems","core","1709.05522","aishell1","AISHELL-1：中文 178h 基线"),
 ("05_dataset_systems","ext","1503.06848","librispeech","LibriSpeech：英文 960h 基线"),
 ("05_dataset_systems","ext","2106.06909","gigaspeech","GigaSpeech：英文万小时"),

 ("06_foundation","core","1706.03762","transformer","Transformer 原文"),
 ("06_foundation","ext","1607.06450","layer-norm","Layer Normalization"),
 ("06_foundation","ext","1910.07467","rmsnorm","RMSNorm(大模型标配)"),
 ("06_foundation","core","1412.6980","adam","Adam 优化器"),
 ("06_foundation","ext","1409.0473","bahdanau-attention","Bahdanau 注意力"),
]

CATS = ["00_survey","01_asr_architecture","02_contextual_biasing","03_speech_enhancement",
        "04_distill_quantization","05_dataset_systems","06_foundation"]


def fetch(aid, dst):
    for k in range(4):
        r = subprocess.run([CURL, "-sL", "--retry", "2", "--retry-delay", "5",
                            "-A", "Mozilla/5.0", "-o", dst,
                            "--max-time", "180",
                            f"https://arxiv.org/pdf/{aid}"],
                           capture_output=True, text=True, errors="ignore")
        if os.path.exists(dst) and os.path.getsize(dst) > 20000:
            return os.path.getsize(dst)
        time.sleep(6 * (k + 1))
    return 0


def main():
    ok, fail = [], []
    for cat, prio, aid, slug, note in P:
        d = os.path.join(BASE, cat)
        os.makedirs(d, exist_ok=True)
        fn = f"{aid}_{slug}.pdf"
        dst = os.path.join(d, fn)
        if os.path.exists(dst) and os.path.getsize(dst) > 20000:
            ok.append((cat, prio, aid, slug, note)); continue
        sz = fetch(aid, dst)
        if sz:
            ok.append((cat, prio, aid, slug, note))
            print(f"[OK  ] {aid:<12} {sz//1024:>5} KB  {slug}")
        else:
            if os.path.exists(dst):
                os.remove(dst)
            fail.append((cat, aid))
            print(f"[FAIL] {aid:<12} {slug}")
        time.sleep(2.5)

    L = ["# 论文索引", "",
         "优先级：**core = 必读**（与小米岗位强相关 / 实验直接依赖）；ext = 进阶选读。",
         "", "建议阅读顺序：**00 → 01 → 02**（02 是岗位1「上下文增强大模型ASR」的命题核心）→ 04（部署前）→ 03（有余力）→ 06（面试补基础）。", ""]
    n = 0
    for c in CATS:
        grp = [x for x in ok if x[0] == c]
        if not grp:
            continue
        L += [f"## {c}", "", "| # | 优先级 | arXiv | 文件名 | 用途 |", "|---|---|---|---|---|"]
        for cat, prio, aid, slug, note in sorted(grp, key=lambda x: (x[1] != "core", x[2])):
            n += 1
            L.append(f"| {n} | {'**core**' if prio=='core' else 'ext'} | "
                     f"[{aid}](https://arxiv.org/abs/{aid}) | `{aid}_{slug}.pdf` | {note} |")
        L.append("")
    L.append(f"共 {n} 篇（计划 {len(P)} 篇，缺 {len(fail)} 篇）。")
    if fail:
        L.append("")
        L.append("缺失：" + ", ".join(f"{a} ({c})" for c, a in fail))
    open(os.path.join(BASE, "README.md"), "w", encoding="utf-8").write("\n".join(L))
    print(f"\n完成 {len(ok)}/{len(P)}，失败 {len(fail)}")


if __name__ == "__main__":
    main()
