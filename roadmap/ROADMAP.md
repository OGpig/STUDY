# 12 周详细路线图

> 总原则：**每周必须有一个「可验证的产物」**。没有产物的那一周等于没发生。
> 阶段之间不并行推进，前一阶段验收不过不进入下一阶段（防止"什么都摸了一点，什么都不深"）。

---

## 阶段 0｜地基（W1–W2）

**目标**：能自己说清楚「一段 wav 怎么变成文字」，并亲手跑通一次最小训练。

### 学什么
- 语音信号基础：采样率 / 加窗 / STFT / Mel 滤波器组 / Fbank vs MFCC；为什么 ASR 用 80 维 Fbank（对数能量、去相关）
- 三大建模范式：**CTC**（条件独立、单调对齐、blank）、**AED**（注意力编码解码、自回归）、**RNN-T/Transducer**（联合网络、流式友好）
- PyTorch 基本功：Dataset/DataLoader、collate + padding、混合精度、梯度累积、`torch.profiler`

### 做什么
1. 环境：在 `F:\embedded\prepare\.venv` 建独立 venv（**不装 C 盘**），按 `docs/ENV.md` 装 torch cu128
2. 用 `torchaudio` 手写一遍 **Fbank 提取**，和自己实现的版本对拍
3. 手写 **CTC 前向算法**（log 域，动态规划），与 `torch.nn.CTCLoss` 数值对拍
4. 跑通一个最小语料训练：AISHELL-1 的 dev 子集（如 200 条）× 5 epoch，哪怕 CER=80%

### 验收标准
- [ ] `docs/ENV.md` 里的命令全部可复现
- [ ] 手写 Fbank 与 torchaudio 输出最大误差 < 1e-3
- [ ] 手写 CTC 前向 loss 与 `CTCLoss` 相对误差 < 1e-4
- [ ] 有一次真实训练日志（loss 在下降，不是 flat line）

### 读论文
`00_survey/2111.01690`（E2E ASR 综述，重点 CTC/AED/RNN-T 三章）+ `06_foundation`（Transformer / Adam）

---

## 阶段 1｜复现（W3–W4）

**目标**：用 WeNet 在 AISHELL-1 上训练一个**能听出人话**的模型，并理解流式是怎么切出来的。

### 做什么
1. 拉取 WeNet（`curl` tarball，见 README），跑通 `examples/aishell/s0` 全流程：数据准备 → 训练 → 解码 → 计算 CER
2. 系统读 `tools/` 里的 feature 与 `wenet/transformer/` 的 Conformer 实现，**逐模块对照论文**（相对位置编码、卷积模块、Macaron 结构）
3. 跑一次 **streaming decode**（chunk 16/32），画出「chunk size → CER / 首字延迟」曲线
4. 打开 TensorBoard，解释每一条曲线为什么长这样

### 验收标准
- [ ] AISHELL-1 test CER ≤ 6%（WeNet 官方 recipe 可达 ~5%）
- [ ] 产出一张「chunk size vs CER / latency」对比图（这是面试可讲的取舍）
- [ ] 能脱离 paper 手画出 Conformer Block 的四个模块顺序和归一化位置
- [ ] `EXPERIMENTS.md` 里至少 3 条实验记录（含 1 条失败的）

### 读论文
`01_asr_architecture`：Conformer → WeNet2 → Transformer-Transducer → Zipformer

---

## 阶段 2｜科研（W5–W6）★ 岗位A 的分水岭

**目标**：把「上下文增强 ASR」这个问题吃透，并**提出一个能被证伪的假设**。

### 做什么
1. 精读 `02_contextual_biasing` 全部 7 篇，按时间线整理技术演进：
   `CLAS(1808.02480)` → `Contextual RNN-T(2006.03411)` → `TCPGen(2109.00627)` → `Adaptive CB(2306.00804)` → `LLM-based CB(2512.21828)`
2. 写一份 **mini survey**（2000~3000 字）：三条技术路线（注意力偏置 / 指针生成 / 检索增强）、各自代价、未解决问题
3. 复现一个最简偏置基线：在已训练的 WeNet 模型上做**浅融合（shallow fusion）热词 boosting**（改解码 beam score，最简单、当天可出结果）
4. 基于文献 gap 提出假设，例如：
   - 「训练期偏置 vs 解码期 boosting，在 OOV 热词上的召回差距有多大？」
   - 「热词列表规模从 10 涨到 1000，CER 劣化曲线是否可被 bias-conditioning 缓解？」

### 验收标准
- [ ] mini survey 成文，能画出演进时间线图
- [ ] 浅融合热词 baseline 跑出结果（热词召回率 / 非热词 CER 劣化幅度，两个指标都要）
- [ ] 写出实验设计文档：假设、自变量、因变量、对照组、预期结果
- [ ] 至少一个"从论文里读出来但没人做过"的点，写成一句话

### 读论文
`02_contextual_biasing` 全组；参考 `01` 的 USM 与 Whisper 章节了解大模型 ASR 现状

---

## 阶段 3｜创新实现（W7–W8）

**目标**：把 W5–W6 的假设变成代码和实验，产出**简历上最硬的一段**。

### 做什么
1. 实现偏置模块（二选一，建议先 A 后 B）：
   - **A（轻量，稳）**：解码期偏置 —— 在 WeNet 的 CTC prefix beam search / attention 解码器里注入 hotword 词图得分 + 上下文缓存
   - **B（重，像论文）**：训练期偏置 —— 照 `Adaptive Contextual Biasing` 加 bias encoder + 交叉注意力 + 门控（防"偏置成瘾"）
2. 消融实验矩阵（至少 2×3）：热词数（10/100/1000）× 方法（无偏置 / shallow fusion / 你的方法）
3. 补充评测维度：**热词召回率**、**非热词 CER 劣化**、**额外推理延迟**（三件套，缺一不可）
4. 把结论写成 1 页 A4 的实验报告（表格 + 曲线 + 结论 + 局限）

### 验收标准
- [ ] 你的方法在热词召回率上显著优于无偏置基线，且非热词 CER 劣化 < 1%
- [ ] 消融表格完整，每个数字都有对应命令可复现
- [ ] 报告里明确写了「这个方法什么时候会失败」
- [ ] 面试自测：能 3 分钟讲清 motivation → method → 结果 → 局限

### 读论文
`02` 复读 + `04_distill_quantization` 开始预习

---

## 阶段 4｜压缩（W9）

**目标**：把模型压到 RK3576 能实时跑，同时把精度损失说清楚。

### 做什么
1. 知识蒸馏：大模型（teacher）→ 小模型（student），对比 logits / CTC 后验 / 中间层特征三种蒸馏目标的差异
2. 量化：ONNX → INT8（PTQ），观察哪些层对量化最敏感（用 per-layer 误差分析）
3. 建立「精度 vs 参数量 vs 延迟」的三方权衡表
4. 目标：参数量 ↓50%+，CER 劣化 < 1.5%（相对）

### 验收标准
- [ ] 有蒸馏前后 / 量化前后的完整对比表
- [ ] 定位出至少 1 个量化敏感层，并给出处理方案
- [ ] 模型导出 ONNX，`onnxruntime` 上 CPU 推理结果与 PyTorch 一致（误差 < 1e-3）

### 读论文
`04_distill_quantization` 全组：Distil-Whisper / INT8 量化 / LSQ / KD-RNN-T

---

## 阶段 5｜板端部署（W10–W11）

**目标**：RK3576 上跑通「麦克风 → 文本」的完整流式闭环。

### 做什么
1. ONNX → RKNN（`rknn-toolkit2`），先跑通 `rknn_model_zoo/examples/zipformer` 的官方样例建立信心
2. 转换自己的模型：encoder / decoder / joiner 三件套（或 CTC 单模型），处理算子不支持的情况
3. 写 C++ 工程（**这是"Linux C++ × PyTorch 联动"的交付物**）：
   - 音频采集（ALSA / PortAudio）
   - 前端：重采样 + Fbank 特征（C++ 实现，与 Python 对拍）
   - 流式状态管理：chunk 缓存、encoder 状态传递
   - NPU 推理（librknnrt）+ 解码（贪心 / beam search）
   - 主循环 + 日志 + 延迟打点
4. 交叉编译 `aarch64`（参考 `build-linux.sh`）
5. 性能报告：RTF（real-time factor）、内存占用、首字延迟、NPU 利用率

### 验收标准
- [ ] 板端 **RTF < 1.0**（实时）
- [ ] C++ 版 Fbank 与 Python 版输出最大误差 < 1e-3
- [ ] 真实麦克风输入可连续出字，有延迟打点日志
- [ ] 一份部署踩坑记录（算子不支持 / 对齐 / 内存，都要写）

### 读论文
`04` 复读 + 回看 `01/zipformer`

---

## 阶段 6｜收尾（W12）

**目标**：把 3 个月的工作转成面试资产。

### 做什么
1. 整理实验报告为**论文初稿**（Abstract / Intro / Related Work / Method / Experiments / Conclusion），对齐岗位A 的「论文撰写与投递」
2. `RESUME.md`：3 条 bullet，每条 = 动作 + 技术 + 量化结果
3. 模拟面试题库自测（含手写算法 + 语音八股，见下）
4. 开源整理：README + 复现脚本 + demo 视频（板端跑起来的 30 秒）

### 验收标准
- [ ] 论文初稿 6 页以上，图表齐全
- [ ] 简历 3 条 bullet 都带数字
- [ ] 板端 demo 视频 ≤ 60s
- [ ] 能答出下面 80% 的问题

---

## 附 A｜手写算法训练清单（贯穿全程，每周 ≥3 题）

**语音/深度学习相关（面试差异化，务必手写）**
- [ ] CTC 前向-后向算法（log 域）
- [ ] Viterbi 解码（HMM）
- [ ] Beam Search（含长度归一化）
- [ ] Fbank / MFCC 特征提取
- [ ] Self-Attention（含 mask）
- [ ] LayerNorm / BatchNorm 手写 + 反向
- [ ] Adam 优化器手写
- [ ] 卷积/反卷积 shape 推导
- [ ] KMeans（GMM-HMM 前置）
- [ ] 在线 Softmax / 数值稳定 trick

**通用算法（笔试用，按高频排序）**
- [ ] 二分（边界版）、双指针、滑动窗口
- [ ] 前缀和 / 差分
- [ ] 单调栈 / 单调队列
- [ ] 堆 / TopK
- [ ] 排序（快排、归并、堆排手写）
- [ ] 链表（反转、环、合并）
- [ ] 二叉树遍历（递归 + 迭代）
- [ ] DFS/BFS、拓扑排序
- [ ] 并查集
- [ ] 背包 / LIS / LCS
- [ ] 字符串（KMP、字典树）

---

## 附 B｜面试八股自测题（W12 前应能答 80%）

1. CTC 的 blank 有什么用？为什么 CTC 需要条件独立假设？
2. RNN-T 比 CTC 强在哪？为什么它适合流式？
3. Conformer 为什么比纯 Transformer 好？相对位置编码解决什么问题？
4. SpecAugment 做了什么？为什么有效？
5. Fbank 和 MFCC 的区别？为什么 ASR 多用 Fbank？
6. 什么是 alignment-free？AED 的暴露偏差（exposure bias）怎么缓解？
7. 流式 ASR 的 chunk 大小如何影响精度与延迟？
8. 上下文偏置中，「偏置成瘾（bias over-fitting）」是什么？怎么抑制？
9. 模型蒸馏有哪些形式（logits / feature / relation）？ASR 里哪种更有效？
10. 量化为什么掉点？PTQ 和 QAT 的取舍？
11. 什么是 RTF？板端实时性瓶颈通常在哪个环节？
12. 语言模型融合（shallow fusion / cold fusion / deep fusion）的区别？
13. BPE / WordPiece / 中文分词的差异，对 ASR 输出有什么影响？
14. 为什么中文 ASR 常用字级建模而不是词级？
15. 端侧部署时 encoder 和 decoder 哪个更该量化？为什么？
