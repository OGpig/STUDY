# 进度追踪

> 规则：每天收工前追加一段。**不写"今天学了 XX"，写"今天产出了什么 / 卡在哪"。**
> 每天早上把本文件最新一段给我，我据此出当天安排。

---

## 总进度看板

| 阶段 | 周 | 状态 | 关键产物 |
|---|---|---|---|
| 0 地基 | W1–2 | ✅ **D2 全绿**（四条验收全过） | 环境 / Fbank bit-exact / CTC 对拍 / 真实训练日志（loss 223.5→91.9） |
| 1 复现 | W3–4 | 🟡 进行中（D3 开工） | AISHELL-1 CER ≤ 6% + chunk 曲线 |
| 2 科研 | W5–6 | ⬜ 未开始 | mini survey + 热词 baseline |
| 3 创新 | W7–8 | ⬜ 未开始 | 偏置模块 + 消融表 |
| 4 压缩 | W9 | ⬜ 未开始 | 蒸馏/量化对比表 |
| 5 部署 | W10–11 | ⬜ 未开始 | RK3576 RTF < 1.0 |
| 6 收尾 | W12 | ⬜ 未开始 | 论文初稿 + 简历 |

---

## 2026-09-18 D1（阶段 0 第 1 天）

- 完成：
  - ✅ **任务 1 环境搭建完成并通过完整自检**（`scripts/test/test_env.py`）
    - venv `F:\embedded\prepare\.venv`（Python 3.13.14），**基于独立解释器**，`base_prefix` 已与 conda 脱钩
    - GPU 自检：RTX 5060，**capability `(12, 0)`**（确认真 Blackwell 路径，非兼容模式）、CUDA 12.8、fp32/fp16/bf16 三种 dtype 矩阵乘全部通过、`synchronize` 正常
    - 依赖全绿：torch/torchaudio **2.11.0+cu128**、numpy 2.5.3、pandas 3.0.6、scipy 1.18.1、soundfile 0.14.0、librosa 1.0.0、numba 0.67.0、tensorboard 2.21.0、scikit-learn 1.9.1
  - 环境基线与风险对策写入 `docs/ENV.md`（唯一真相来源）
  - 新增工具脚本：`scripts/setup_env.ps1`（一键重建环境）、`scripts/activate_project.ps1`（会话级隔离 conda）、`scripts/test/test_env.py`（环境自检）
- 今日踩坑（均已修复，详见 `docs/ENV.md`）
  1. ~~venv 建在 anaconda 上（半独立）~~ → 改用 WorkBuddy 独立 Python 重建，`base_prefix` 脱钩 ✅
  2. ~~清华源报 `from versions: none`，连 `six` 都装不上~~ → 根因是 **venv 自带 pip 26.1.2 与镜像 simple API 不兼容**；用官方源升到 26.2.1 后立刻恢复 ✅
  3. ~~终端提示符 `(base)` 清不掉~~ → 根因是 conda 注入的 `prompt()` 读 `$Env:CONDA_PROMPT_MODIFIER`（第一版脚本漏清此项）；已重写脚本 + 执行 `conda config --set auto_activate_base false` 根治新终端 ✅
- 待验证风险（已识别，未触发）：
  - WeNet 的 `requirements.txt` 不能全量装（deepspeed 版本敏感 / `openai-whisper==20231117` 与 numpy 2.x 冲突 / flake8 老 pin）
  - Python 3.13 可能缺 `distutils` → 先 `pip install setuptools`，不行再建 3.11 副 venv
  - **Windows 跑不了 WeNet 的 `run.sh`** → 手动分步执行 Python 入口；W2 末尾先在 200 条子集上验证一次
- 明天第一件事：
  - D1 全部任务已收口：~~Fbank 对拍~~ ✅ bit-exact ｜ ~~CTC 前向~~ ✅ 对拍全过
    ｜ ~~CTC 转移图~~ ✅ 已评审+参考图 ｜ ~~WeNet 源码~~ ✅ 17 个文件读完
  - D2 起点：按 DAY01 任务 5 末尾的**阅读顺序**精读 `encoder.py::forward` → `mask.py` → `encoder_layer.py`
    → `forward_chunk` → `asr_model.py::forward`，然后开始 AISHELL-1 数据准备
- 今日已完成的其他任务：
  - ✅ **论文精读（§1 引言 + §2 三大范式）**，笔记 `notes/P01_e2e-survey.md`——含三问 QA 逐条批改（引用原文页码）、三范式对比表、方法论 4 条
  - ✅ 新增读论文工具 `scripts/tools/paper.py`（`toc` 摸结构 / `grep` 定位 / `page` 看单页），对付两栏 PDF 检索很有效
  - ✅ **手写 Fbank 对拍完成并已修正 → bit-exact**（脚本 `scripts/test/compare_fbank.py`，代码 `scripts/test/Fbank.py`）
    - 修正前：log 域 mean abs err **1.45**、rel **139%**（特征范围才 [-23.0, 4.3]），不能算对齐
    - 修正后：**`torch.equal == True`，44720 个数值零误差**；参数/音频边界扫描 **20/20 全 bit-exact**
    - 修掉的 3 处（详见 `roadmap/DAY01.md` 任务 3 完成记录）：
      a) 🔴 **方法性错误**——mel 滤波器组把端点在 **bin 索引域** `floor` 量化后插值，低频 **5 个 mel bin 整行塌陷为 0**（bin 0/2/5/9/14，与端到端误差最大的 5 个 bin 完全重合）→ 改为 Kaldi 的 **mel 域** `clamp(min(up,down))` 向量化
      b) 🔴 FFT 点数 400 → **512**（`round_to_power_of_two` 零填充），bin 宽度 40 Hz → 31.25 Hz
      c) 🟡 补 `remove_dc_offset`，且**预加重改为帧内做**（首样本 `0.03·x[0]`，非整段波形）
    - 归因消融（修正前）：只换 mel banks → err 降 77%；只换 FFT 点数 → 降 53%；两者都换 → 0.22%
    - 做对的部分：Povey 窗、mel 刻度公式、low/high_freq、功率谱、帧数公式 ✅
  - ✅ 新增环境坑记录：`torchaudio >= 2.9` 的 `load()` 需要额外装 `torchcodec`，测试脚本改用 `soundfile` 绕开
  - ✅ **新建 `coding/` 算法刷题区 + C++ 工具链配通**（见 `coding/README.md`）
    - `coding/build.py` 一键编译运行驱动：自动探测工具链、拼 MSVC env、喂测试输入、计时、超时保护
    - 两套编译器实测均通过：**g++ 16.2.0（默认，`-Wall -Wextra` 零警告）** / MSVC 14.51 + SDK 10.0.26100.0
    - `.vscode/` 5 个配置放在**项目根**（用户实际打开的工作区层），`Ctrl+Shift+B` = 编译并运行
    - 新坑两条（已写进 `win-msvc-vscode-cpp` skill）：Git Bash 的 PATH 分隔符是 `:` 不是 `;`；
      `<VSCode>/bin/code` CLI 在本机不可用 → cpptools 扩展需手动一键装
    - ⚠️ **待用户操作**：打开项目时点一下「推荐扩展」里的 Install（`ms-vscode.cpptools`），
      否则没有 IntelliSense 和 F5 调试（不影响编译运行）
  - 🎯 **已出题 P01｜最长递增子序列（LIS，中等）** → `coding/problems/P01_LIS.md`，骨架 `coding/solutions/P01_LIS.cpp`
    - 选它是因为**正解就是 D1 任务 6 练过的 `lower_bound` 唯一实战用法**，接得上
    - 四问：① O(n²) DP ② O(n log n) + 追问「为什么不是 upper_bound」/「非严格递增该换哪个」
      ③ 还原一条具体 LIS + 自写 `is_valid()` 校验 ④ 两解法耗时实测对比
    - 红线遵守：只给 I/O 骨架和方向性提示，**不给算法实现**
    - ✅ **用户写完 → 审查 → 修正 → 验收全部做完**（详见 `coding/README.md`「P01 审查记录」）
      - 用户的**算法逻辑 100% 正确**（10/10 边界用例、n=100000 得 614 = 独立参考值），
        但有 2 个 bug 让它拿不到分：① `return len;` 恒返回 0（回溯把 len 减到 0 了）
        ② `dp[i]` 存了整条序列 → 空间 O(n²)、**最坏时间 O(n³)**，单调递增 n=8000 要 33906 ms
      - 另有 6 个 `-Wsign-compare` 警告、n=0 段错误（rc 3221225477）、4 条注释 TODO 未填
      - 修正后：**12/12 用例通过 + 零警告 + n=0 正常**；DP 从 33906 ms → **9.5 ms**（3582×）
      - ⚠️ 只留了两处「为什么」没写（追问 A/B），那是面试真正的考点
    - 📌 **可复用的方法论教训**：**随机数据不是复杂度检验的试金石** ——
      这个 O(n³) 在随机输入下实测指数 ≈ 2.0（完全看不出来），
      只有**单调递增**输入才暴露（规模 8 倍 → 耗时 1685 倍）。**两种输入都要测。**
    - 📌 另一个教训：**刷题必须看退出码**。n=0 那次 stdout/stderr 全部正常打印完了，
      只有退出码是 0xC0000005 —— 看输出会以为成功，评测机上直接判 RE
  - ✅ **手写 CTC 前向算法对拍完成并已修正**（代码 `scripts/test/CTC.py`，对拍 `scripts/test/compare_ctc.py`）
    - **α 递推逻辑经消融验证本身完全正确**：把 blank 换成与自己硬编码一致的值后 **max|Δ| = 0**；
      换成 torch 默认的 `blank=0` 也一致（3.8e-06）→ 说明三角递推 / `logaddexp` / 跳过一格的约束全写对了
    - 修正后：主对拍 4/4 PASS（T=100 时 max|Δ|=9.16e-05）、数值稳定性 PASS（无下溢无 NaN）、边界 10/10 PASS
    - 修掉的 4 处（详见 `roadmap/DAY01.md` 任务 4 完成记录）：
      a) 🔴 **致命：`blank_idx = C-1` 硬编码**——torch 默认 `blank=0`，两者混用会"看起来对、结果全错" → 改为参数 `blank=0`
      b) 🟡 终止统计写在 `t` 循环内部 → 重复算 T-1 次，且 **T==1 时 `losses` 未定义直接崩** → 移到循环外
      c) 🟡 `last_index-1` 在 L==0 时 = **-1 负索引绕回最后一列**（恰好那列是 -inf 才没暴露）→ 显式判边界
      d) 🟡 自测语义错配：targets 含 0 与 `blank=0` 自相矛盾；返回 (N,) 却与 `reduction='mean'` 的标量比大小
    - 顺带得到的可行性约束：**T_min = L + 相邻重复标签对数**（实测 `[1,1,2,2]` 最少 6 帧）
    - ⚠️ 教训：「T==L」「帧数不足」两个用例在**修正前也是 PASS**——通过边界用例 ≠ 实现正确，必须做随机中大规模主对拍
    - 对拍脚本含**基线自检**：内置纯定义的暴力枚举器（非 DP）先与 `CTCLoss` 对齐，证明参照本身可信
  - ✅ **CTC 转移图已手绘并评审**（18:15）——交替结构、三种箭头形态、初始化位置都画对了；
    待补 3 处：**少末尾那个 blank 行（S 应为 2L+1）**、可达区应是斜带而非一条线、缺坐标轴与图例
    - 已补一张**标准参考图** `roadmap/assets/DAY01_ctc-lattice.svg`（已内嵌进 DAY01）：
      含坐标轴、三色箭头图例、可达域标注、起止圈选，外加「目标 `aa` 时 ③ 为何禁止」的对照图
    - 评审详情见 `roadmap/DAY01.md` 任务 4「转移图（参考图）」+「手绘稿评审」
  - ✅ **WeNet 源码已拉取并读完 `wenet/models/transformer/` 全 17 个文件**
    - ⚠️ **路径校正**：实际在 `examples/wenet-main/`（不是 DAY01 计划的 `third_party/`）；
      目录是 **`wenet/models/transformer/`**（新版把 backbone 变体全收进 `wenet/models/`，
      含 14 个同级目录：branchformer / squeezeformer / efficient_conformer / transducer / paraformer / …）
    - 逐文件一句话解读 + Conformer 层真实结构 + 流式三层机制 → 见 `roadmap/DAY01.md` 任务 5 完成记录
    - 流式机制摸清：**① `chunk_mask`（`utils/mask.py::subsequent_chunk_mask`，流式总开关）
      ② 训练期动态 chunk（`use_dynamic_chunk`，一半概率采全上下文 = U2 的流式/非流式统一）
      ③ `att_cache` + `cnn_cache`**；subsampling 故意不做 cache，靠输入重叠补右上下文
    - ★★ **关键发现：WeNet 主线没有模型侧的上下文偏置**。全仓库搜 `contextual/biasing/hotword`
      只有 3 个文件命中，且全是**解码期浅融合**——`utils/context_graph.py` 是个带 fail 弧的
      Aho-Corasick trie，beam search 每步查一次、命中就加分，**模型权重完全不动**。
      → 意味着「Conformer + 上下文热词偏置」的**模型侧部分要自己写**（创新点空间在这里）；
      可参考 icefall/sherpa-onnx 的 Zipformer 偏置实现（本机 `rknn_model_zoo` 有 zipformer 示例）
      或论文侧 CLAS / TCPGen
    - 两个可讲的细节：`RelPositionMultiHeadedAttention` **把 `rel_shift` 注释掉了**
      （源码注明 speech 里无用且流式难处理）；macaron 的 `ff_scale=0.5` 是**写死的**设计而非超参
    - 部署路径确认：`runtime/` 下 RK3576 走 **`onnxruntime/`** 后端

---

## 2026-09-20 D2 前置：AISHELL-1 数据获取（AI 代做，非当日任务）

> 这段是 **D2 开工前由 AI 做掉的准备工作**，记录下来是因为**选型过程和实测数字比结果值钱**。
> D2 的正式安排见 `roadmap/DAY02.md`。

### 核心发现：瓶颈是「请求数」不是「带宽」

在同一镜像（hf-mirror.com）上实测：

| 场景 | 请求数 | 体积 | 实测速度 |
|---|---|---|---|
| 单个 455 MB 大 parquet | 1 | 455 MB | **11.1 MB/s** |
| 300 个 ~150 KB 小 wav | 300 | 45.7 MB | **0.16 MB/s（≈1 文件/秒，24 并发也无改善）** |

→ 镜像对小请求限流，**加线程救不回来**。选数据源第一原则：能少发请求就少发请求。

### 四个源的实测对比

| 源 | 内容 | 请求数 | 体积 | 实测 | 预估耗时 |
|---|---|---|---|---|---|
| OpenSLR 官方 | 全量（官方 tgz） | 1 | 15.58 GB | 0.59 MB/s | **~7.3 h** |
| `AISHELL/AISHELL-1` | train 100/340 说话人 | 100 | 3.45 GB | ~19 MB/s | ~4 min |
| `shenyunhang/AISHELL-1` | 全量官方划分 | 122,000 | ~14 GB | ~1 文件/秒 | **~33 h** |
| ★ `carlot/AIShell` | **全量官方划分** | **37** | 20.3 GB | 11.1 MB/s | **~30 min** |

### 采用的方案

- **train**：`AISHELL/AISHELL-1` 的 100 个 speaker tar 分片 → 已解压成标准布局，
  **34,716 条 / 4.61 GB**（覆盖全语料约 29%）
- **dev/test**：`carlot/AIShell` 的 `validation` + `test`（官方 14,326 / 7,176 条）
  → 2 个请求 3.4 GB，约 5 分钟。**必须用官方划分**，否则 CER 与文献不可比
- **全量 train**（可选，W3-W4 前完成）：`carlot/AIShell` 35 片 / 17.3 GB / ~26 min

### carlot parquet 结构（已扒开验证）

```
audio: struct<bytes: binary, path: string>
    bytes = 原始 RIFF WAV 数据（与官方文件逐字节一致）
    path  = 原始文件名 = utt id，如 BAC009S0002W0122.wav
transcription: string   # '而 对 楼市 成交 抑制 作用 最 大 的 限 购'
```
转换实测 **1902 条/秒**（用 `iter_batches` 流式读，不整片载入内存）。

### 当前数据状态

```
data/aishell/raw/
├── wav/train/S0002 … S0101/          34,716 个 wav   (4.6 GB)
├── wav/dev/S0724 … S0763/            14,326 个 wav   ✅ 官方 dev
├── wav/test/S0764 … S0783/            7,176 个 wav   ✅ 官方 test
├── dev_text.txt / test_text.txt       14,326 / 7,176 行
├── transcript/aishell_transcript_v0.8.txt   141,600 行  ✅ 与官方一致
└── resource/lexicon.txt                     139,874 行
data/aishell/download/                 6.58 GB  ← 原件缓存，确认无误后可删
```

### 全量校验（`scripts/check_aishell.py`，8 项检查）

| # | 检查项 | 结果 |
|---|---|---|
| 1 | 数量 | train 34,716 wav（**34,679 条有文本**）/ dev **14,326** ✅ / test **7,176** ✅ |
| 2 | wav ↔ text 一一对应 | dev/test 0 孤儿 ✅；train「仅 wav 37 / 仅 text 0」（正常，见下） |
| 3 | 音频规格 | 全部 (1ch, 16000, 16bit) ✅ |
| 3b | 空音频 | ⚠️ train 1 个，dev/test 0 |
| 4 | 时长 | train 42.93h / dev 18.09h / test 10.03h，p50 约 4.2~4.7s |
| 5 | 文本合法性 | 空文本 0；不在 lexicon 的字 **0 种** ✅ |
| 6 | 说话人隔离 | train/dev/test 两两无交叉 ✅ |
| 7 | 文本来源交叉验证 | dev/test 各 **100% 仅空白差异，0 实质不一致** ✅ |
| 8 | 无文本音频 | train 37 个 = 36 未标注正常音频 + 1 空音频 |

**查出 3 个问题：**

1. ★ **train 有 1 个空音频**：`train/S0048/BAC009S0048W0490.wav` = 44 字节
   （WAV 头合法但 data 块 = 0）。该 utt 官方 transcript 里也没文本 → AISHELL-1 原版缺陷。
   → 后果：data.list 里保留它会在训练中途崩（启动时不报错）。
2. ★ **carlot 的 dev/test 文本带统一的 4 个前导空格**（14,326 + 7,176 条 100% 都有）；
   官方 transcript 带尾部空格。去掉所有空白后两者**逐字一致、长度差全为 0**。
   → 归一化必须用 `re.sub(r"\s+","",t)`，**不能只 strip 首尾**。
   💡 这类 bug「每一条都错、错得一模一样」，抽查发现不了，只有全量交叉比对才行。
3. ★★ **train 的 wav 数 ≠ text 数，这是正常的**：wav 34,716 / 有文本 34,679 / **差 37**。
   AISHELL-1 官方本身就不对齐：wav 共 **141,925** 个文件，transcript 只有 **141,600** 行
   （官方脚本里 `[ $n -ne 141925 ]` 那句警告就是说这个）。
   37 个拆开：**36 个是正常音频（2.35~8.01s）但官方没给标注** + 1 个空音频。
   → 官方 recipe 用 `filter_scp.pl` **取交集**处理 → **最终 train = 34,679**
     （我一开始误报成 34,715，查清后已修正——这一点要写进 DAY02 的验收标准，
      否则会去追一个不存在的 off-by-one bug）

**审计脚本的两次自我修正（都已修）**：
- 第一版把「抽样 300 个文件的时长求和」当全量总时长打印（train 显示 0.4h，实际 42.93h）
  → **抽样只能验规格，总量统计必须全量**。
- 第一版 [1][2] 两项**跳过了 train**（因为 train 没有现成 text 文件），导致漏检上面第 3 条
  → 改为用「官方 transcript ∩ wav id」现场推导，并新增 [8] 专门解释这个不对齐。

### 新增资产

- `scripts/download_aishell.py` —— 下载器（三个源的全部实测数据与选型理由写在文件头注释里；
  支持 `--stage meta | train-tar | carlot-devtest | carlot-train`，可断点续传）
- 新依赖 `pyarrow 25.0.1`（读 parquet）

### 顺带确认的 Windows 坑（D2 会用到）

- `examples/wenet-main/examples/aishell/s0/` 里的 **`tools` 和 `wenet` 本该是符号链接，
  被 `tar.exe` 解压成了普通文本文件**（内容就是路径字符串）→ 官方 `run.sh` 整条流程在 Windows 不可用
- WeNet 新版 `train.py` 里 `init_distributed(args)` 是必经路径，**没有"不开分布式"的开关**
  → 单卡也要用 `torchrun --nproc_per_node=1` 启动
- 官方 recipe 写的 `dist_backend=nccl` **Windows 上没有** → 单机必须改 `gloo`
- ⚠️ **`warmup_steps=25000` 与小数据量严重不匹配**：200 条 / batch 8 = 每 epoch 25 步，
  5 epoch 才 125 步 → 学习率全程停在爬升最底部，**loss 会是一条平线**。
  这会被误判成"模型/数据有问题"，是 D2 最容易踩的坑

---

## 2026-09-20 D2 任务 2/3：encoder 精读 + 数据准备（AI 代做，用户要求）

### 任务 2 批改结果：6 题全对

逐题结论与源码依据见 `roadmap/DAY02.md` 任务 2 完成记录。补了 3 处精确度
（第一个 mask 是 padding mask；mask 有第三分支且 `max_chunk_size=25` 写死；
流式下 `pos_emb` 按 `offset − cache_t1` 重算到 `attention_key_size`）。

产出 **`roadmap/assets/DAY02_conformer-streaming-flow.svg`**（viewBox 680×1276，五段结构，
配色与 D1 的 CTC 图一致）：前端维度变化 / 两个 mask / 12 层横向展开 + 双 cache /
单层 macaron 放大 / 输出两路 + 损失加权。

### 任务 3 完成：数据准备一条命令跑通

**交付**：`scripts/prepare_aishell.py`（9 阶段）+ `scripts/run_prepare.ps1`（一键封装）

```powershell
cd F:\embedded\prepare
.\.venv\Scripts\python.exe scripts\prepare_aishell.py --stage all
```
不需要激活 venv、不需要设环境变量（脚本自动找 FFmpeg 并挂 PATH）。

**实测产物**（全新环境从零跑，退出码 0）：

| 产物 | 数量 |
|---|---|
| train `wav.scp`/`text`/`data.list` | **34,679**（三者一致） |
| dev / test | 14,326 / 7,176 |
| `dict/lang_char.txt` | 3,590 行 |
| `train/global_cmvn` | 80 维，`frame_num=15,367,984`（≈42.7h），**65 s / 12 进程** |
| `train_smoke` / `dev_smoke` | 200 / 50 |
| `conf/smoke.yaml` | 7 项覆盖，流式参数保持官方默认 |

**三处与官方 recipe 的有意不同**：① 文本统一用官方 transcript（carlot 只做交叉验证）
② 归一化删**所有**空白 ③ 显式剔除空音频并打日志。

### ★★ 本次最大的坑：torchaudio 2.11 解码需要 FFmpeg 共享库

`torchaudio.load()` 报 `Could not load libtorchcodec`；`torchaudio.info()` **在 2.11 已被删除**。
根因：解码改为依赖 torchcodec，而 torchcodec **不捆绑 FFmpeg 共享库**。
**WeNet 的 `processor.py:156` 就是 `torchaudio.load` → 训练会直接起不来**，
而报错与"数据准备"看起来毫无关系，极难查。

→ 装 FFmpeg **shared** 构建（`ghproxy.net` 镜像下载，86 MB），
位置 `F:\data\toolchains\ffmpeg-shared\...\bin`。**WeNet 源码一行没改。**
→ 连带：`tools/compute_cmvn_stats.py` 同时用被删的 `torchaudio.info`，**即使有 FFmpeg 也跑不了**
  → CMVN 自写（`soundfile` 读 + `kaldi.fbank` 算 + 多进程），实测 34,679 条 / 65 秒。

### 另两个工程坑（已写进 `docs/ENV.md` 风险 7）

- `$ErrorActionPreference="Stop"` 会把**原生命令写到 stderr 的每一行**当成终止性错误
  → `run_prepare.ps1` 第一版打印一行后就中止，Python 一行没跑
- **无 BOM 的 UTF-8 `.ps1` 被 PowerShell 5.1 当 GBK 读**，中文错位吃掉引号
  → PSParser 报"字符串缺少终止符"；含中文的 .ps1 必须存 **UTF-8 with BOM**

### 纠正 DAY02 的一处笔误

原文说 `data.list` 里"`duration` 必须算对"—— **错的**。新版 `make_raw_list.py` 只写
`{"key","wav","txt"}` 三个字段，**没有 duration/sample_rate**，时长由 dataset 运行时算。

### 新增依赖

pyyaml 6.0.3、pyarrow 25.0.1 ｜ 新增环境资产：FFmpeg shared（214 MB，非 Python 包）

---

## 2026-09-20 D2 任务 4：★ 训练跑通，阶段 0 全绿

**最终指令**（不需要激活 venv、不需要设环境变量）：
```powershell
cd F:\embedded\prepare
.\.venv\Scripts\python.exe scripts\run_train.py --preset smoke
```

**结果**：`EXIT=0`；train loss 274.45 → 203.37 → … → **70.34**；cv_loss 114.46 → … → **89.88**。
产物 `work/aishell/exp/smoke/{epoch_0..4.pt, final.pt}` + `work/aishell/tensorboard/smoke`。

✅ **阶段 0 四条验收全部通过**（环境 / Fbank bit-exact / CTC 对拍 / 真实训练日志）。

### 从报错到跑通，修了 7 个问题

| # | 问题 | 修法 |
|---|---|---|
| ① | `use_libuv` 报错（**提示设 `USE_LIBUV=0` 是误导，实测无效**） | `scripts/patch_torch_libuv.py` 给 torch 里 9 处 TCPStore 补 `use_libuv=False` |
| ② | `No module named 'langid'` | 补装 |
| ③ | `No module named 'tqdm'` | 补装 |
| ④ | `Union` 无法从 `torch.nn.modules.conv` 导入 | `scripts/patch_wenet.py` 改从 typing 导入 |
| ⑤ | 顶层 `import deepspeed`（用不到却必需） | 改 try/except 可选 |
| ⑥ | 顶层 `import whisper`（同上） | 改惰性导入 |
| ⑦ | `exp/smoke/train.yaml` 不存在 | 启动器补 `mkdir`（官方 run.sh 里有） |

**两个自己踩的坑**：Git Bash 会把 `export PATH="F:/..."` 改写掉（→ 改为在 Python 里设 PATH）；
`--num_workers 0` 与 WeNet 硬传的 `prefetch_factor` 冲突（→ 必须 ≥1）。

### ✅ GPU 训练已跑通（原「CUDA 段错误」已解决）

**根因**：PyTorch 的 **Windows gloo 是 CPU-only 构建**（无 NCCL），
而 DDP 反向要把 **CUDA** 梯度 `all_reduce` → gloo 拿不到 CUDA tensor → **段错误**。
WeNet 的 `wrap_cuda_model` 里 DDP **无条件包**（`world_size==1` 也包）。
→ 修法：`scripts/patch_wenet.py` 第 8 处，`world_size == 1` 时跳过 DDP。

**定位方法**（值得复用的消融顺序）：
① 7 个基础 CUDA 算子 ✅ → 排除算子
② 模型单独在 CUDA 上前向 ✅ → 排除模型
③ + `set_device` + `init_process_group('gloo')` 后前向+反向 ✅ → 排除分布式初始化
④ 读 `wrap_cuda_model` 源码 → 发现 DDP 无条件包
⑤ **最小复现：`nn.Linear` 裸跑 vs 包 DDP** → 裸跑正常、包 DDP 后 backward **立刻 segfault(139)**

**实测**（`run_train.py --preset smoke`，自动选 cuda）：`EXIT=0`，
cv_loss 114.38 → 101.56 → 91.99 → 90.16 → **89.88**，
**14.4 steps/sec**（CPU 0.55 → **快 26 倍**）。

> ⚠️ **长远影响**：Windows 只有 gloo，**DDP 多卡在 Windows 上走不通**，多卡要上 Linux/WSL。

### 环境资产

- `scripts/run_train.py` —— 训练启动器（PATH/PYTHONPATH/mkdir/rdzv/device 一次做掉）
- `scripts/patch_torch_libuv.py` —— 修 torch Windows 轮子的 TCPStore bug（9 处）
- `scripts/patch_wenet.py` —— 修 WeNet 兼容性（**8 处**，含单卡不包 DDP）
- 新依赖：pyyaml / tqdm / sentencepiece / tensorboardX / Pillow / textgrid / langid

### ✅ 手动命令也跑通了（追加）

用户在 DAY02 4.3 的**手敲命令**上又撞了两层问题，均已修复：

| 现象 | 根因 | 修法 |
|---|---|---|
| `UnicodeDecodeError: 'gbk' codec ...` | `train.py` 读 yaml 未指定编码 + 配置含中文注释 | 13 处 yaml 读写统一加 `encoding='utf-8'` |
| 修完变成 `ZeroDivisionError` | ★ **`data.list` 被按 GBK 逐行读** → 含中文的行解码失败 → `map_ignore_error` **静默丢光样本** | `FileOpener(..., encoding='utf-8')` |
| `FileNotFoundError: exp/smoke\train.yaml` | 官方 `run.sh` 有 `mkdir -p $dir`，手敲漏了 | 自动 `os.makedirs` |
| FFmpeg 缺失也会伪装成同一个 `ZeroDivisionError` | `decode_wav` 里的 `torchaudio.load` 同样在 `map_ignore_error` 里 | `scripts/install_env_fix.py` 用 `.pth` + `os.add_dll_directory` 一劳永逸 |

**最终实测**（用户的原命令，**不设任何 UTF-8 变量、PATH 里没有 FFmpeg**）：
`EXIT=0`，gbk/ZeroDiv 错误 **0/0**，cv_loss 114.38 → **89.88**，**14.6 steps/sec**。

> 🔍 **排查口诀（重要）**：看到 `ZeroDivisionError`（在 `executor.cv`）→
> **先查 ① 文件编码 ② FFmpeg**，别去读 cv 的代码，它没问题。
> 本质是"丢数据而不自知"：不报错、不崩溃，只在很远的地方表现为不相干的错误。

**补丁总数**：`patch_wenet.py` **25 处 / 16 个文件** / `patch_torch_libuv.py` **9 处**（都可 `--check` / `--revert`）

### ✅ 收尾：`final.pt` 断链（第 3 轮报告的错）

训练已跑完 5 个 epoch，只在最后 `os.symlink` 那行失败。读源码发现 **WeNet 自身两个 bug**：

| # | Bug | 后果 |
|---|---|---|
| 1 | 清理用 `os.path.exists` —— **对断链返回 False** → 清理被跳过 | 同目录重跑必然 `FileExistsError` |
| 2 | 链接目标写成 `4.pt`，实际文件是 `epoch_4.pt` | **`final.pt` 天生是断链**；`run.sh` stage 5 用它解码会失败 |

> ⚠️ 严重性提醒：这个 bug 容易被漏 —— `ls` 能看到 `final.pt`、命令"看起来跑完了"，
> 但那是指向不存在文件的断链。**检查文件要 `islink()` + `exists()` 两个都看。**
> 我此前几次"✅ final.pt 产出"的检查**全都是假阳性**。

**修复后实测**：连跑两次都 `EXIT=0`（cv_loss 89.8847）；
`final.pt` → `islink=True`、指向 `epoch_4.pt`、**目标存在**、108 MB。

### ✅ 任务 4.4 验收标准（四条全部完成）

| # | 验收项 | 证据 |
|---|---|---|
| 1 | 训练能启动，不报错跑完 5 个 epoch | `EXIT=0`；连跑两次均通过（`epoch_0..4.pt` + `final.pt`） |
| 2 | loss 从 ~300 量级降到明显更小，`loss_att`/`loss_ctc` 都在动 | 见下表（**从 tfevents 直接读出**） |
| 3 | `tensorboard --logdir` 能看到下降曲线 | 实测起服务：`GET /` → 200；`/data/plugin/scalars/tags` → 11 个 tag；`/scalars?tag=train/train_loss` → 280 个真实点 |
| 4 | 写 1 条 `EXPERIMENTS.md` 记录 | **`EXPERIMENTS.md`** 已建（项目根，含模板 + E01 完整记录） |

**验证集（`epoch/*`，最干净的证据）**

| epoch | loss | loss_att | loss_ctc | acc |
|---|---|---|---|---|
| 0 | 114.38 | 105.45 | 135.21 | 0.0000 |
| 1 | 101.56 | 98.29 | 109.20 | 0.0661 |
| 2 | 91.99 | 88.32 | 100.57 | 0.0512 |
| 3 | 90.16 | 86.52 | 98.65 | 0.0566 |
| 4 | **89.88** | **86.03** | **98.88** | **0.0606** |

训练侧：`train/train_loss` 首 batch 273.87 → 末 132.35（序列前 1/5 均值 171.72 → 后 1/5 均值 93.58）；
`train/loss_ctc` 670.08 → 148.76；`grad_norm` 343.75 → **41.09**（降一个量级）；
`lr` 0 → 2.8e-4（warmup 按预期）。

**曲线图**：`work/aishell/tensorboard/smoke/curves.svg`
（新工具 `scripts/plot_tensorboard.py`，**零依赖 SVG 绘图**，后续画 chunk 曲线可复用）

> ⚠️ 自查记录：这份记录我改过两次数字 ——
> ① 第一版 `loss_att`/`loss_ctc` 的中间三行是**凭趋势推测填的**，核对后发现不符
> （epoch 1 的 att 我写 96.36，**实际 98.29**）；
> ② 第二版"每 epoch 均值"按 56 点切分，而 56 点在拼接序列里**横跨两个 epoch**，整体是错的。
> 已全部改为从 tfevents 读出、并可复现的真值。
> **教训：写进记录的数字必须能追溯到原始数据文件；"看起来合理"不是合格标准。**

### 🐛 曲线图的三个问题（已修）

用户指出图有问题，定位并修复：

| # | 问题 | 根因 | 修法 |
|---|---|---|---|
| 1 | **折线有锯齿**（降到最低又跳回起点） | `tensorboard/smoke/` 下堆了**两次运行的 event 文件**，EventAccumulator 合并 → 每个 step 出现 2 次（280 点其实只有 140 唯一 step） | 绘图器**按 step 去重**并**告警**；同时删掉重复的旧 event 文件 |
| 2 | **`loss_att` 被压成一条看不见的线** | 同面板里 `loss_ctc` 早期到 670，`loss_att` 只有 ~104，线性轴下后者贴底 | 训练面板改用**对数轴**（只改坐标不改数据）；面板标题标注 |
| 3 | 训练侧"每 epoch 均值"**数字是错的** | 去重前 280 点，按 56 点切片横跨两个 epoch | 按 **28 步/epoch** 重算（边界已由数值验证） |

**修正后的训练侧每 epoch 均值（全部单调下降）**

| epoch | train_loss | loss_att | loss_ctc |
|---|---|---|---|
| 0 | 223.54 | 116.65 | 472.95 |
| 1 | 119.90 | 113.77 | 134.22 |
| 2 | 103.07 | 98.96 | 112.66 |
| 3 | 95.25 | 90.86 | 105.50 |
| 4 | **91.91** | **87.21** | **102.86** |

**验证**：图 7 条折线 **0 次回退**、**0 条文字越界**；
TensorBoard 起服务后 `train/train_loss` → **140 点**、`epoch/loss` → **5 点**（已无重复）。

### ⚠️ 新增一个必须在阶段 1 之前查清的疑点

**每个 epoch 有 28 步，但 `train_smoke` 是 200 条 / batch 8 = 25 步。多出 3 步。**
（140 唯一 step ÷ 5 = 28.0；且按 28 步切出的每 epoch 均值单调，说明这个边界是数值上站得住的。）
怀疑 `shuffle_size=1500` / `sort_size=500` **都远大于数据集大小（200）**，
缓冲区冲刷时会多吐 batch。
**如果数据管线每 epoch 会重复喂样本，等于训练集被悄悄换了个版本，会直接影响 CER 可信度** ——
验证方法：把这两个值改成 ≤ 数据集大小再跑一次，数 step 是否变成 125。

### 新增工具

`scripts/plot_tensorboard.py` —— tfevents → **零依赖 SVG** 曲线图（支持对数轴、按 step 去重、
多 event 文件检测告警）。阶段 1 画「chunk size vs CER/延迟」曲线直接复用。
（13:10 后改用 matplotlib 3.11.2 重写：x 轴统一为 epoch 级 4 面板，per-batch 面板降为 `--show-batch` 可选项。）

---

## 2026-09-20 D2 收尾（14:00–16:40）★ 阶段 0 全绿并纠正了一次验收误判

### ★★ 最重要的一次纠正：「loss 下降」不等于「学会了」

用户直接问「这到底在训练什么」→ 查完发现**模型基本什么都没学会**（我此前一直拿 loss 当验收依据，是错的）：

| 判据 | 实测 |
|---|---|
| **解码 CER** | **100.00%**（dev_smoke 50 条，识别为非空的 **0/50**，全部输出空串=每帧预测 blank） |
| 与「完全不听音频」基线比 | 只学会汉字边际分布的 loss_att = **92.5**，实测 87.2 → **只低 6%** |
| epoch acc | 6%，与上面完全吻合 |

→ **纪律（已写进 `EXPERIMENTS.md`）**：判据必须是**任务指标 + 与平凡基线比**，不能只跟自己比。
→ 但**不违反阶段 0 验收**：ROADMAP 阶段 0 原话是「哪怕 CER=80%」+「loss 在下降，不是 flat line」，
本来就不要求识别能力；E01 的真实价值是**工程链路通**。

### E02 决定性对照实验 → 证明训练管线没有 bug

| | E01 冒烟 | E02 mid |
|---|---|---|
| 数据 / epoch | 200 × 5 | **3000 × 20**（3760 步） |
| **CER** | 100% | **72.70%**（0 空串） |
| cv acc | 6% | **22.6%** |
| train th_acc | 6% | **95~97%** |
| 耗时 | 35 秒 | **~21 分钟** |

→ **管线正确**，E01 的 CER=100% 纯粹是欠训练（27.2M 参数 vs 1000 次曝光，参数:样本 = 27,178:1）。

**顺带查出一个真 bug**：`warmup_steps=500` > 总步数 140 → lr 峰值只爬到 **28%**、全程还在爬升。
→ 改为 **25**。**规则：`warmup_steps` 必须 ≪ 总步数（≤10~20%），改 epoch/数据量后要重算。**

### ✅ 「每 epoch 28 步 vs 应为 25」疑点解开：benign

实测（`scripts/_diag_batchcount.py`）：`num_workers=0` → **25 批**；`num_workers=4` → **28 批**
（24×8 + **4×2** = 200 条，unique utt 200、重复 0）。是 DataLoader worker 分片各自组批所致，**无数据重复**。
→ 教训：**数 batch 数必须用与训练一致的 `num_workers`**。阶段 1 可放心上全量。

### 实验① CTC 对齐可视化（已完成）

`scripts/plot_ctc_alignment.py` → `work/aishell/exp/mid/ctc_alignment_BAC009S0724W0283.png`。
图上四件事：参考字峰值帧**严格从左到右递增**（单调对齐学到了）；大部分帧仍是 blank；
相对概率 0.10~0.16 ≈ 1/9（**基本分不清**，与 CER 72.7% 一致）；**argmax 选的字 ≠ 参考字**。
踩坑：`init_model()` 只做随机初始化，权重必须单独 `load_checkpoint()`；全词表 3591 类下必须按帧归一化才能出图。

### 方向调整（用户 16:26 决定）

**读代码 / 写代码权重降低，回到模型与算法**（代码基本由 AI 生成，面试不考"读过几行"）。
- 新学习载体：**实验 + 数学笔记**（CTC 对齐可视化 / chunk mask 热力图 / chunk 曲线 / 基线表 / ctc_weight 消融）
- 分工：**用户**提假设、定方向、读结果、追问为什么；**AI** 读代码、写代码、调试、跑实验
- 「手画 Conformer block」验收项改为交「数学公式 + 图」

### 全量数据已就位（但 list 还没重新生成）

`download_aishell.py --stage carlot-train` 完成：train **120,135** wav（16.19 GB，= 120,098 + 37 已知无标注）、dev 14,326 / test 7,176。
⚠️ **`data/train/data.list` 仍是 34,679** → 上全量前必须先重跑 `prepare_aishell.py --stage scp` + `--stage list` + `verify`。

---

## 既有资源盘点（开工前已就绪）

| 资源 | 位置 | 说明 |
|---|---|---|
| RKNN Model Zoo | `F:\embedded\rknn_model_zoo-main` | 含 **zipformer / whisper / wav2vec2 / yamnet / mms_tts** 示例，zipformer 与 whisper **官方支持 RK3576**，附 `build-linux.sh`（aarch64） |
| 论文库（42 篇） | `papers/` + `papers/README.md` | 分 6 类，标了 core/ext 优先级 |
| GPU | RTX 5060 8GB（sm_120） | 训练走 torch cu128 轮子 |
| CPU | 20 核 | 数据预处理可上多进程 |

**已确认的环境约束（踩过的坑，别再踩）**
- `git clone` 不可用 → 一律 `curl` 抓 tarball
- Bash 里任何命令前先 `export PATH="/usr/bin:/bin:$PATH"`
- 系统 exe（robocopy/tar）必须用绝对路径
- 给原生 exe 传路径用 `F:/...` 正斜杠，不要 `/f/...`
- `cmd.exe` 被禁 → 不要写依赖 cmd 的脚本
