# 进度追踪

> 规则：每天收工前追加一段。**不写"今天学了 XX"，写"今天产出了什么 / 卡在哪"。**
> 每天早上把本文件最新一段给我，我据此出当天安排。

---

## 总进度看板

| 阶段 | 周 | 状态 | 关键产物 |
|---|---|---|---|
| 0 地基 | W1–2 | 🟡 进行中（D1） | 环境 / Fbank / CTC 前向 |
| 1 复现 | W3–4 | ⬜ 未开始 | AISHELL-1 CER ≤ 6% + chunk 曲线 |
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
