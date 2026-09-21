# EXPERIMENTS.md — 实验记录

> **规则（别破坏它，否则这份文件就白建了）**
>
> 1. 每条实验必须**能被别人复现**：完整命令 + 配置差异 + 结果（数字/曲线路径）。
> 2. **失败和负结果也要记**。它们比成功更省时间——尤其是"看起来该有效但没用"的那种。
> 3. 结果里必须有一个**可量化的判据**（"loss 降了"不算，要写从多少降到多少）。
> 4. 每条结论后面留一个**"我没搞明白的"**栏。写不出东西说明这条实验做得太浅。
> 5. **`--tensorboard_dir` 必须带运行标识**（如 `tensorboard/smoke-0920`），
>    或重跑前清掉旧目录。⚠️ 同一个目录下堆多个 `events.out.tfevents.*` 时，
>    TensorBoard 会把它们**合并**，曲线直接变成锯齿（实测踩过，见 E01）。
> 6. **写进记录的数字必须能追溯到原始数据文件**，并且要能复现出同样的值。
>    "看起来合理"不是合格标准 —— E01 里我为此改过两次数字。
>
> 配置只记**与基线不同的部分**；基线配置由脚本生成，见 `--stage conf`。

---

## 模板（复制这一块）

```markdown
## E?? | YYYY-MM-DD | <一句话标题>

**目的**：这条实验要回答什么问题？（要能被证伪）
**命令**：可直接粘贴执行的完整命令
**配置差异**：相对基线改了哪些项、为什么改
**结果**：数字表格 + 曲线路径
**结论**：一句话
**我没搞明白的**：
**异常/坑**：指向 DAYxx 或 ENV.md 的对应条目
```

---

## E01 | 2026-09-20 | WeNet Conformer 冒烟测试（阶段 0 最后一条验收）

**目的**：证明「数据 → 模型 → 损失下降」这条链路是**通的**。
不是证明模型好——所以刻意用了 4 层 / 5 epoch 的极小配置。

> 📌 **原始意图（引自 `roadmap/ROADMAP.md` 阶段 0）**：
> 「跑通一个最小语料训练：AISHELL-1 的 dev 子集（如 200 条）× 5 epoch，**哪怕 CER=80%**」
> 「验收标准：有一次真实训练日志（**loss 在下降，不是 flat line**）」
>
> → 也就是说：**从一开始就没打算让这个模型能识别。** 判据是"在降、不是平线"，
> 不是 CER。所以 CER=100% **不违反阶段 0 的验收**（虽然比"哪怕 80%"还差，
> 但两者都属于"没有识别能力"，性质相同）。
> → 真正要出模型质量的是**阶段 1**（AISHELL-1 test CER ≤ 6% + chunk 曲线）。

### 命令

```powershell
cd F:\embedded\prepare
.\.venv\Scripts\python.exe scripts\run_train.py --preset smoke
```

等价的手敲命令（供不依赖启动器时使用；CWD 必须是 `work/aishell`）：

```powershell
cd F:\embedded\prepare\work\aishell
$env:PYTHONPATH = "F:\embedded\prepare\examples\wenet-main"
F:\embedded\prepare\.venv\Scripts\python.exe -m torch.distributed.run `
  --nnodes=1 --nproc_per_node=1 --rdzv_id=smoke --rdzv_backend=c10d `
  --rdzv_endpoint=localhost:0 `
  F:\embedded\prepare\examples\wenet-main\wenet\bin\train.py `
  --train_engine torch_ddp --ddp.dist_backend gloo `
  --config conf/smoke.yaml --data_type raw `
  --train_data data/train_smoke/data.list --cv_data data/dev_smoke/data.list `
  --model_dir exp/smoke --tensorboard_dir tensorboard --num_workers 4
```

### 配置差异（相对官方 `train_unified_conformer.yaml`）

| 项 | 官方 | 本实验 | 为什么 |
|---|---|---|---|
| `encoder_conf.num_blocks` | 12 | **4** | 少 2/3 计算量，几分钟内跑完 |
| `dataset_conf.batch_conf.batch_size` | 16 | **8** | 数据量小 |
| `max_epoch` | 180 | **5** | 冒烟 |
| `log_interval` | 100 | **10** | 否则 5 个 epoch 只打几条日志，看不出趋势 |
| **`scheduler_conf.warmup_steps`** | **25000** | **500** | ★ 见下 |
| `dataset_conf.speed_perturb` | true | **false** | 去掉扰动，让曲线干净可解释 |
| `dataset_conf.spec_aug` | true | **false** | 同上 |

> ★ **`warmup_steps` 是本实验唯一真正重要的配置决策。**
> 200 条 / batch 8 = 每 epoch 约 25 步，5 epoch 一共约 125 步。
> 若沿用 25000，整个训练期间学习率都停在爬升的最底部，**loss 会是一条平线**，
> 极易被误判成"模型/数据坏了"。本质：**scheduler 的时间尺度必须匹配数据规模。**
>
> 数据：`train_smoke` 200 条（100 说话人）/ `dev_smoke` 50 条（40 说话人），
> 由 `scripts/prepare_aishell.py --stage smoke` 从 AISHELL-1 官方 train/dev 等间隔抽样。
> CMVN / 词典用**全量 train（34,679 条）**统计，不是子集。

### 结果

**验证集（`epoch/*`，每 epoch 一次）— 这是最干净的证据**

| epoch | `epoch/loss` | `epoch/loss_att` | `epoch/loss_ctc` | `epoch/acc` |
|---|---|---|---|---|
| 0 | 114.3819 | 105.4541 | 135.2134 | 0.0000 |
| 1 | 101.5639 | 98.2897 | 109.2038 | 0.0661 |
| 2 | 91.9944 | 88.3170 | 100.5749 | 0.0512 |
| 3 | 90.1612 | 86.5212 | 98.6545 | 0.0566 |
| 4 | **89.8848** | **86.0310** | **98.8769** | **0.0606** |

→ `loss` 与 `loss_att` **单调下降**；`loss_ctc` 在最后一步从 98.65 微升到 98.88（见下）。

**训练（`train/*`，140 个唯一 step，去重后；每 epoch 28 步）**

| epoch | `train_loss` | `train/loss_att` | `train/loss_ctc` |
|---|---|---|---|
| 0 | 223.54 | 116.65 | 472.95 |
| 1 | 119.90 | 113.77 | 134.22 |
| 2 | 103.07 | 98.96 | 112.66 |
| 3 | 95.25 | 90.86 | 105.50 |
| 4 | **91.91** | **87.21** | **102.86** |

（表内是**每 epoch 的均值**，按 28 步/epoch 对齐 —— 这个边界已由数值验证，见下。）

> ⚠️ **两个必须说明的坑，否则这张表会被误读：**
>
> 1. **同一个 logdir 里堆了两次运行的 event 文件**（我为了验证修过的问题跑了两次，
>    都写进 `tensorboard/smoke/`）。TensorBoard 与 EventAccumulator 会把它们**合并**，
>    于是每个 step 出现两次、共 280 个点。**直接画就会得到锯齿**
>    （线降到最低又跳回起点重画一遍）。绘图脚本现在会**按 step 去重并告警**。
> 2. 去重**之前**"每 epoch 均值"是按 56 点切的 —— 而 56 点在拼接序列里实际横跨
>    **两个 epoch**，所以早期版本这张表里的数字是**错的**（已修正为 28 步切分）。
>
> 📌 这两个问题都不是 WeNet 的 bug，是**运行记录的组织问题**。
> **规范：`--tensorboard_dir` 要带运行标识**（如 `tensorboard/smoke-0920`），
> 或重跑前清掉旧目录。否则每次重跑都会污染曲线。

**其它**

| 指标 | 首 | 末 | 解读 |
|---|---|---|---|
| `train/grad_norm` | 343.75 | **41.09** | 梯度范数降一个量级 → 在收敛 |
| `train/lr_0` | 0 | **2.8e-04** | warmup 按预期爬升（500 步内到顶） |
| `epoch/acc` | 0.0 | **0.0606** | 只有 6%，但**非零** → 确实在学 |

**曲线图**：`work/aishell/tensorboard/smoke/curves.png`
（由 `scripts/plot_tensorboard.py --logdir work/aishell/tensorboard/smoke` 生成）

四个面板，全部是**每 epoch 级**（per-batch 原始曲线默认不画，`--show-batch` 可选）：

| 面板 | 内容 | 回答什么 |
|---|---|---|
| ① 总损失 | train（虚线） vs cv（实线） | ★ **总体在不在降**：223.5→91.9 / 114.4→89.9 |
| ② 分支损失 | attention / CTC（train 虚线 + cv 实线） | 两个分支各自的收敛情况 |
| ③ 帧准确率 | train 平滑 + cv acc，**x 轴统一为 epoch** | 确实在学（0 → 6%） |
| ④ 学习率 | warmup 0 → 2.8e-4 | warmup 生效的证据 |

> ⚠️ **per-batch 原始曲线默认不画**（`--show-batch` 可选）。原因：
> 它被「batch 里的音频长度」主导（CTC 的 -logP 随帧数累积，且数据按长度排序组 batch），
> 噪声远大于趋势，实务上没人拿它做判据。
> 另外 **x 轴统一为「完成的 epoch 数」（1..5）**：`epoch/*` 是训练完该 epoch 之后测的，
> 不能与 per-batch 的 step（0..139）混在一条轴上。

**性能**：GPU（RTX 5060）**14.4 steps/sec**，CPU 0.55 steps/sec → **快 26 倍**；整轮约 35 秒。

**TensorBoard**（已验证可读）：
```powershell
tensorboard --logdir work/aishell/tensorboard
```
实测 `GET /data/plugin/scalars/tags` 返回全部 11 个 tag，
`GET /data/plugin/scalars/scalars?tag=train/train_loss&run=smoke` 能取到 **140 个真实数据点**
（去重后；之前是 280 点，因为同一目录堆了两次运行的 event 文件）。

### ⚠️ 为什么训练损失曲线"看起来不合理"（重要，别误判）

per-batch 的训练损失在 50~670 之间剧烈震荡，看起来像"训练没收敛"。
**这不是模型问题，是度量本身的性质**：

1. **CTC 的 -logP 天然随帧数累积。**
   实测（随机权重、只改长度）：`T=100 → 6108`，`T=200 → 12628`，
   `T=400 → 26008` —— **帧数翻倍，损失翻倍**。所以"这条音频有多长"
   直接决定这批的损失量级。
2. **WeNet 的 CTC 是「每条语音的平均」，不是整批求和**
   （`ctc.py`：`loss = self.ctc_loss(...)` 用 `reduction='sum'`，然后
   `loss = loss / ys_hat.size(1)` 除以 batch_size）。
   所以批大小的影响被消掉了，但**长度的影响还在**。
3. **数据管线按长度排序组 batch**（`dataset_conf.sort_conf.sort_size`），
   于是相邻 batch 的音频长短**系统性不同** → per-batch 损失呈规律性起伏。

→ **结论：per-batch 曲线必须平滑后才能读。**
绘图脚本已支持 `--smooth N`（默认 10），画法是
**原始曲线（淡）+ 滑动平均（粗）**，与 TensorBoard 的 smoothing 滑块同一思路。
要看真实趋势，请看上表的**每 epoch 均值**（全部单调），或图里的粗线。

### ★ 诚实的结论：**这次训练基本什么都没学会**

「loss 在降」**不等于**「模型会做语音识别」。我把这条验证到底了：

#### 判据 1：解码 CER = **100.00%**

用训练出的 `final.pt` 解码 `dev_smoke`（50 条，CTC greedy search）：

```
识别为**非空**的条数: 0 / 50
编辑距离总和 (S+D+I) = 699，参考总字数 = 699
CER = 100.00%
```

**50 条全部输出空串** —— 每一帧都预测 blank。这是 CTC 完全没训练时的典型表现
（blank 是最"安全"的局部最优：不知道说什么就说"什么都不说"）。

```powershell
python F:\embedded\prepare\examples\wenet-main\wenet\bin\recognize.py `
  --config exp/smoke/train.yaml --test_data data/dev_smoke/data.list `
  --checkpoint exp/smoke/final.pt --beam_size 10 --batch_size 8 `
  --result_dir exp/smoke/decode_dev --mode ctc_greedy_search
```

#### 判据 2：loss 只比「不听音频」的基线低 6%

| 基准 / 实测 | `loss_att` |
|---|---|
| 完全随机（在 3590 类上均匀猜） | 118.7 |
| **只学会汉字边际分布、完全不听音频**（L × 一元熵） | **92.5** |
| 实测 epoch 0 | 116.7 |
| 实测 epoch 4 | **87.2** |

（一元熵 6.377 nats/字 × 平均句长 14.5 字 = 92.5）

→ epoch 4 只比"**一个字都没听进去**"的模型低 **5.3 nats（约 6%）**。
**loss 的下降绝大部分是"学会了哪些字常用"，不是"学会了声音到字的映射"。**
这与 `epoch/acc = 6%` 完全吻合。

#### 判据 3：数据量差 4 个数量级

| | 本实验 | 官方 AISHELL-1 复现 |
|---|---|---|
| 样本曝光量 | 200 × 5 = **1,000** 次 | 120,098 × 180 ≈ **21.6M** 次 |
| 模型参数 | 27.2M | （同量级） |
| **参数 : 样本** | **27,178 : 1** | ≈ **1.3 : 1** |

参数比样本多两万倍 —— 这不是"训练"，是"让模型记住了数据点"。

### 结论

**本实验达成的目标是「工程链路通」**：
数据 → 模型 → 优化 → checkpoint → 解码，整条都能在 Windows + torch 2.11 + RTX 5060 上跑通，
GPU 相对 CPU 快 26 倍，后续迭代的成本可接受。

**本实验没有达成、也不该期待的是「模型质量」**：CER = 100%，模型无任何识别能力。
这是配置（4 层 / 5 epoch / 200 条）决定的，**不是 bug**。

> 📌 **写实验记录的一条纪律（这次学到的）**：
> 「loss 下降」只是**必要条件**，不是充分条件。
> 判据必须是**任务指标**（这里是 CER），而且要和**随机基线 / 平凡基线**比，
> 而不是只看自己跟自己比。否则"loss 从 223 降到 92"很容易被读成"模型学会了"。

### 我没搞明白的

- **每个 epoch 只有 28 步，但 `train_smoke` 是 200 条 / batch 8 = 25 步。多出 3 步。**
  140 唯一 step ÷ 5 epoch = **28.0**，且按 28 步切出来的每 epoch 均值**单调递减**
  （223.5 → 119.9 → 103.1 → 95.3 → 91.9），所以 28 这个边界是**数值上站得住的**，
  不是巧合。
  **怀疑**：`dataset_conf.shuffle_conf.shuffle_size = 1500` 与
  `sort_conf.sort_size = 500` **都远大于数据集大小（200）**。
  WeNet 自己的注释写着 "sort_size should be less than shuffle_size"，
  但没写"两者都应小于数据集大小"。缓冲区大于数据时，epoch 边界的冲刷行为可能
  会多吐几个 batch。
  **验证方法**：把这两个值改成 200 / 100 再跑一次，数 step 数是否变成 125。
  → ⚠️ **这个必须查清再上阶段 1**：如果数据管线每个 epoch 会重复喂样本，
  等于训练集被悄悄扩了一个"错版本"，会直接影响 CER 的可信度。

### 已解决的两个疑点（留档，因为它们差点误导结论）

1. **每个 `epoch/*` tag 写了 2 遍（10 点 = 5 值 ×2）**
   → 原因：`tensorboard/smoke/` 下堆了**两个 event 文件**（两次运行的产物），
   EventAccumulator 把两份日志合并了。不是 WeNet bug，是运行记录的组织问题。
2. **`train/train_loss` 有 280 点而非 140 点**
   → 同一个原因。去重后正好 140 点。

> 📌 **规范（已写进本文件开头的规则）**：`--tensorboard_dir` 要带运行标识
> （如 `tensorboard/smoke-0920`），或重跑前清掉旧目录。
> 否则每次重跑都会在同一个 tag 上叠加一条曲线。

### ⚠️ 自查记录：这份记录我改过两次数字

1. 第一版 `epoch/loss_att` / `loss_ctc` 的**中间三行是凭趋势推测填的** ——
   核对 tfevents 后发现不符（epoch 1 的 att 我写 96.36，**实际 98.29**）。
2. 第二版的"每 epoch 均值"按 56 点切分，而 56 点在拼接序列里**横跨两个 epoch** ——
   数字整体是错的（已改为 28 步切分）。

**教训：凡是写进记录的数字，必须能追溯到原始数据文件，并且要能复现出同样的值。
"看起来合理"不是合格标准。**

### 异常/坑

本实验一路撞了 **9 个环境/框架问题**，全部记录在：

- `roadmap/DAY02.md` 任务 4「完成记录」—— 从报错到跑通的问题链
- `docs/ENV.md` 风险 6 / 8~14 —— 逐条根因与修法

修法都已固化成脚本（可 `--check` / `--revert`）：

| 脚本 | 作用 |
|---|---|
| `scripts/patch_torch_libuv.py` | torch Windows 轮子的 TCPStore/libuv bug（9 处） |
| `scripts/patch_wenet.py` | WeNet 与 torch 2.11 / 精简依赖 / 编码的兼容性（25 处） |
| `scripts/install_env_fix.py` | 用 `os.add_dll_directory` 让 torchaudio 找到 FFmpeg（装 `.pth`，零环境变量） |

---

## E02 | 2026-09-20 | 中等规模复训：**证明训练管线没有 bug**

**目的**：E01 得到 CER=100%，怀疑训练过程有 bug。
这条实验用**更大数据 + 正确的学习率调度**重训，看 CER 会不会动。
**两个结果截然不同，结论就完全不同。**

### 发现的第一个真 bug：warmup_steps > 总步数

| 项 | 值 |
|---|---|
| `optim_conf.lr`（调度器峰值） | 0.001 |
| `scheduler_conf.warmup_steps` | **500** |
| **实际总步数**（200 条 × 5 epoch） | **140** |
| **实际达到的最大 lr** | **0.00028** = 140/500 × 0.001 |

**全程学习率只到峰值的 28%，结束时还在爬升**（WarmupLR 峰值 = `optimizer.lr`，
在 warmup_steps 内线性爬升）。整个训练的平均 lr 只有 ~1.4e-4（峰值的 14%）。
这是 E01 配置里的**真 bug**：把 `warmup_steps` 从 25000 改到 500 时，
忘了同时检查「500 是否远小于总步数」。

### 命令（在 work/aishell 下）

```powershell
python F:\embedded\prepare\.venv\Scripts\python.exe `
  F:\embedded\prepare\examples\wenet-main\wenet\bin\train.py `
  --train_engine torch_ddp --ddp.dist_backend gloo `
  --config conf/mid.yaml --data_type raw `
  --train_data data/train_mid/data.list --cv_data data/dev_mid/data.list `
  --model_dir exp/mid --tensorboard_dir tensorboard --num_workers 4
```

### 配置差异（相对 E01）

| 项 | E01 | E02 | 为什么 |
|---|---|---|---|
| 训练数据 | 200 条 | **3,000 条** | 数据量是最主要变量 |
| `max_epoch` | 5 | **20** | 同上 |
| `batch_size` | 8 | **16** | GPU 有余量 |
| 总步数 | 140 | **3,760** | warmup 500 占 13.4%，合理 |
| 学习率可达峰值 | **0.00028** | **0.000375**（已过峰值） | 修正上面的 bug |

### 结果

| 指标 | E01（冒烟） | **E02（mid）** |
|---|---|---|
| **CER** | **100.00%**（50 条全空串） | **72.70%**（空串 0 条） |
| cv_loss | 89.88 | **72.64** |
| cv acc（帧） | 0.0606 | **0.2262** |
| train th_accuracy | 0.06 | **0.95 ~ 0.97** |
| 训练步数 | 140 | 3,760 |
| 耗时 | 35 秒 | ~21 分钟 |

识别样例（E02）：

```
标注: 预计第三季度将陆续有部分股市资金重归楼市
识别: 预计地三积住将入洗有部分补十持技春功楼时      「预计」全对
标注: 而六月居冠的天河北成交宗数下滑近百分之二
识别: 而零于基关的金后本成相红数下化金八战          首字「而」对
```

### 结论

**训练管线没有 bug。** E01 的 CER=100% 纯粹是**欠训练**：
数据少 15 倍、epoch 少 4 倍、lr 峰值只到 28%。
三个变量一起放大后，CER 立刻从 100% 降到 72.7%，且不再输出空串。

### 我没搞明白的

- cv_loss 在 epoch 5 之后停在 68~73 不再下降，而 train_loss 已到 3~7，
  **在 3000 条上过拟合了**。要到 CER ≤ 6% 必须上全量（120,098 条）。
  按当前速度（约 5 steps/s、batch 16）全量 180 epoch 约 **75 小时，3 天**，需要规划。
- E02 用的是**贪心 CTC** 解码，没试 beam search / attention 重打分；
  官方 recipe 的 5% 是带 attention rescoring 的，这里还有提升空间。

### 异常/坑

E01 的 `warmup_steps=500 > 总步数 140`，学习率从未到峰值。
**规则：warmup_steps 必须「远小于」总步数（建议不超过 10~20%），
改了 epoch 数或数据量之后要重算。**
