# 项目长期记忆 —— 语音算法实习冲刺（`F:\embedded\prepare`）

> 只放**跨天有效**的约定与事实。日常进度写 `PROGRESS.md`，日程写 `roadmap/DAYxx.md`。

## 用户偏好（硬约定）

- 🚫 **不要另建"没用的 md 文件"**。诊断/复盘/对拍结论**直接写进 `roadmap/DAYxx.md` 的对应任务段落**。
  `notes/` 只放用户明确要求交付的笔记（如 `notes/P01_e2e-survey.md`）。
- 用户要**自己手写算法**，AI 只做辅助：可以给方向、给对拍、给批改，**不要直接给完整实现**（红线写在 DAY01）。
- 用户会自己动手跑命令并把终端输出贴回来；回复要**"数字说话"**，少铺垫。
- 术语要精确——用户容易在方向过度延伸（把"局部因果"说成"全局更强"），批改时要**明确指出哪些是原文级命中**并说明依据页码。

## 项目主线

端到端中文流式 ASR：**Conformer + 上下文热词偏置**，PyTorch 训练 → 蒸馏/量化 → ONNX → RKNN → RK3576 Linux C++ 流式推理。
主线框架 **WeNet**，板端对照 **icefall/sherpa-onnx Zipformer**（`F:\embedded\rknn_model_zoo-main` 有官方 RK3576 示例）。

## WeNet 源码（已拉取并读完 transformer 目录）

- 落位：**`F:\embedded\prepare\examples\wenet-main\`**，无 `.git`（tarball）
- 模型目录是 **`wenet/models/transformer/`**（不是 `wenet/transformer/`）；
  `wenet/models/` 下有 14 个 backbone 变体目录，`transformer/` 只是主线 Conformer 那个
- ★★ **WeNet 主线没有模型侧上下文偏置**。只有解码期浅融合：
  `wenet/utils/context_graph.py` = 带 fail 弧的 Aho-Corasick trie，
  `recognize.py` 里 beam search 每步 `forward_one_step()` 查一次加分。
  → 「Conformer + 热词偏置」的**模型侧要自己写**（创新点空间 + W7-W8 的实际工作量）
- 流式三层机制：`utils/mask.py::subsequent_chunk_mask`（chunk mask，单位=编码器输出帧，已 ÷4）
  + 训练期 `use_dynamic_chunk`（一半概率全上下文 = U2 统一流式/非流式）
  + `att_cache` / `cnn_cache`；**subsampling 故意不做 cache**，靠输入重叠补右上下文
- `RelPositionMultiHeadedAttention` **注释掉了 `rel_shift`**（`attention.py:407-409`），
  理由：speech 里无用且流式难处理
- Conformer 层 macaron 的 `ff_scale = 0.5` 是**写死的**（`encoder_layer.py:176`），不是超参
- 类注册表在 `wenet/utils/class_utils.py`；`models/transformer/__init__.py` 是空文件
- 部署：`runtime/onnxruntime/` 是 RK3576 要走的后端

## 环境基线（唯一真相来源：`docs/ENV.md`）

- venv：`F:\embedded\prepare\.venv`，Python 3.13.14，**基于 WorkBuddy 独立解释器，已与 conda 脱钩**
- torch/torchaudio **2.11.0+cu128**，GPU **RTX 5060 capability (12,0)**（真 Blackwell 路径）
- 会话隔离用 `scripts/activate_project.ps1`；一键重建用 `scripts/setup_env.ps1`
- 自检：`scripts/test/test_env.py`（依赖 + venv 独立性 + GPU 真实算力）

## 本机操作约束（反复踩，务必遵守）

- Bash 工具跑任何命令**必须先 `export PATH="/usr/bin:/bin:$PATH"`**，否则 `ls`/`head`/`grep` 全部 127
- 系统 exe（`tar`/`robocopy`）用**绝对路径**；给原生 exe 传路径用 `F:/...` 正斜杠（`/f/...` 会被转成 `f:\f\...`）
- `git clone` 不可用 → `curl` 抓 tarball；`cmd.exe` 被安全策略禁用 → 不写依赖 cmd 的脚本
- 中文逻辑（含中文表头/中文注释）**一律写进 `.py` 文件再跑**，别用 `python -c '...'`（反斜杠转义会炸 SyntaxError）
- 跑测试统一加 `PYTHONIOENCODING=utf-8 PYTHONUTF8=1`

## 目录约定

- 🖼 **图统一放 `roadmap/assets/`**，在 `roadmap/DAYxx.md` 里用相对路径内嵌（`![](assets/xxx.svg)`）。
  优先 SVG（无依赖、可编辑、VSCode Markdown 预览直接渲染）。已产出：`DAY01_ctc-lattice.svg`。
- 诊断/复盘结论写 `roadmap/DAYxx.md`；进度写 `PROGRESS.md`；笔记只放用户点名要的（`notes/P01_*`）。
- 🧩 **算法刷题在 `coding/`**：题面 `problems/`、代码 `solutions/`、测试 `tests/*.in`、产物 `build/`（已 gitignore）。

## C++ 工具链（`coding/build.py` 一键驱动）

- 装了两套，`build.py` 自动探测 + 拼环境，**不需要 Developer Command Prompt、不经过 cmd.exe**：
  - **`gcc`（默认）= MinGW-W64 g++ 16.2.0**，`F:\data\toolchains\mingw64\bin`（含 gdb 17.2、mingw32-make）
  - `msvc` = cl.exe 14.51.36231 + Windows SDK 10.0.26100.0（VS 18 BuildTools）
- 用法：`python coding/build.py <源文件> [--cc msvc] [--dbg] [--no-run] [--in x.in] [--manual]`
  自动喂 `coding/tests/<源文件同名>_1.in`；超时 10 s；编译产物 → `coding/build/<名>.exe`
- `.vscode/` 放在**项目根**（用户实际打开的工作区层），`Ctrl+Shift+B` = 编译并运行
- ⚠️ **Git Bash 的 PATH 分隔符是 `:` 不是 `;`** → 在 bash 里拼 MSVC 的 PATH 必然失败。
  一律交给 Python（`os.pathsep`）拼。MSVC 版本号/SDK 版本号要 `glob` 自动探测，别硬编码。
- ⚠️ **`<VSCode>/bin/code` CLI 在本机不可用**（Git Bash 没有 `sh`；`code.cmd` 要 cmd.exe；
  `Code.exe --install-extension` 报 `bad option`）→ 扩展只能靠 `.vscode/extensions.json`
  的 recommendations 让用户手动一键装。
- ⚠️ **C++ 程序别打印中文**：Windows 控制台默认 GBK，C++ 写 UTF-8 中文会乱码
  （Python 不会，走宽字符 API）。中文标签全部由 `build.py` 打印，C++ 侧只用 ASCII。

## 审查用户代码的方法论（本次 P01/Fbank/CTC 都用的同一套）

1. **先跑，后读**。别只看代码，先拿真实数据跑一遍（含边界），再看代码解释现象。
2. **消融隔离**：只改可疑的一处、其余一行不动，验证"其余全对"。
   能把多个 bug 拆成互不干扰的独立结论（如 P01 的"返回值"vs"复杂度"两个 bug）。
3. **对拍要独立**：不用题面样例当唯一依据，自己写参考实现（Python/PyTorch）对拍，
   边界用例 ≥10 组。P01 用 Python `bisect` 版 LIS，10/10 通过。
4. **复杂度必须实测反推指数**，且**随机数据与有序/最坏数据必须分开测**：
   P01 的 DP 在随机输入下指数 ≈ 2.0（看不出问题），
   在**单调递增**输入下才暴露 **O(n³)**（n=8000 要 34 秒）→ 只测随机数据会漏掉。
   方法：多规模取最小值、用 log 比求指数。
5. **看退出码，不只看输出**。P01 的 n=0 场景：stdout/stderr 全部正常打印，
   但退出码是 **3221225477 = 0xC0000005**（段错误）。评测机上直接判 RE。
6. 编译必须查 `-Wall -Wextra` 的**警告条数**。
7. **复杂度必须实测，不能靠外推**。P01 实测：O(n²) 从 n=8000 外推到 n=100000 应 1.5 s，
   实测 9.67 s（慢 6.5 倍）—— 原因是工作集 96 KB → 1.2 MB，超出 L2 后内存带宽兜不住。
   **量级对了不代表常数因子对，缓存效应可以再差一个数量级。**
8. **写工具脚本时，自己的信息一律走 stderr，被测程序 stdout 用 `sys.stdout.buffer.write` 原始字节透传**
   （Windows 上绝不能 `sys.stdout.write(decoded_text)`，Python 会把 `\n` 再翻成 `\r\n` → `\r\r\n`）。
   这样 `工具 > out.txt` 才能直接和期望输出 diff。

### 可复用的领域教学点
- **DP 状态存"值"还是存"整条路径"，会把 O(n²) 变成 O(n³)** —— 每次"更优就整段拷贝 vector"
  的累计代价是 O(n³)。P01 实测：规模 8 倍 → 耗时 1685 倍（O(n²) 只该 64 倍）。
- **CTC 的 blank 约定**：`torch.nn.CTCLoss` 默认 `blank=0`，老实现常用 `blank=C-1`，混用会"看着对、全错"。
- **Kaldi Fbank**：mel 滤波器组必须在 **mel 域**算，绝不能把端点 floor 成 bin 索引再插值（低频整行塌陷）。

## 已核实的领域事实（避免重复踩坑）

- `torchaudio >= 2.9` 的 `torchaudio.load()` 需额外装 `torchcodec` → 本项目统一用 `soundfile` 读 wav
- `torch.max(Tensor, 标量)` 不接受 Python float → 用 `torch.clamp(x, min=eps)`
- Kaldi Fbank 真实顺序：**先分帧 → 每帧去直流 → 帧内预加重 → Povey 加窗 → 零填充到 512 → rfft → 功率谱 → mel 域滤波器组 → log(eps)**
- mel 滤波器组**必须在 mel 域算**（`clamp(min(up,down), 0)`），**绝不能**把端点 floor 成 bin 索引再插值（低频会整行塌陷）
- 权威参考实现就在本机：`F:\embedded\prepare\.venv\Lib\site-packages\torchaudio\compliance\kaldi.py`（对拍前先读它，别凭记忆）

### CTC（D1 已验证）

- `torch.nn.CTCLoss` 默认 **`blank=0`**（blank 占 0 号类别）；不少教科书/老实现用 `blank=C-1`。
  两者混用会「递推看起来完全正确、结果却全错」——**这是 CTC 实现的头号陷阱**。
- 目标序列可达性约束不是 `T >= L`，而是 **`T_min = L + 相邻重复标签对数`**（`[1,1,2,2]` 最少 6 帧）。
- 手写实现返回 `(N,)` loss 时，对拍必须用 `ctc_loss(..., reduction='none')`；默认 `'mean'` 返回标量，不能直接比。
- **通过边界用例 ≠ 实现正确**：`T==L`（路径不经过 blank）和「帧数不足」（双方都 +inf）
  在 blank 约定错误时也会 PASS。必须做随机 + 中大规模主对拍。

## 对拍工具的方法论（可复用到所有手写算法）

1. **先读参考实现的源码**，别凭记忆或博客。
2. **先做基线自检**：用「纯定义」的独立实现（如 CTC 用暴力枚举全部帧级路径）先与官方库对齐，
   证明对拍参照本身可信，再用同一比较器评选手写实现。否则无法排除"参照自己就错了"。
3. **做归因消融**：逐行复制犯错实现，只替换可疑组件，用「换成正确值后是否对齐」定位根因。
4. **主对拍用随机 + 中大规模 + 多组参数**，边界用例只作补充。
5. 临时脚本写系统 temp 并删掉，不污染项目目录。

已产出的对拍脚本：`scripts/test/compare_fbank.py`、`scripts/test/compare_ctc.py`

## 数据集获取：HF 镜像的瓶颈是「请求数」不是「带宽」（跨项目可复用）

同一镜像（hf-mirror.com）实测：单请求 455 MB → **11.1 MB/s**；
300 个 ~150 KB 小文件 → **0.16 MB/s（≈1 文件/秒）**，且**开 24 线程并发毫无改善**。
→ 镜像对小请求限流。**选数据源第一原则：能少发请求就少发请求。**
→ 旁证：同样 ~14 GB 数据，37 个请求要 30 分钟，122,000 个请求要 33 小时（差 66 倍）。
→ 对比：OpenSLR 官方 15.58 GB 单体包只有 0.59 MB/s（~7.3 h）→ 国内一律优先 hf-mirror。

## AISHELL-1 数据源（已定，别再重新调研）

| 源 | 内容 | 请求数 | 体积 | 耗时 |
|---|---|---|---|---|
| `AISHELL/AISHELL-1` | train 100/340 说话人（per-speaker tar.gz） | 100 | 3.45 GB | ~4 min |
| ★ `carlot/AIShell` | **全量官方划分（parquet）** | **37** | 20.3 GB | ~30 min |
| `shenyunhang/AISHELL-1` | 全量官方划分（逐个 wav） | 122,000 | ~14 GB | ~33 h（不可用） |
| OpenSLR 官方 | 全量官方 tgz | 1 | 15.58 GB | ~7.3 h |

- **carlot parquet 结构**：`audio: struct<bytes:binary, path:string>` + `transcription:string`。
  `bytes` = 原始 RIFF WAV 字节（与官方逐字节一致）；`path` = **原始 utt id**（`BAC009S0002W0122.wav`）。
  split：train 120,098 / **validation 14,326（= 官方 dev）** / test 7,176。分片 35/1/1。
  → **注意 HF 把官方 dev 叫 `validation`**。
- 转换用 pyarrow `iter_batches(batch_size=256)` 流式读，实测 **1902 条/秒**。
- 本机现状：`data/aishell/raw/wav/train/` 34,716 条（100 说话人）；
  transcript 141,600 行 ✅ 官方完整；lexicon 139,874 行。**dev/test 用 carlot 补**。
- 下载器：`scripts/download_aishell.py`（`--stage meta|train-tar|carlot-devtest|carlot-train`，可续传）。
- ⚠️ **必须用官方划分的 dev/test**，否则 CER 与文献不可比。

### AISHELL-1 数据质量已知缺陷（审计实测，别再重新发现）

审计工具：`scripts/check_aishell.py`（8 项检查）。

1. ★★ **train 的 wav 数 ≠ text 数，这是正常的，别去找 bug**
   `train: wav 文件 34,716 | 有文本 34,679 | 差 37`。
   根因：AISHELL-1 官方本身不对齐 —— **wav 目录共 141,925 个文件，transcript 只有 141,600 行**
   （官方 `aishell_data_prep.sh` 里 `[ $n -ne 141925 ]` 的警告就是说这个）。
   37 个 = **36 个正常音频（2.35~8.01s）但官方没给标注** + **1 个空音频**。
   → 官方用 `filter_scp.pl` **取交集** → **最终 train wav.scp 与 text 都是 34,679**。
   ✅ **可用样本数：train 34,679 / dev 14,326 / test 7,176。**
2. **1 个空音频**：`train/S0048/BAC009S0048W0490.wav` = 44 字节（WAV 头合法但 data 块 = 0）。
   不剔除会在训练**中途**崩（Fbank 得 0 帧），**启动时不报错**。取交集会自然排除。
3. ★ **carlot 的 dev/test 文本带统一的 4 个前导空格**（14,326 + 7,176 条 100% 都有）；
   官方 transcript 则带尾部空格。**去掉所有空白后两者逐字一致、长度差全为 0。**
   → 归一化必须用 `re.sub(r"\s+", "", t)`，**不能只 strip 首尾**，否则空格会变成空标签混进目标序列。
   💡 这类 bug 的特征是「每一条都错、错得一模一样」，抽查发现不了，只有全量交叉比对才行。
4. 其它体检项全绿：音频规格统一 (1ch/16000/16bit)；空文本 0；不在 lexicon 的字 0 种；
   train/dev/test 说话人两两无交叉（无泄漏）。时长：train 42.93h / dev 18.09h / test 10.03h。

**审计脚本自身的两个教训（都是"看着对其实漏了"）**：
- 第一版把「抽样 300 个文件的时长求和」当全量总时长打印（train 显示 0.4h，实际 42.93h）
  → **抽样只能验「规格统一」，总量统计必须全量**（可用 `(filesize-44)/(16000*2)` 换算）。
- 第一版 [1][2] 两项**跳过了 train**（因为 train 没有现成 text 文件），漏检了那 37 个
  → **检查项覆盖不全时不报错，只会给出一个"看起来对"的数字。**

## ★ 训练已跑通（2026-09-20）—— 一条命令 + 两个补丁

**最终指令**（不需要激活 venv / 设环境变量）：
```powershell
cd F:\embedded\prepare
.\.venv\Scripts\python.exe scripts\run_train.py --preset smoke
```
实测 `EXIT=0`，train loss 274.45→…→**70.34**，cv_loss 114.46→…→**89.88**。

### torch 2.11 Windows 轮子的 TCPStore/libuv bug（**报错信息是误导的**）

`use_libuv was requested but PyTorch was built without libuv support, run with USE_LIBUV=0`
—— **设 `USE_LIBUV=0` 完全无效**（实测）。C++ 层根本不读这个变量。
根因：torch 里 12 处 `TCPStore(` 调用有 **9 处**漏传 `use_libuv`，走 C++ 默认 True，
而 Windows 轮子没编 libuv。
→ `scripts/patch_torch_libuv.py` 括号配对扫描补参数（并跳过 docstring 示例）。
→ ⚠️ **`--standalone` 不是解法**（run.py 内部就设成 c10d）；`--rdzv_backend=static`
   能绕 elastic rendezvous 的 store，但 `init_process_group` 自己还会建一个 → 照样崩。

### WeNet main 与 torch 2.11 / 精简依赖不兼容（7 处）

`scripts/patch_wenet.py`（精确文本替换，匹配失败即报错）：
1. `squeezeformer/conv2d.py` 从 `torch.nn.modules.conv` import `Union`（2.11 不再带出）
2. `utils/train_utils.py` 顶层 `import deepspeed` + 3 子模块 → try/except 可选
3-4. `deepspeed.add_config_arguments` / `init_distributed` → 加可用性判断
5. ★ `utils/common.py` 顶层 `from whisper.tokenizer import LANGUAGES`
   —— **`common.py` 被整个模型栈依赖**（encoder.py → common.py）→ try/except
6-7. `utils/init_tokenizer.py` 顶层 import `WhisperTokenizer` → 改惰性
→ ④⑤⑥ 致命原因：`init_model.py` 会把**所有** backbone 都 import 一遍
  （哪怕只写 `encoder: conformer`）
→ 还要补装：`langid`、`tqdm`、`sentencepiece`、`tensorboardX`、`Pillow`、`textgrid`

### Windows 训练的两个操作坑

- **Git Bash 会改写 PATH 里的 `F:/...`**（`export PATH="F:/data/...:$PATH"` 变成
  `...PortableGit\versions\...\data\...`）→ 主进程和子进程都找不到 FFmpeg。
  → **必须在 Python 里设 `os.environ["PATH"]`**（不会被改写，子进程继承）。
- **`--num_workers 0` 撞 `prefetch_factor`**：WeNet 无条件传 `prefetch_factor=args.prefetch`，
  PyTorch 要求 `num_workers>0` 才允许 → 必须 ≥1。

### ⚠️ 未解决：CUDA 训练段错误

`--device cuda` → 第一个 batch 前向崩，退出码 **3221225477 = 0xC0000005**。
**已排除**：CUDA 基础算子（7 个全过）、**模型前向在 CUDA 上正常**（`model.to('cuda')`
跑 encoder+CTC 输出 `(2,49,256)`）、amp/fp16（根本没开）、worker 数（1 也崩）、
异步错误（`CUDA_LAUNCH_BLOCKING=1` 无信息）。
→ 下一个方向：**DataLoader worker 产出张量 → `.to('cuda')`** 这条路径。
**CPU 训练可用**，但阶段 1 训 12 万条必须在 W3 前解决。

### ✅ GPU 训练已跑通（2026-09-20）—— 根因：Windows gloo 是 CPU-only，DDP 无条件包

`--device cuda` 训练崩（退出码 3221225477 = 0xC0000005，Python 层无异常）。
**根因**：Windows 轮子无 NCCL（`is_nccl_available()==False`）→ 只能用 gloo，
而 **gloo 在 Windows 上是 CPU-only 构建**，DDP 反向要 allreduce **CUDA** 梯度 → 段错误。
WeNet 的 `wrap_cuda_model` 里 DDP **没有 world_size 判断，单卡也照包**。

→ `patch_wenet.py` 第 8 处：`world_size == 1` 时跳过 DDP
（安全性：全仓库无一处用 `.module`，`save_model` 走 `state_dict()`）。
实测 `EXIT=0`，cv_loss 114.38 → **89.88**，**14.4 steps/sec（CPU 0.55，快 26 倍）**。

**⚠️ 长远影响**：Windows 只有 gloo → **DDP 多卡在 Windows 上走不通**，多卡要上 Linux/WSL。
单卡 RTX 5060 训 AISHELL-1 子集够用。这条直接影响阶段 1 的规模化训练计划。

**定位段错误类问题的消融顺序（可复用）**：
① 基础算子逐个数 → ② 模型单独跑 → ③ 补上训练里实际存在但②缺的步骤（set_device / init_process_group）
→ ④ 读包装层源码 → ⑤ **「裸 vs 包装」最小对照**（`nn.Linear` 裸跑 vs 包 DDP）。
⚠️ 取退出码别用 `cmd | tail; echo $?`（那是 tail 的码），要 `cmd > log 2>&1; echo $?`。

## ★ `final.pt` 是断链（WeNet 自身 bug）—— 别被 `ls` 骗了

`wenet/bin/train.py` 收尾有两个 bug：
1. 清理用 `os.path.exists` —— **对断链返回 False**（会 follow 链接）→ 清理被跳过 → 同目录重跑必然 `FileExistsError`
2. 链接目标写成 `'4.pt'`，而 `save_model` 存的是 `'epoch_4.pt'` → **`final.pt` 天生就是断链**

⚠️ **`run.sh` stage 5 用 `decode_checkpoint=$dir/final.pt` 解码 → 断链会直接失败。**
⚠️ **我此前几次"✅ final.pt 产出"的检查全是假阳性**：`ls` 能看到名字，但它是断链。
→ **检查文件必须 `islink()` + `exists()` 两个都看；`lexists` 只说明"有个条目"。**

已修（`patch_wenet.py` 第 25 处）：`lexists` + 正确目标名 + `os.rename` 兜底（断链删除实测会报
WinError 5）+ 无符号链接权限时退化为 `shutil.copy2`。
修复后连跑两次 `EXIT=0`，`final.pt` 指向 `epoch_4.pt` 且目标存在。

**通用教训**：`os.path.exists` vs `lexists` 的差异只在"目标是断链"时暴露；
`os.remove(...) if os.path.exists(...) else None` 这种清理写法在断链下**静默失效**。

## ★ 编码导致「静默丢样本」（2026-09-20，最隐蔽的一类 bug）

**症状伪装**：不崩溃、不报错，只在很远的地方表现为
`ZeroDivisionError: division by zero`（在 `executor.cv`）。

**两个成因，症状完全相同**：
1. `wenet/dataset/datapipes.py:353 TextLineDataPipe` 用 `FileOpener(..., mode=mode)` 打开
   `data.list`，**encoding 默认 None = 系统编码（GBK）** → 含中文的行（`txt` 字段，我方用
   `ensure_ascii=False` 生成）解码失败 → 外面包着 `map_ignore_error` → **样本被静默全丢**
   → dataloader 出 0 batch → `sum(total_acc)/len(total_acc)` 除零。
2. **FFmpeg 找不到**时 `decode_wav` 里的 `torchaudio.load` 失败 —— 同样在 `map_ignore_error` 里，
   **同样表现为 ZeroDivisionError**。

> ### 🔍 排查口诀
> 看到 `executor.cv` 里的 `ZeroDivisionError` → **先查 ① 文件编码 ② FFmpeg**，别读 cv 的代码。

**已修**：`patch_wenet.py` 第 10 处给 `map_ignore_error` 补打**完整 traceback**
（原版只打一行 `str(ex)`，丢样本时看不出哪一步失败）；第 11 处 `FileOpener` 强制 utf-8；
第 12~24 处 13 个文件的 yaml 读写统一加 `encoding='utf-8'`。

## FFmpeg DLL：用 `os.add_dll_directory` 而不是 PATH（2026-09-20）

`scripts/install_env_fix.py` 在 venv 的 site-packages 放 `.pth`
（内容 = 「把 `<项目>/scripts/pyfix` 加进 sys.path」+「import wenet_dll_fix」），
模块里用 **`os.add_dll_directory()`**（CPython 3.8+ 官方 API）注册 FFmpeg 目录。

- **不依赖 PATH** → 不受 Git Bash 改写 `F:/...` 的影响
- `.pth` 让**每个进程（含 spawn worker）**自动生效，零环境变量
- 不 import torch，启动开销几乎为零
- 实测：全新进程 + 零环境变量 → `torchaudio.load` → `(1, 95984) @16000Hz` ✅

## 我踩过的三个「实验方法」坑（比结论值钱）

1. **检查太弱造成假阳性**：用 `'ffmpeg-shared' in PATH` 判断可用性，但 Git Bash 改写后的路径
   **仍含该子串**（目录不存在）→ 误判。**检查路径必须 `os.path.isdir()` 实测。**
2. **变量转义写错**：`export PATH="\$FF;$PATH"` 里 `\$FF` 是**字面量** → 整轮 A/B/C/D 实验白跑。
   **跑实验前先验证前提成立。**
3. **heredoc 不能做 multiprocessing 测试**：Windows spawn 需重新 import 主模块，
   `<stdin>` 报 `OSError: Invalid argument` → 必须写成真实文件。
4. 取退出码：`cmd | tail; echo $?` 取到的是 **tail** 的码（会骗你说没崩，真实是 139）
   → 必须 `cmd > log 2>&1; echo $?`。

## WeNet 训练在 Windows 上的坑（D2 挖出）

0. ★★ **torchaudio 2.11 解码音频必须要有 FFmpeg 共享库**（最容易卡死训练的一条）
   `torchaudio.load()` 报 `Could not load libtorchcodec`；`torchaudio.info()` 在 2.11 **已被删除**。
   根因：torchaudio ≥2.9 解码改依赖 **torchcodec**，而 torchcodec **不捆绑 FFmpeg 共享库**
   （只捆绑 libwebp/libavif/zstd）。`pip install torchcodec` 成功也没用。
   → **WeNet 的 `processor.py:156` 就是 `torchaudio.load`，它挂了训练直接起不来**，
     而报错与"数据准备"毫无关系，极难查。
   → 解决：装 FFmpeg **shared 构建**，bin 挂 PATH。本机位置：
     `F:\data\toolchains\ffmpeg-shared\ffmpeg-master-latest-win64-gpl-shared\bin`
     （github 直连不通，走 `ghproxy.net` 镜像 + `curl -C -` 续传；解压用 Python zipfile）
   → **WeNet 源码一行不用改**。`prepare_aishell.py::ensure_ffmpeg_on_path()` 会自动挂上。
   → 连带：`tools/compute_cmvn_stats.py` 同时用 `torchaudio.info`+`load`，**即使有 FFmpeg 也跑不了**
     → CMVN 必须自己写（`soundfile` 读 + `kaldi.fbank` 算 + 多进程；实测 34,679 条 / 65 s）。
1. **`tools` / `wenet` 软链被 tar.exe 解成了普通文本文件**（内容就是路径字符串）→
   `examples/aishell/s0/` 下官方 `run.sh`（`. ./path.sh` + `local/*.sh`）整条不可用。一切走绝对路径绕开。
2. **新版 `train.py` 没有「不开分布式」的开关**：`init_distributed(args)` 是必经路径
   → 单卡也必须 `torchrun --nproc_per_node=1 --rdzv_backend=c10d`。
3. **Windows 无 NCCL** → recipe 的 `dist_backend=nccl` 必须改 **`gloo`**。
4. ★ **`warmup_steps=25000` 与小数据量严重不匹配**：200 条/batch 8 = 每 epoch 25 步，
   5 epoch 共 125 步 → lr 全程停在爬升底部，**loss 是平线**，极易误判成"模型坏了"。
   冒烟阶段调到 500。**本质：scheduler 时间尺度必须匹配数据规模。**
5. 流式主线配置是 `train_unified_conformer.yaml`（多 `causal:true` / `use_dynamic_chunk:true` /
   `cnn_module_norm:layer_norm`，`accum_grad:1`、`max_epoch:180`、`lr:0.001`）。
   WeNet 新版 yaml 结构：顶层有 `tokenizer` / `cmvn` / `dataset` / `ctc` / `model` 等键。
6. **`data.list` 只有 `{"key","wav","txt"}` 三个字段**（新版 `make_raw_list.py`），
   **没有 duration / sample_rate** —— 时长由 dataset 运行时算。别自己加。
7. **`yaml` 不是自带依赖**，但 WeNet 读配置必需 → `pip install pyyaml`。
8. 新依赖 **pyarrow**（读 parquet）。数据集目录 `data/` 已在 `.gitignore`。
   `.gitattributes` 强制 LF。训练 CWD 必须是 `work/aishell/`（yaml 里是相对路径）。

## 数据准备流水线（D2 落地）

- **一条命令**：`cd F:\embedded\prepare` 然后
  `.\.venv\Scripts\python.exe scripts\prepare_aishell.py --stage all`
  —— 不需要激活 venv / 设环境变量（脚本自己找 FFmpeg）。
- 产物在 **`work/aishell/data/`**（刻意对齐 WeNet recipe 布局，yaml 相对路径直接可用）
- 实测：train **34,679** / dev 14,326 / test 7,176；dict 3,590 行；
  global_cmvn 80 维 frame_num=15,367,984（≈42.7h）；train_smoke 200 / dev_smoke 50
- 三处与官方不同：① 文本统一用官方 transcript（carlot 只做交叉验证）
  ② 归一化删**所有**空白（不只 strip）③ 显式剔除空音频并打日志

## PowerShell 5.1 的两个坑（写 .ps1 必踩）

- **`$ErrorActionPreference = "Stop"` 会把原生命令写到 stderr 的每一行当成终止性错误。**
  本项目 Python 脚本按设计把信息写 stderr → 脚本会在打印第一行后中止，Python 根本没跑。
  → 调原生命令前降回 `"Continue"`。
- **无 BOM 的 UTF-8 `.ps1` 被当 GBK 读**，中文错位吃掉引号 → PSParser 报"字符串缺少终止符"。
  → 含中文的 .ps1 **必须存成 UTF-8 with BOM**。
  校验：`[System.Management.Automation.PSParser]::Tokenize((Get-Content -Raw $p),[ref]$errs)`
- ⚠️ **PowerShell 工具在本机是沙箱化的**：`*>` 重定向的写入与子进程副作用会丢失，
  却仍返回退出码 0。**验证副作用一律用 Bash 工具查文件系统，别信 PowerShell 的退出码。**

