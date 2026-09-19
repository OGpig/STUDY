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

