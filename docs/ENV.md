# 环境基线（ENV）

> 本文件是**唯一的真相来源**：任何人在任何一天都能照着这里复现出完全一样的环境。
> 环境变了（升级/换版本）就更新这里，别只在脑子里记。

---

## 一、当前基线（2026-09-18 重建后实测；2026-09-20 补充）

| 项 | 值 |
|---|---|
| 项目 venv | `F:\embedded\prepare\.venv` |
| **venv 基础解释器** | `C:\Users\ADMIN\.workbuddy\binaries\python\versions\3.13.12`（**独立 Python，非 conda**） |
| 实际 Python 版本 | 3.13.14 |
| pip | **26.2.1**（必须 ≥ 26.2.1，见风险 2） |
| PyTorch | **2.11.0+cu128** |
| torchaudio | 2.11.0+cu128 |
| GPU | NVIDIA RTX 5060 8GB（sm_120 / Blackwell） |
| 其他 | numpy 2.5.3 / pandas 3.0.6 / scipy 1.18.1 / librosa 1.0.0 / numba 0.67.0 / tensorboard 2.21.0 / soundfile 0.14.0 |
| **D2 新增** | **pyyaml 6.0.3**（WeNet 读 yaml 配置必需）、**pyarrow 25.0.1**（读 carlot parquet）、torchcodec 0.16.0+cpu |
| ★ **FFmpeg 共享库** | `F:\data\toolchains\ffmpeg-shared\ffmpeg-master-latest-win64-gpl-shared\bin`（**torchaudio 解码音频的硬依赖**，见风险 6） |

**归一化验证（每次重建后都要跑）**
```powershell
F:\embedded\prepare\.venv\Scripts\python.exe -c "import sys; print(sys.executable); print(sys.base_prefix)"
```
✅ 合格：`base_prefix` = `C:\Users\ADMIN\.workbuddy\binaries\python\versions\3.13.12`
❌ 不合格：`base_prefix` = `F:\anaconda3`（说明又用 conda 的 python 建了 venv）

**一键重建**：直接跑 `scripts/setup_env.ps1`（已封装下文全部步骤与顺序）。

**手工重建的顺序（不能颠倒）**
```powershell
# 1) 用独立解释器建 venv（--clear 会清空目录，venv 是纯派生产物，无数据损失）
C:\Users\ADMIN\.workbuddy\binaries\python\versions\3.13.12\python.exe -m venv --clear F:\embedded\prepare\.venv

# 2) 先升级 pip —— 走官方源（不要走镜像，原因见风险 2）
F:\embedded\prepare\.venv\Scripts\python.exe -m pip install -U pip --index-url https://pypi.org/simple

# 3) 再装 torch —— 必须走 pytorch 官方源，且必须是 cu128（RTX 5060 是 sm_120）
F:\embedded\prepare\.venv\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4) 最后装其余 —— 走默认源（本机 pip.ini 已配清华源）
F:\embedded\prepare\.venv\Scripts\python.exe -m pip install numpy pandas soundfile librosa tensorboard
```
> ⚠️ **三步必须分开执行**。原因：pytorch 源里没有 pandas，清华源里没有 cu128 的 torch，一次写在一起会整体解析失败。

---

## 二、GPU 自检：建议升级成这个版本

当前 `scripts/test/test.py` 能跑通，但它只证明了「CUDA 可用」。做 ASR 训练前，**要证明这张卡真的以 sm_120 在跑**，否则可能在兼容模式下性能大打折扣却不自知。

```python
import torch, torchaudio

print("torch      :", torch.__version__)
print("torchaudio :", torchaudio.__version__)
print("cuda avail :", torch.cuda.is_available())
print("device name:", torch.cuda.get_device_name(0))          # 必须是 RTX 5060
cap = torch.cuda.get_device_capability(0)
print("capability :", cap)                                     # 必须是 (12, 0)
print("cuda ver   :", torch.version.cuda)                      # 12.8

# 真实 kernel：fp16/bf16 矩阵乘（ASR 训练会大量用到）
for dt in (torch.float32, torch.float16, torch.bfloat16):
    a = torch.randn(1024, 1024, device="cuda", dtype=dt)
    b = torch.randn(1024, 1024, device="cuda", dtype=dt)
    print(f"matmul {str(dt):16s} ok  sum={float((a @ b).sum()):.1f}")

# 显存与设备端同步（确认没有异步错误被吞掉）
torch.cuda.synchronize()
print("mem allocated: %.1f MB" % (torch.cuda.memory_allocated() / 1024**2))
```

**判读要点**
- `capability == (12, 0)` → 轮子是真的为 Blackwell 编译的；若显示 `(8, 6)` / `(7, 5)` 之类，说明落到了兼容路径，换轮子
- bf16 那一行如果报错，说明该 dtype 支持有问题，后面训练要注意混合精度策略
- `synchronize()` 没抛异常才算真的跑完（CUDA 错误是异步的，不 sync 可能吞掉）

---

## 三、已知风险与对策

### ✅ 风险 1：conda 与 venv 的耦合 —— **已解决**

**问题**：原始 venv 是从 `F:\anaconda3\python.exe` 建的，`pyvenv.cfg` 里 `home = F:\anaconda3`。
`include-system-site-packages = false` **只隔离第三方包，不隔离解释器本体和标准库**，`.venv\Scripts\python.exe`（253 KB）只是个壳，真正加载的是 `F:\anaconda3\python313.dll`。

**危害**：`F:\anaconda3\Library\bin` 里实测存在这些冲突源——

| DLL | 会让谁炸 |
|---|---|
| `libiomp5md.dll` | Intel OpenMP → PyTorch 经典 `OMP: Error #15`，或静默性能劣化 |
| `libssl-3-x64.dll` / `libcrypto-3-x64.dll` | `cryptography` / `requests` → `DLL load failed while importing _ssl` |
| `ffi.dll` | **`cffi`**（`soundfile` 依赖它）→ `_cffi_backend` 导入失败 |
| `mkl_rt.2.dll` | 与 numpy/scipy 自带的 MKL/OpenBLAS 抢符号 |
| `liblzma.dll` | 压缩相关包（onnx 工具链会碰到） |

> ⚠️ 顺带纠正一个常见误解：**光用绝对路径调 venv 的 python 解决不了问题**。
> Windows 的 DLL 搜索走 `PATH`，只要终端激活过 conda base，`Library\bin` 就在搜索范围内。
> **关键是让 `PATH` 里没有 anaconda，而不是「用哪个 python」。**

**解决方案（已执行）**：用独立解释器 `C:\Users\ADMIN\.workbuddy\binaries\python\versions\3.13.12\python.exe` 重建 venv。
现在 `base_prefix` 已指向该目录，环境与 conda 完全脱钩。

**日常辅助：消除 `(base)` 提示符（分两个层面）**

**先看机制** —— `C:\Users\ADMIN\Documents\WindowsPowerShell\profile.ps1` 里有一段 `conda initialize`，每个新 PowerShell 都会执行 `conda.exe shell.powershell hook`。该 hook 只做两件事：
1. 设置 `CONDA_EXE` / `_CONDA_EXE` / `_CE_M` / `_CE_CONDA` / `CONDA_PYTHON_EXE` / `_CONDA_ROOT`
2. `Import-Module F:\anaconda3\shell\condabin\Conda.psm1` —— **`conda` 函数和 `(base)` 前缀都来自这个模块**

`(base)` 由模块注入的 `prompt()` 渲染，而它读的是 **`$Env:CONDA_PROMPT_MODIFIER`**。
> ⚠️ **只清 `CONDA_PREFIX` 是没用的** —— 必须清 `CONDA_PROMPT_MODIFIER`，否则提示符照旧显示 `(base)`。（这是第一版脚本的漏项）

**层面 1｜会话级：脚本已修**

`scripts/activate_project.ps1` 现在做六件事：
1. 剔除 `PATH` 中所有 conda / anaconda 条目
2. 清**全部** `CONDA*` / `_CE_*` 变量（含关键的 `CONDA_PROMPT_MODIFIER`）+ `PYTHONPATH` / `PYTHONHOME`
3. 激活 venv
4. **重定义 `prompt()`** —— 因为新终端会重新加载 profile、重新注入 conda 的 prompt，只清变量不够，必须覆盖函数
5. 设置 F 盘缓存/临时目录
6. 自检并报告 conda 残留

```powershell
powershell -ExecutionPolicy Bypass -File F:\embedded\prepare\scripts\activate_project.ps1
```
成功标志：提示符 `(.venv) PS F:\embedded\prepare>`，并打印：
```
[env] OK    PATH is free of conda/anaconda
[env] OK    prompt modifier cleared (no '(base)' prefix)
```
> 📌 脚本里的提示行用 `Write-Host`（带颜色），`>` / `2>&1` 抓不到，要抓日志用 `*>&1 | Out-File`。

**层面 2｜全局级：新终端默认就干净 ✅ 已执行**

```powershell
conda config --set auto_activate_base false
```
已执行并验证。`.condarc` 新增一行 `auto_activate: false`
（新版 conda 会提示 `auto_activate_base` 是 `auto_activate` 的别名，等价）。

**效果**：新开的终端不再自动激活 base —— 没有 `(base)` 前缀，`PATH` 里也没有 anaconda 的 `Library\bin`，**风险 1 的触发条件从根上消失**。

**验证依据**：conda 源码 `conda/activate.py` 中 `hook()` 的逻辑是
`if auto_activate is None and context.auto_activate or auto_activate:` ——
设置后导出的 hook 里已不含任何 activate 逻辑。

**撤销**（如果哪天需要）：
```powershell
conda config --set auto_activate_base true
```
或临时手动激活：`conda activate base`（conda 功能完好，只是不再自动进）。

> ⚠️ 你的 `.condarc` 里还有 `envs_dirs` / `pkgs_dirs`（都指向 F 盘）和清华 conda 源，这次**只加了一行**，其余未动。

---

### ⚠️ 风险 2：**pip 版本过低会与国内镜像源不兼容**（本次实际踩到）

**症状**（照抄真实报错）：
```
Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
ERROR: Could not find a version that satisfies the requirement numpy (from versions: none)
ERROR: No matching distribution found for numpy
```
注意 **`from versions: none`** —— 这不是「包不存在」，而是**索引返回了空列表**。而且连 `six` 这种包也一并失败。

**诊断路径（照这个顺序排查，别瞎猜）**
| 步骤 | 命令 | 结果 | 结论 |
|---|---|---|---|
| 1. 验证镜像是通的 | `curl -sI https://pypi.tuna.tsinghua.edu.cn/simple/numpy/` | HTTP 200，1.3 MB HTML | 镜像**正常** |
| 2. 排除缓存损坏 | `pip download --no-deps --no-cache-dir six` | 仍失败 | **不是缓存问题** |
| 3. 换官方源验证包存在 | `pip index versions numpy --index-url https://pypi.org/simple` | 正常列出 2.5.3 等全部版本 | 包**存在**，问题在 pip×镜像 |
| 4. 升级 pip（官方源） | `pip install -U pip --index-url https://pypi.org/simple` | 26.1.2 → **26.2.1** | — |
| 5. 回到清华源复测 | `pip index versions numpy` | ✅ 正常 | **根因确认为 pip 版本** |

**根因**：venv 自带的 pip 是 **26.1.2**，与清华源的 simple API 交互异常（拿不到包列表）。升级到 **26.2.1** 后立刻恢复。
对照证据：用户之前那个 conda venv 里 pip 恰好已被升到 26.2.1，所以同样的源、同样的包，那边成功、这边失败。

**对策（已固化为规矩）**：
> **建完新 venv 的第一件事，永远是先把 pip 升到最新**，而且**用官方源升**（因为此时镜像可能正是故障源，走镜像会死循环）。
> ```powershell
> .venv\Scripts\python.exe -m pip install -U pip --index-url https://pypi.org/simple
> ```
> 若日后又遇到 `from versions: none`，先跑 `pip -V` 看版本，再按上表五步排查。

---

### ⚠️ 风险 3：**不要 `pip install -r requirements.txt`（WeNet 的）**

WeNet 仓库根目录的 `requirements.txt` 是**CI/开发依赖合集**，不是运行依赖，里面有一堆 pin 死的老包和 lint 工具：

```
Pillow / pyyaml>=5.1 / sentencepiece / tensorboard / tensorboardX / textgrid / pytest /
flake8==3.8.2 / flake8-bugbear / flake8-comprehensions / flake8-executable / flake8-pyi /
mccabe / pycodestyle==2.6.0 / pyflakes==2.2.0 / clang-format / cpplint /
torch>=2.1.2 / torchaudio>=2.1.2 / tqdm / deepspeed>=0.14.0 / librosa /
openai-whisper==20231117 / pre-commit==3.5.0 / langid
```

直接全量装有三个雷：
1. `deepspeed>=0.14.0` —— 对 torch 版本极敏感，torch 2.11 不保证兼容，且只在多机训练时才需要
2. `openai-whisper==20231117` —— 2023 年的老版本，**与 numpy 2.x 不兼容**（我们装的是 numpy 2.5.3）
3. `flake8==3.8.2` / `pycodestyle==2.6.0` —— 4 年前的老 pin，Python 3.13 下未必装得上

注意 torch 的约束是 `torch>=2.1.2`，**没有上限** —— 所以 2.11 不会被拒，但不代表兼容。真出问题别怀疑版本约束。

**正确做法：只装运行必需的**
```powershell
python -m pip install pyyaml sentencepiece tensorboard tensorboardX textgrid tqdm Pillow
# torch / torchaudio / librosa 已装，无需重复
# deepspeed 只在需要多卡时再装；openai-whisper 只在做 whisper 蒸馏 recipe 时另起 venv
```

---

### ⚠️ 风险 4：Python 3.13 与 WeNet 的兼容性

WeNet 的主要开发和 CI 都跑在 Python 3.8–3.11 上。Python 3.12 起 `distutils` 被移出标准库，3.13 又清理了一批老 API，WeNet 的 `tools/` 和部分工具脚本可能因此报 `ModuleNotFoundError: No module named 'distutils'`。

**预案（不要一上来就降级，先装再试）**：
1. `pip install setuptools`（setuptools 会以 shim 形式提供 `distutils`，多数情况能救回来）
2. 仍失败 → 用 Python 3.11 单独建一个训练 venv（`F:\embedded\prepare\.venv311`），把训练和板端工具链隔离
3. 结果记到下方「踩坑日志」

---

### ⚠️ 风险 5：Windows 上跑不了 WeNet 的 `run.sh`

WeNet 的 recipe（如 `examples/aishell/s0/run.sh`）是 bash 脚本，依赖一堆 Unix 工具；数据准备环节通常还要 Kaldi 风格的小工具。本机 **WSL 被安全策略禁用**，Docker 守护进程未启动，所以「整条 `run.sh` 一把梭」这条路走不通。

**这不是死路——正确姿势是绕过 shell 脚本，手动分步执行 Python 命令**：
```powershell
# 1) 数据准备：AISHELL-1 是 wav + 文本，不需要 Kaldi 工具，
#    只要自己生成 data/train/{wav.scp,text,utt2spk,spk2utt} 并调用：
python tools/make_shard_list.py --num_workers 8 --num_utts_per_shard 1000 \
    --prefix shards --shuffle True --input data/train/wav.scp data/train/text

# 2) 训练 / 解码 / 打分：直接调 python 入口，参数从 run.sh 里抄
python wenet/bin/train.py --config conf/train_conformer.yaml --gpu 0 ...
python wenet/bin/recognize.py --config conf/decode.yaml ...
python wenet/bin/compute_wer.py ...
```
**关键认知**：`run.sh` 只是这些命令的编排器 + 一个 stage 开关（`--stage` / `--stop_stage`）。把参数抄出来手动跑，等价且更可控，唯一代价是不能 `--stage 5` 一键续跑。

📌 这件事要在 **W3 之前**解决掉（阶段 1 复现要用）。建议 W2 末尾先拿 AISHELL 的 200 条子集做一次端到端手动流水线。

> ✅ **2026-09-20 更新**：已完成。`examples/aishell/s0/` 的 stage 0~3 全部移植进
> `scripts/prepare_aishell.py`，一条命令跑完，见第五节。

---

### 🔴 风险 6：**torchaudio 2.11 解码音频需要 FFmpeg 共享库**（训练的头号拦路虎）

**现象**：

```python
>>> torchaudio.load("x.wav")
RuntimeError: Failed to create AudioDecoder for x.wav: Could not load libtorchcodec.
>>> torchaudio.info("x.wav")
AttributeError: module 'torchaudio' has no attribute 'info'      # ← 2.11 已删除该 API
```

**根因**：torchaudio ≥ 2.9 把音频解码整体改为依赖 **torchcodec**，而 torchcodec
**自己不捆绑 FFmpeg 的共享库**（`avcodec/avformat/avutil/swresample/swscale`），
只捆绑了 libwebp/libavif/zstd 这些。所以即使 `pip install torchcodec` 成功，
第一次解码仍会失败。本机 PATH 里也没有任何 FFmpeg。

**为什么这是致命的**：WeNet 训练的数据管线 `wenet/dataset/processor.py::decode_wav`
第 156 行就是 `torchaudio.load(wav_file)` —— 它挂了，**训练直接起不来**，
而且报错信息（libtorchcodec）跟"数据准备"看起来毫无关系，很难往这个方向查。

**解决**：装 FFmpeg 的 **shared 构建**（不是 static），把它的 `bin` 加到 PATH。

```bash
# 下载（本机直连 github 不通，走 ghproxy 镜像；合计 86 MB）
curl -L -C - -o ffmpeg-shared.zip \
  "https://ghproxy.net/https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl-shared.zip"
# 解压（用 Python 的 zipfile，不依赖 unzip/7z）
python -c "import zipfile;zipfile.ZipFile('ffmpeg-shared.zip').extractall('ffmpeg-shared')"
# 验证关键 DLL 存在
ls ffmpeg-shared/*/bin/avcodec-*.dll
```

安装位置（本机约定）：`F:\data\toolchains\ffmpeg-shared\ffmpeg-master-latest-win64-gpl-shared\bin`

**用起来**：本项目的 `scripts/prepare_aishell.py` **会在启动时自动检测并把该目录塞进
`os.environ["PATH"]`**（见 `ensure_ffmpeg_on_path()`），所以正常调用不需要手动设环境变量。
若你自己写脚本要用到 `torchaudio.load`，记得先：
```powershell
$env:PATH = "F:\data\toolchains\ffmpeg-shared\ffmpeg-master-latest-win64-gpl-shared\bin;$env:PATH"
```

**验证方法**：
```python
import torchaudio
w, sr = torchaudio.load(r"data\aishell\raw\wav\train\S0002\BAC009S0002W0122.wav")
print(w.shape, sr)   # 期望 (1, 95984) 16000
```

**连带影响**：WeNet 的 `tools/compute_cmvn_stats.py` 同时用了 `torchaudio.info`
（2.11 已删）与 `torchaudio.load`，**即使有 FFmpeg 也跑不了**。
→ 本项目改用自己写的 CMVN（`prepare_aishell.py` 的 `cmvn` 阶段）：
用 `soundfile` 读音频 + **torchaudio 的 `kaldi.fbank`**（与训练时同一个函数，统计可比），
并且用 12 进程并行。实测 **34,679 条 / 65 秒**（单项 ~8 ms），比官方脚本快得多。

---

### ⚠️ 风险 7：PowerShell 5.1 的两个坑（写 .ps1 脚本必踩）

**坑 A：`$ErrorActionPreference = "Stop"` 会把原生命令的 stderr 当成终止性错误**

本项目所有 Python 脚本按设计把信息写到 **stderr**（这样 stdout 可以干净地做数据管道）。
但在 PowerShell 里，只要 `$ErrorActionPreference = "Stop"`，
**原生命令往 stderr 写的每一行都会变成 `NativeCommandError` 并中断脚本**。
实测：`run_prepare.ps1` 第一版在"开始数据准备"那一行之后直接退出，退出码 1，
而 Python 其实一行都没跑。

→ 修法：调用原生命令前把 `$ErrorActionPreference` 降回 `"Continue"`。

**坑 B：无 BOM 的 UTF-8 `.ps1` 会被当成 GBK 读，中文字节错位后把引号吃掉**

```powershell
[System.Management.Automation.PSParser]::Tokenize((Get-Content -Raw $p), [ref]$errs)
# 报 "字符串缺少终止符" / "缺少右 }" —— 但文件本身完全合法
```
PowerShell 5.1 对无 BOM 文件按系统 ANSI（本机 GBK）解码，中文变成乱码后
引号配对失败。**含中文的 `.ps1` 必须存成 UTF-8 with BOM。**

→ 修法：写文件时加 `\xEF\xBB\xBF` 前缀。

> 顺带：用 `Set-Content -Encoding UTF8` 写日志得到的是 **UTF-16LE**（带 BOM），
> 用 Git Bash 按 UTF-8 读会乱码。跨工具读日志时要注意编码。

---

## 四、数据准备流水线（D2 落地，一条命令跑完）

WeNet 官方 `run.sh` 在 Windows 不可用（见风险 5），stage 0~3 已移植为
**`scripts/prepare_aishell.py`**。日常只需要这一条命令：

```powershell
cd F:\embedded\prepare
.\.venv\Scripts\python.exe scripts\prepare_aishell.py --stage all
```

或用封装脚本（自动设好 PATH / 编码）：`scripts\run_prepare.ps1`

**产物布局**（刻意对齐 WeNet 自己的 recipe 布局，yaml 里的相对路径直接可用）：

```
work/aishell/                      ← 训练时的 CWD 必须是这里
├── conf/smoke.yaml                ← 由 --stage conf 从官方流式配置派生
└── data/
    ├── train/  {wav.scp, text, data.list, global_cmvn}
    ├── dev/    {wav.scp, text, data.list}
    ├── test/   {wav.scp, text, data.list}
    ├── train_smoke/ {…}
    ├── dev_smoke/   {…}
    └── dict/lang_char.txt
```

**实测数字（2026-09-20）**

| 产物 | 数量 |
|---|---|
| train wav.scp / text / data.list | **34,679** 行（三者一致） |
| dev | 14,326 |
| test | 7,176 |
| dict/lang_char.txt | 3,590 行（3 特殊符号 + 3,587 字符） |
| global_cmvn | 80 维，`frame_num = 15,367,984`（≈42.7 h），计算耗时 **65 s**（12 进程） |
| train_smoke / dev_smoke | 200 / 50 条（覆盖 100 / 40 个说话人） |

**三个与官方 recipe 的有意不同**（都源自数据审计的实际发现）：

1. **文本统一以官方 transcript 为唯一来源**；dev/test 的 carlot 文本只用于交叉验证
   （脚本内断言，实测 21,502 条 0 实质不一致）。少一个来源就少一类 bug。
2. **归一化删掉所有空白**，不是只 `strip()` —— carlot 文本带统一 4 个前导空格。
3. **显式剔除空音频**并打日志（train 有 1 个 44 字节的空 wav）。
   train 的 34,679 是「wav ∩ 有标注」的交集，**不是** 34,716 个 wav 文件数。

---

### 🔴 风险 8：torch 2.11 的 Windows 轮子有 TCPStore/libuv bug（**训练起不来**）

**现象**：
```
torch.distributed.DistStoreError: use_libuv was requested but PyTorch was built
without libuv support, run with USE_LIBUV=0 to disable it.
```

**⚠️ 报错信息是误导的 —— 设 `USE_LIBUV=0` 完全无效。** 实测三种写法：

| 写法 | 结果 |
|---|---|
| `TCPStore(...)` 默认参数（`USE_LIBUV` 未设） | ❌ |
| `TCPStore(...)` 默认参数（`USE_LIBUV=0`） | ❌ **设了也没用** |
| `TCPStore(..., use_libuv=False)` | ✅ |

**根因**：torch 里创建 TCPStore 的地方**根本没传 `use_libuv`**，于是走 C++ 默认值 True，
而这个 Windows 轮子没编 libuv 支持。全仓库共 12 处 `TCPStore(` 调用，**9 处**漏传，
包括 `dynamic_rendezvous.py` 里用限定名 `dist.TCPStore` 的那处。

**修法**：`python scripts/patch_torch_libuv.py`（括号配对扫描统一补 `use_libuv=False`，
自动跳过文档字符串里的示例代码；`--check` 只看、`--revert` 还原）。

> 💡 `--standalone` **不是**解法：读 `run.py:964` 可见它内部就是把 backend 设成 `c10d`。
> 用 `--rdzv_backend=static` 能绕过 elastic rendezvous 的 store，但
> `init_process_group` 自己还会建一个 TCPStore，照样崩 —— 所以必须打补丁。
>
> torch 升级后补丁会失效，重跑一次即可（`--check` 会告诉你状态）。

---

### 🔴 风险 9：WeNet main 与 torch 2.11 / 精简依赖不兼容（7 处）

`python scripts/patch_wenet.py`（精确文本替换；匹配不上就报错退出，不静默改错）

| # | 文件 | 问题 | 修法 |
|---|---|---|---|
| 1 | `models/squeezeformer/conv2d.py` | 从 `torch.nn.modules.conv` import `Union`，2.11 不再带出 | 改为从 `typing` 导入 |
| 2 | `utils/train_utils.py` | 顶层 `import deepspeed` + 3 子模块 | try/except 可选 + `DEEPSPEED_AVAILABLE` |
| 3 | 同上 | `deepspeed.add_config_arguments` 无条件调用 | 加可用性判断 |
| 4 | 同上 | `deepspeed.init_distributed` 无兜底 | 加 assert + 清晰报错 |
| 5 | `utils/common.py` | 顶层 `from whisper.tokenizer import LANGUAGES`（**被整个模型栈依赖**） | try/except，`WHISPER_LANGS = ()` |
| 6 | `utils/init_tokenizer.py` | 顶层 import `WhisperTokenizer`（会 `import whisper`） | 移除顶层、改惰性 |
| 7 | 同上 | 用到时才 import | 移进 `if tokenizer_type == "whisper":` 分支 |

**为什么 1 和 5 是致命的**：`init_model.py` 会把**所有** backbone 都 import 一遍
（哪怕配置里只写 `encoder: conformer`），而 `encoder.py` 又依赖 `common.py` ——
所以任意一处坏掉，整个训练都起不来。

---

### ⚠️ 风险 10：Windows 训练的两个操作坑

**A. Git Bash 会改写 PATH 里的 `F:/...`**
```bash
export PATH="F:/data/toolchains/ffmpeg-shared/.../bin:$PATH"   # ❌ 被改写成
#   ...PortableGit\versions\1.2.0\data\toolchains\...    ← 主进程和子进程都找不到 FFmpeg
```
→ **在 Python 里设 `os.environ["PATH"]`**（不会被 shim 改写，子进程照常继承）。
`scripts/run_train.py` 与 `scripts/prepare_aishell.py` 都已这么做。

**B. `--num_workers 0` 会撞 `prefetch_factor`**
WeNet 无条件传 `prefetch_factor=args.prefetch`，而 PyTorch 要求 `num_workers>0` 时才允许。
→ 必须 `--num_workers >= 1`（配方的 4 就可以）。

---

### 🔴 风险 11：**Windows 上 gloo 是 CPU-only → DDP 反向段错误**（GPU 训练起不来）

**现象**：`--device cuda` 训练时第一个 batch 崩溃，退出码 **3221225477 (0xC0000005)**，
Python 层**拿不到任何异常**；`CUDA_LAUNCH_BLOCKING=1` 也没用。

**根因**：PyTorch 的 Windows 轮子**没有 NCCL**（`is_nccl_available() == False`），
只能用 gloo；而 **gloo 在 Windows 上是 CPU-only 构建**。
但 `torch.nn.parallel.DistributedDataParallel` 反向时要把 **CUDA** 梯度 `all_reduce`
→ gloo 拿不到 CUDA tensor → **段错误**。
WeNet 的 `wrap_cuda_model` 里 DDP 是**无条件包的**（`world_size == 1` 也照包）。

**最小复现**（这就是定位它的方法）：
```python
import torch, torch.nn as nn, torch.distributed as dist
dist.init_process_group("gloo"); torch.cuda.set_device(0)
m = nn.Linear(64, 64).cuda()
m(torch.randn(4, 64, device="cuda")).sum().backward()   # ✅ 裸模型 OK
ddp = nn.parallel.DistributedDataParallel(nn.Linear(64, 64).cuda())
y = ddp(torch.randn(4, 64, device="cuda"))              # 前向 OK
y.sum().backward()                                      # ❌ 段错误，退出码 139
```

**修法**：`scripts/patch_wenet.py` 第 8 处 —— **`world_size == 1` 时不包 DDP**。
安全性已确认：全仓库无一处使用 `.module`，`save_model` 走 `state_dict()`，两种包装都兼容。

**修复后实测**：`run_train.py --preset smoke` 自动选 cuda → `EXIT=0`，
cv_loss 114.38 → 89.88，**14.4 steps/sec**（CPU 是 0.55，快 26 倍）。

> ⚠️ **长远影响**：Windows 上只有 gloo（无 NCCL），所以 **DDP 多卡在 Windows 上走不通**。
> 多卡训练要上 Linux/WSL。单卡 RTX 5060 训 AISHELL-1 子集够用。

---

### 🔴 风险 12：**编码导致的"静默丢样本"** —— 报错会伪装成 `ZeroDivisionError`

**最坑的一类问题：不崩溃、不报错，只是样本被悄悄丢掉。**

`wenet/dataset/datapipes.py:353 TextLineDataPipe`：
```python
_dp = datapipes.iter.FileOpener(_dp, mode=mode)   # encoding 默认 None = 系统编码（GBK）
for line in stream:                                # 逐行读 → 撞中文就炸
```
`data.list` 的 `txt` 字段含中文（我方生成时用 `ensure_ascii=False`）→ **每条样本都解码失败**
→ 外面包着 `map_ignore_error` → **样本被静默全丢** → dataloader 产出 0 batch
→ `executor.cv` 里 `sum(total_acc)/len(total_acc)` → **`ZeroDivisionError: division by zero`**

**同一个症状还有第二个成因**：`decode_wav` 也用 `torchaudio.load`，也在 `map_ignore_error` 里。
**FFmpeg 找不到时，样本同样被静默全丢 → 同样报 `ZeroDivisionError`。**

> ### 🔍 排查口诀
> **看到 `ZeroDivisionError`（executor.cv）→ 先查两件事：① 文件编码 ② FFmpeg**
> 别去读 cv 的代码，它没问题。

**已在 `scripts/patch_wenet.py` 里修掉**：
- 第 11 处：`FileOpener(..., encoding='utf-8')`
- 第 10 处：`map_ignore_error` 补打 **完整 traceback**
  （原版只打一行 `str(ex)`，丢样本时根本看不出哪一步失败 —— **丢数据而不自知是最危险的**）
- 第 12~24 处：13 个文件的 yaml 读写统一加 `encoding='utf-8'`（YAML 规范就是 UTF-8）

---

### 🟡 风险 13：FFmpeg 的 DLL 用 `add_dll_directory` 而不是 PATH

`scripts/install_env_fix.py` 在 venv 的 site-packages 放一个 `.pth`：

```
<项目>/scripts/pyfix
import wenet_dll_fix
```

`wenet_dll_fix` 用 **`os.add_dll_directory()`**（CPython 3.8+ 官方 API）把 FFmpeg 目录
注册进进程的 DLL 搜索路径。Python 启动时 `site.py` 会执行 `.pth` 里的 import，
所以**每个进程（含 DataLoader 的 spawn worker）自动生效**。

**为什么不用「把目录加到 PATH」**：
- ⚠️ **Git Bash 会改写 PATH 里的 `F:/...`**（变成不存在的 `...PortableGit\versions\...\data\...`），
  主进程和 worker 都找不到 → 又变成上面那个静默丢样本
- PATH 是"每个 shell 都要记得设"的东西，忘了就复现这个假错误

实测：全新进程、**零环境变量** → `torchaudio.load` 返回 `(1, 95984) @16000Hz`。

命令：`python scripts/install_env_fix.py`（`--check` 只看，`--uninstall` 移除）

---

### 🟡 风险 14：`final.pt` 是断链（WeNet 自身的两个 bug）

`wenet/bin/train.py` 收尾处：
```python
os.remove(final_model_path) if os.path.exists(final_model_path) else None
os.symlink('{}.pt'.format(final_epoch), final_model_path)
```

| # | Bug | 后果 |
|---|---|---|
| 1 | 清理用 `os.path.exists` —— **对断链返回 False**（它会 follow 链接）→ 清理被跳过 | 同一 `model_dir` 重跑必然 `FileExistsError` |
| 2 | 链接目标写成 `4.pt`，而 `save_model` 存的是 `epoch_4.pt` | **`final.pt` 从来就是断链**；`run.sh` stage 5 用 `$dir/final.pt` 解码会失败 |

> ⚠️ **这个 bug 特别容易被漏掉**：`ls` 能看到 `final.pt`，命令也可能"看起来跑完了"，
> 但那是个指向不存在文件的断链。**检查要用 `os.path.islink()` + `os.path.exists()` 两个都看**：
> ```
> lexists=True  exists=False  islink=True  指向: 4.pt   ← 断链！
> ```

**已在 `patch_wenet.py` 第 25 处修掉**：`lexists` + 正确目标名 + 删除失败时 `os.rename` 兜底
+ 无符号链接权限时退化为 `shutil.copy2`。

---

## 五、踩坑日志

| 日期 | 问题 | 根因 | 解决 |
|---|---|---|---|
| 2026-09-18 | `pip install nuumpy` 报找不到包 | 打字错误 | 重敲为 `numpy` |
| 2026-09-18 | `python test.py` 报文件不存在 | 脚本实际在 `scripts/test/test.py` | 用相对路径调用 |
| 2026-09-18 | torch 下载中断（475MB/2.75GB） | 网络抖动 | pip 自动 resume，42.4 MB/s 完成 |
| 2026-09-18 | venv 基础解释器是 anaconda | 用 `python`（解析到 conda base）建 venv | 改用独立 Python 重建，见风险 1 |
| 2026-09-18 | **清华源报 `from versions: none`，连 six 都装不上** | **venv 自带 pip 26.1.2 与镜像 simple API 不兼容** | **用官方源升 pip 到 26.2.1，立刻恢复**。见风险 2 |
| 2026-09-20 | `torchaudio.load` 报 `Could not load libtorchcodec`；`torchaudio.info` 报 AttributeError | torchaudio ≥2.9 解码改依赖 torchcodec，而 torchcodec 不捆绑 FFmpeg 共享库；`info` 已被移除 | 装 FFmpeg **shared** 构建并挂 PATH；CMVN 改用自写版本（soundfile + kaldi.fbank）。见风险 6 |
| 2026-09-20 | `import yaml` 报 ModuleNotFoundError | WeNet 用 yaml 读配置，但 yaml 不在依赖里 | `pip install pyyaml` |
| 2026-09-20 | `run_prepare.ps1` 在打印第一行输出后即以退出码 1 中止 | `$ErrorActionPreference="Stop"` 把原生 Python 的 stderr 当成终止性错误 | 调原生命令前降级为 `"Continue"`。见风险 7 坑 A |
| 2026-09-20 | `.ps1` 被 PSParser 报"字符串缺少终止符"，但文件本身合法 | PowerShell 5.1 把无 BOM 的 UTF-8 当 GBK 读，中文错位吃掉引号 | 存成 **UTF-8 with BOM**。见风险 7 坑 B |
| 2026-09-20 | `nohup cmd &` 起的后台任务在工具调用返回后被杀 | 非交互式运行的进程生命周期跟主流程绑定 | 改用工具提供的后台执行机制 |
| 2026-09-20 | github.com 直连下载 FFmpeg 得到 `HTTP=000` 且 0 字节 | 直连被重置 | 走 `ghproxy.net` 镜像；大文件用 `curl -C -` 续传 |
| 2026-09-20 | `use_libuv was requested...` 按提示设 `USE_LIBUV=0` 无效 | torch 创建 TCPStore 的 9 处都没传 `use_libuv`，C++ 默认 True 而轮子无 libuv | **报错信息误导**；打补丁显式传 `use_libuv=False`。见风险 8 |
| 2026-09-20 | `ImportError: cannot import name 'Union' from 'torch.nn.modules.conv'` | torch 2.11 的 conv.py 不再带出 typing 名字 | 改从 `typing` 导入。见风险 9 |
| 2026-09-20 | `ModuleNotFoundError: langid` / `tqdm` | WeNet 运行时依赖，不在已装列表 | 补装（安全子集，跳过 deepspeed/whisper/flake8） |
| 2026-09-20 | `ModuleNotFoundError: deepspeed` / `whisper`（即使不用它们） | 被无条件顶层 import | 改成 try/except + 惰性导入。见风险 9 |
| 2026-09-20 | `FileNotFoundError: exp/smoke\train.yaml` | 官方 run.sh 有 `mkdir -p $dir`，手动跑漏了 | 启动器里补建目录 |
| 2026-09-20 | `torchaudio.load` 在训练里失败，但单测成功 | Git Bash 把 `export PATH="F:/..."` 改写了 | 改为在 Python 里设 `os.environ["PATH"]`。见风险 10A |
| 2026-09-20 | `ValueError: prefetch_factor option could only be specified in multiprocessing` | `--num_workers 0` 与 WeNet 硬传的 prefetch_factor 冲突 | `--num_workers >= 1`。见风险 10B |
| 2026-09-20 | `--device cuda` 训练崩，退出码 3221225477（access violation），Python 层无异常 | **Windows 的 gloo 是 CPU-only 构建，DDP 反向 allreduce CUDA 梯度 → 段错误**；WeNet 无条件包 DDP | `world_size==1` 时不包 DDP。见风险 11 |
| 2026-09-20 | 定位 CUDA 段错误时一度怀疑是 DataLoader 张量路径 | 消融显示：裸模型 CUDA 前向+反向正常，包 DDP 后 backward 立刻崩 | **教训：先做「裸 vs 包装」的最小对照，别一上来就怀疑数据管线** |
| 2026-09-20 | `UnicodeDecodeError: 'gbk' codec ...` 读 `conf/smoke.yaml` | `train.py:80` 打开 yaml 未指定编码；配置文件含中文注释 | 13 处 yaml 读写统一加 `encoding='utf-8'`。见风险 12 |
| 2026-09-20 | 修完 yaml 后变成 `ZeroDivisionError: division by zero` | **`data.list` 被按 GBK 逐行读**（`FileOpener` 未传 encoding）→ 含中文的行全部解码失败 → `map_ignore_error` 静默丢光样本 | `FileOpener(..., encoding='utf-8')`。见风险 12 |
| 2026-09-20 | 排查中一度被 `PATH有FFmpeg=True` 误导 | 我的检查只做了**字符串匹配**，而 Git Bash 改写后的路径仍含 "ffmpeg-shared" 字样（假阳性） | 检查路径要 **`os.path.isdir()` 实测**，不能只匹配字符串 |
| 2026-09-20 | 逐项加环境变量的二分实验结论全错 | `export PATH="\$FF;$PATH"` 里的 `\$FF` 被转义成**字面量** `$FF`，FFmpeg 从未进 PATH | 写错一次变量，整轮实验白跑 —— **跑实验前先验证前提成立** |
| 2026-09-20 | 定位 `ZeroDivisionError` 的关键一步 | `map_ignore_error` 只打一行 `str(ex)`，看不出哪一步失败 | 给它补打完整 traceback（并长期保留）。见风险 12 |
| 2026-09-20 | `FileExistsError: '4.pt' -> 'final.pt'`（训练已跑完，收尾那步炸） | WeNet 两个 bug：清理用 `exists` 而非 `lexists`（对断链返回 False）；链接目标少 `epoch_` 前缀 | 见风险 14 |
| 2026-09-20 | 多次"✅ final.pt 产出"的检查全是假阳性 | 我用 `ls final.pt` 判断，而它是个**断链** —— 名字存在但目标不存在 | 检查文件要用 `islink()` + `exists()` 两个都看。见风险 14 |
| 2026-09-20 | `PermissionError: [WinError 5]` 删断链失败（但单独跑同一个删除却成功） | 断链这种特殊状态下 Windows 的删除行为异常 | 删除加 `os.rename` 兜底（重命名只需目录写权限） |
