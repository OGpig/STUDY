# 环境基线（ENV）

> 本文件是**唯一的真相来源**：任何人在任何一天都能照着这里复现出完全一样的环境。
> 环境变了（升级/换版本）就更新这里，别只在脑子里记。

---

## 一、当前基线（2026-09-18 重建后实测）

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

---

## 四、踩坑日志

| 日期 | 问题 | 根因 | 解决 |
|---|---|---|---|
| 2026-09-18 | `pip install nuumpy` 报找不到包 | 打字错误 | 重敲为 `numpy` |
| 2026-09-18 | `python test.py` 报文件不存在 | 脚本实际在 `scripts/test/test.py` | 用相对路径调用 |
| 2026-09-18 | torch 下载中断（475MB/2.75GB） | 网络抖动 | pip 自动 resume，42.4 MB/s 完成 |
| 2026-09-18 | venv 基础解释器是 anaconda | 用 `python`（解析到 conda base）建 venv | 改用独立 Python 重建，见风险 1 |
| 2026-09-18 | **清华源报 `from versions: none`，连 six 都装不上** | **venv 自带 pip 26.1.2 与镜像 simple API 不兼容** | **用官方源升 pip 到 26.2.1，立刻恢复**。见风险 2 |
