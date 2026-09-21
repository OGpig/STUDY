# Git 操作日志

## 基本信息

| 项目 | 内容 |
| --- | --- |
| 日期 | 2026-09-18 |
| 系统 | Windows |
| 终端 | PowerShell |
| Git 版本 | 2.55.0.windows.5 |
| GitHub 账号 | OGpig |
| 仓库 | STUDY (`https://github.com/OGpig/STUDY.git`) |
| 邮箱 | 3170756265@qq.com |
| 工作目录 | `C:\Users\ADMIN\Desktop` |

## 操作记录

1. `Test-Path "C:\Program Files\Git\mingw64\libexec\git-core\git-remote-https.exe"` → False（无 HTTPS 辅助程序）
2. `git --version` → git version 2.55.0.windows.5
3. `echo "# STUDY" >> README.md`
4. `git init` → 初始化仓库
5. `git add README.md`
6. `git commit -m "first commit"` → 提交成功 `c152504`
7. `git branch -M main` → 主分支改为 main
8. `git remote add origin https://github.com/OGpig/STUDY.git`
9. `git push -u origin main` → 失败（见问题 1）
10. `git remote set-url origin ssh://git@ssh.github.com:443/OGpig/STUDY.git` → 改用 SSH 443 端口
11. `ssh -T -p 443 git@ssh.github.com` → 认证失败（见问题 2）
12. `ssh-keygen -t ed25519 -C "3170756265@qq.com"` → 生成密钥
13. 将公钥 `~/.ssh/id_ed25519.pub` 添加到 GitHub
14. `ssh -T -p 443 git@ssh.github.com` → 认证成功：`Hi OGpig!`
15. `git push -u origin main` → 推送成功，`main` 已跟踪 `origin/main`

## 遇到的问题

### 问题 1：无法连接 github.com:443

- 现象：
  - `Recv failure: Connection was reset`
  - `Failed to connect to github.com:443 after 21086 ms: Could not connect to server`
- 原因：HTTPS 方式访问 github.com 被重置/超时（网络限制），本机也没有 `git-remote-https.exe`。
- 解决：放弃 HTTPS，改用 SSH 走 443 端口：
  `git remote set-url origin ssh://git@ssh.github.com:443/OGpig/STUDY.git`

### 问题 2：SSH 认证失败 Permission denied (publickey)

- 现象：
  - 首次连接提示 `Host key verification failed`（ssh.github.com:443 主机指纹未确认）
  - 确认主机后报 `git@ssh.github.com: Permission denied (publickey)`
  - 查看 `~/.ssh` 目录只有 `known_hosts`，没有密钥
- 原因：本地没有 SSH 密钥，GitHub 未绑定公钥。
- 解决：
  1. `ssh-keygen -t ed25519 -C "3170756265@qq.com"` 生成密钥（直接回车，无密码短语）
  2. `cat ~/.ssh/id_ed25519.pub` 复制公钥，添加到 GitHub 的 SSH Keys
  3. `ssh -T -p 443 git@ssh.github.com` 验证通过后即可推送

## 备注

- 重复出现的连接超时（失败 6 次）属同一问题，只记录一次。
- 主机指纹确认问题与密钥问题同属 SSH 认证环节，一并记录。

---

# 第二次操作：项目记录首次上传（D1 归档）

## 基本信息

| 项目 | 内容 |
| --- | --- |
| 日期 | 2026-09-18 |
| 系统 | Windows |
| 终端 | Git Bash |
| 工作目录 | `F:\embedded\prepare` |
| 仓库 | STUDY (`ssh://git@ssh.github.com:443/OGpig/STUDY.git`) |
| 提交 | `1c12020`（父提交 `c152504`） |
| 推送结果 | `c152504..1c12020  main -> main` ✅ 快进推送，无需 force |

## 背景问题：仓库位置与远程绑定错位

第一次操作时 `git init` 是在 **`C:\Users\ADMIN\Desktop`** 下做的 —— 那是桌面目录，不是项目目录。
结果出现两个各自独立的仓库：

| 位置 | HEAD | 是否绑定远程 | 说明 |
| --- | --- | --- | --- |
| `C:\Users\ADMIN\Desktop` | `c152504`（README 仅 20 字节） | ✅ 绑定 SSH 443 | 真正的 `origin` 在这里，但目录不对 |
| `F:\embedded\prepare` | `124f495`（只跟踪了一个 README） | ❌ 未绑定 | 项目实际所在，却推不出去 |

两个 `first commit` 不同源，无法直接合并。

**处理方式**：以远端为准重建项目仓库的历史 —— 把项目仓库的 `main` 指到远端的 `c152504`，
在其上做一次增量提交。这样推送是**快进（fast-forward）**，
不需要 `--force`，也不会覆盖远端已有历史。

```bash
# 1) 备份原引用，防止重置后找不回来
git branch backup-local-main

# 2) 把 main 重置到远端提交，实现与远端同源
git update-ref refs/heads/main c152504c6ce5948e7d39e18b61f21dfe2383a289

# 3) 切换 origin 到 SSH 443（沿用第一次操作已验证可用的通道）
git remote set-url origin ssh://git@ssh.github.com:443/OGpig/STUDY.git

# 4) 暂存 → 提交 → 推送
git add -A
git commit -F ./commit-msg.txt
git push origin main

# 5) 确认无误后删除备份分支
git branch -D backup-local-main
```

## 入库策略：只提交"人写的东西"

原则 —— **凡是能重新生成的、第三方的、体积大的，一律不入库。**

### ✅ 入库（36 个文件 / 4871 行）

| 目录 | 内容 | 为何值得留档 |
| --- | --- | --- |
| `README.md` `PROGRESS.md` | 项目主线、进度总表 | 求职时可直接展示的叙事线 |
| `roadmap/` | ROADMAP、DAY01、CTC 转移图 SVG | 学习过程的**可验证证据**（含 bug 归因过程） |
| `notes/` | E2E ASR 综述精读笔记 | 科研能力的物证 |
| `docs/ENV.md` | 环境基线唯一真相来源 | 换机器可一键复原 |
| `scripts/test/` | 手写 Fbank / CTC + 两个对拍工具 | **核心自写代码**，证明底层理解 |
| `scripts/*.ps1` `tools/` | 环境搭建、论文批量下载 | 工程化能力 |
| `coding/` | build.py + P01 题面/解/测试 | 算法基本功，含审查与修正记录 |
| `.vscode/` | 任务与调试配置 | 开箱可跑 |
| `.gitignore` `.gitattributes` | 忽略规则、换行统一 | 工程规范 |

### ❌ 不入库及理由

| 路径 | 体积 | 理由 |
| --- | --- | --- |
| `papers/` | 46 MB / 43 篇 PDF | 版权 + 体积；**清单保留在 `papers/README.md`**，可用 `scripts/download_papers.py` 一条命令重新拉全 |
| `examples/wenet-main/` | 26 MB / 1147 文件 | 第三方源码，非本人成果，tarball 可随时复原 |
| `.venv/` | 4924 MB / 34362 文件 | 虚拟环境，`scripts/setup_env.ps1` 可重建 |
| `coding/build/` | — | 编译产物（`.exe`/`.obj`） |
| `__pycache__/` | — | Python 字节码 |

> 💡 关键取舍：**论文不入库但保留清单和下载脚本**。
> 这样仓库轻量（首次推送秒级完成），而"我读过 43 篇论文"这件事仍然可复现、可证明。

## 本次新增的配置文件

- **`.gitignore`** —— 分七类写明排除规则，每条带注释说明"为什么排除、如何复原"
- **`.gitattributes`** —— `* text=auto eol=lf`，强制仓库内存 LF。
  因为脚本将来要在 Linux（RK3576 板端）上跑，若 Git 自动转 CRLF 会引入 `^M` 让 shebang 失效

## 遇到的问题

### 问题 3：两个独立仓库，无法合并

- 现象：`F:\embedded\prepare` 的 `first commit` 是 `124f495`，远端的却是 `c152504`，
  `git push` 会因历史不相关被拒，只能用 `--force`（会覆盖远端）。
- 原因：第一次 `git init` 在 `Desktop` 下执行，那是个"空壳远程仓库"；
  项目真正的开发目录是另一个仓库，两者从未关联。
- 解决：不做合并、不做 force push —— 直接把项目仓库的 `main` 重置到远端提交上，
  在其上追加提交，实现快进推送（见上文命令 1~5）。

### 问题 4：Git Bash 里 `git remote -v` 仍是 HTTPS

- 现象：项目仓库继承的 `origin` 是 `https://github.com/OGpig/STUDY.git`，
  而第一次操作已确认 HTTPS 走不通（连接被重置）。
- 原因：`git init` 时的默认远端配置，未同步第一次操作里改好的 SSH 443 地址。
- 解决：`git remote set-url origin ssh://git@ssh.github.com:443/OGpig/STUDY.git`。
  推送前先用 `git ls-remote origin` 探测可达性（返回 ref 列表即通）。

### 问题 5：`git add` 刷屏 "LF will be replaced by CRLF"

- 现象：35 个文件全部提示换行符将被替换。
- 原因：Windows 上 `core.autocrlf` 默认为 `true`，Git 想把仓库里的 LF 在检出时换成 CRLF。
- 解决：新增 `.gitattributes` 固定 `eol=lf`，用
  `git add --renormalize .` 让已暂存文件按新规则重新规范化。

## 验证结果

```
$ git ls-remote origin
1c120206c115350542877827b9d6a4391f82b684  HEAD
1c120206c115350542877827b9d6a4391f82b684  refs/heads/main

$ git status -sb
## main                      ← 无 ahead/behind，本地与远端完全一致

$ git ls-tree -r --name-only origin/main | wc -l
36                          ← 与预期一致，无遗漏、无多余
```

## 可复用的经验

1. **`git init` 前先 `cd` 到正确的项目目录** —— 本次所有麻烦的根源就是这一步做错了。
   桌面不是项目目录，`git init` 会静默创建一个"孤儿仓库"。
2. **不要把仓库建在用户主目录 / 桌面** —— 这些目录下文件极多，
   `git status` 会列出上百个无关文件，且容易误提交隐私内容。
3. **推不上去时先分清是"认证问题"还是"历史问题"** ——
   前者表现为 `Permission denied` / `Connection reset`，后者表现为 `non-fast-forward` 或 `unrelated histories`。
   本次两轮分别踩了这两种，解法完全不同。
4. **能用快进就别用 `--force`** —— `update-ref` 重置到远端提交再增量提交，
   既不丢远端历史，也不需要在共享仓库上冒险。
5. **"体积大所以不传" 要配一个"可复原方案"** ——
   保留清单 + 下载脚本，比硬塞几十 MB PDF 更专业。

---

# 第三次操作：D2 归档推送（数据准备 + 训练跑通）

## 基本信息

| 项目 | 内容 |
| --- | --- |
| 日期 | 2026-09-21 |
| 系统 / 终端 | Windows / PowerShell |
| 工作目录 | `F:\embedded\prepare` |
| 仓库 | STUDY (`ssh://git@ssh.github.com:443/OGpig/STUDY.git`) |
| 操作前 | 本地与远端同为 `4bf846b`（`git ls-remote` 探测可达 ✅） |
| 提交 | `85a6760`（D2 归档）+ 本次 git-log 记录 |
| 推送结果 | `4bf846b..85a6760  main -> main` ✅ 快进推送，无需 force |

## 本次入库内容（25 个文件 / 6949 行）

| 类别 | 文件 |
| --- | --- |
| 计划 / 进度 | `roadmap/DAY02.md`、`PROGRESS.md`（补 D2 收尾段 + 看板）、`EXPERIMENTS.md`（新建） |
| 可视图 | `roadmap/assets/DAY02_conformer-streaming-flow.svg`、`tensorboard/smoke/curves.png` |
| 数据管线 | `scripts/download_aishell.py` / `check_aishell.py` / `prepare_aishell.py` / `run_prepare.ps1` |
| 训练链路 | `scripts/run_train.py`、`patch_wenet.py`（25 处）、`patch_torch_libuv.py`（9 处）、`install_env_fix.py` |
| 绘图工具 | `scripts/plot_tensorboard.py`、`plot_ctc_alignment.py` |
| 诊断脚本 | `scripts/_diag_batchcount.py`、`_diag_dl_worker.py`（结论可复现，故保留） |
| 配置 | `work/aishell/conf/{smoke,mid}.yaml`（手写配置，首次入库） |
| 记录 | `.workbuddy/memory/2026-09-18.md`、`MEMORY.md`、`2026-09-20.md` |

## 遇到的问题

### 问题 6：TensorBoard 的 event 文件该不该入库

- 现象：`work/aishell/tensorboard/` 下的 `events.out.tfevents.*` 是二进制，且**随训练线性增长**
  （本次两个文件 1.2 MB + 45 KB）。若不处理，`git add -A` 会把它们全部提交。
- 原因：`.gitignore` 只按目录名排除了 `data/` `exp/` `runs/`，没有覆盖 `tensorboard/` 下的运行产物。
- 解决：新增一条**窄规则**而不是整目录排除：
  ```
  events.out.tfevents.*
  ```
  理由是 `curves.png`（曲线图）是**交付证据**要留，而 event 文件能由训练重放再生 ——
  与"只跟踪人写的东西"原则一致，也避免二进制文件污染 diff。

### 问题 5（复现）：`git add` 再报 CRLF 警告

- 现象：`work/aishell/conf/*.yaml` 两个文件提示 "CRLF will be replaced by LF"。
- 说明：与第二次操作的**问题 5 同源**（Windows `core.autocrlf=true`），
  已有 `.gitattributes`（`* text=auto eol=lf`）兜住，仓库内存 LF，**属预期行为，无需处理**。

## 可复用的经验

1. **推送前先 `git ls-remote origin` 探测** —— 一条命令同时确认「认证通 + 远端 HEAD 在哪」，
   比直接 push 失败后再排查快得多（本次远端 `4ff...` 与本地一致，直接快进）。
2. **`git add -A --dry-run` 必须先看一眼** —— 这是唯一能拦住"误提交大文件 / 隐私文件"的关口。
   本次正是靠它发现 event 文件会被带上。
3. **"归档"要连进度文档一起归档** —— 只推代码不推进度，下次自己都看不出当时做到哪。
4. **诊断脚本也值得入库** —— 它们记录的"当时怎么证明的"比结论更容易被遗忘。

## 本次归档涉及的工程问题（详情见 `PROGRESS.md`，此处只列索引）

| 问题 | 一句话结论 |
| --- | --- |
| torchaudio 2.11 解码需 FFmpeg 共享库 | 装 shared 构建 + `.pth` 注册 DLL 目录，WeNet 源码零改动 |
| `data.list` 被按 GBK 读 → 静默丢光样本 | 表现为 `ZeroDivisionError`；补 `encoding='utf-8'` |
| Windows gloo 无 CUDA → DDP 段错误 | 单卡跳过 DDP；**多卡在 Windows 走不通，要上 Linux** |
| `use_libuv` 报错（提示设 `USE_LIBUV=0` 是误导） | 必须改源码给 TCPStore 显式传 `use_libuv=False` |
| `final.pt` 是断链（WeNet 自身两个 bug） | 用 `lexists` + 目标名改 `epoch_N.pt` |
| CER=100% 但 loss 在降 | **判据必须是任务指标 + 平凡基线，不能只看 loss** |