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