# ============================================================
#  AISHELL-1 数据准备 —— 一键运行
#
#  用法（在任意 PowerShell 窗口）：
#      powershell -ExecutionPolicy Bypass -File F:\embedded\prepare\scripts\run_prepare.ps1
#  或先 cd 到项目再：
#      & .\scripts\run_prepare.ps1
#
#  这个脚本做三件事：
#    1. 把 FFmpeg 共享库目录加进 PATH —— **必须**。
#       torchaudio >= 2.9 靠 torchcodec 解码音频，torchcodec 需要
#       avcodec/avformat 这些 DLL；缺了训练会报 "Could not load libtorchcodec"。
#    2. 固定 UTF-8 与控制台编码，避免中文乱码。
#    3. 调 scripts/prepare_aishell.py --stage all
#
#  ⚠️ 两个 PowerShell 的坑（都踩过，别再踩）：
#    A. `$ErrorActionPreference = "Stop"` 会把**原生命令写到 stderr 的每一行**
#       当成终止性错误。本项目所有 Python 脚本按设计把信息写 stderr，
#       所以调用前必须把它改回 "Continue"，否则脚本会在第一次输出时中断。
#    B. 不设 [Console]::OutputEncoding 的话，重定向出来的中文是 GBK，
#       用 Git Bash / 编辑器按 UTF-8 读会全是乱码。
#
#  为什么不用 .bat：本机安全策略禁止 cmd.exe，.bat 跑不起来。
# ============================================================

$Root      = "F:\embedded\prepare"
$FFmpegBin = "F:\data\toolchains\ffmpeg-shared\ffmpeg-master-latest-win64-gpl-shared\bin"
$Python    = Join-Path $Root ".venv\Scripts\python.exe"

# ---- UTF-8（坑 B）
$env:PYTHONIOENCODING   = "utf-8"
$env:PYTHONUTF8         = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding           = [System.Text.Encoding]::UTF8

Write-Host "=== 环境准备 ===" -ForegroundColor Cyan

if (-not (Test-Path $Python)) {
    Write-Host "x 找不到 venv 的 python: $Python" -ForegroundColor Red
    Write-Host "  先跑 scripts\setup_env.ps1 重建环境"
    exit 1
}
Write-Host "  python  : $Python"

$ffmpegOk = Test-Path (Join-Path $FFmpegBin "avcodec-63.dll")
if ($ffmpegOk) {
    $env:PATH = "$FFmpegBin;$env:PATH"
    Write-Host "  ffmpeg  : 已加入 PATH" -ForegroundColor Green
    Write-Host "            $FFmpegBin"
} else {
    Write-Host "  ffmpeg  : 未找到共享库！训练时会报 libtorchcodec 加载失败" -ForegroundColor Yellow
    Write-Host "            期望路径: $FFmpegBin" -ForegroundColor Yellow
}

Write-Host "  编码    : UTF-8"
Write-Host "  工作目录: $Root"
Write-Host ""

Write-Host "=== 开始数据准备 ===" -ForegroundColor Cyan

# ---- 坑 A：调用原生命令前把 ErrorActionPreference 降级
#     否则 Python 脚本写到 stderr 的每一行都会变成终止性错误
$ErrorActionPreference = "Continue"

Set-Location $Root
& $Python "scripts\prepare_aishell.py" --stage all
$code = $LASTEXITCODE

Write-Host ""
if ($code -eq 0) {
    Write-Host "=== 完成（退出码 0）===" -ForegroundColor Green
    Write-Host "产物目录: $Root\work\aishell\data"
    Write-Host ""
    Write-Host "下一步：训练冒烟测试（命令见 roadmap\DAY02.md 任务 4）"
} else {
    Write-Host "=== 失败（退出码 $code）===" -ForegroundColor Red
}
exit $code
