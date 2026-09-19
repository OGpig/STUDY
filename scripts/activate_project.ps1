# activate_project.ps1
# Purpose: isolate this project's venv from conda/anaconda, then activate it.
#
# What it does
#   1. strip every conda/anaconda entry out of PATH
#   2. delete ALL conda-injected environment variables
#      (CONDA_PROMPT_MODIFIER is the one that renders the "(base)" prefix)
#   3. activate the project venv
#   4. replace conda's prompt() wrapper with a clean one
#   5. point pip cache / TMP / TEMP at the F: drive
#   6. verify and report
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File F:\embedded\prepare\scripts\activate_project.ps1
#   # or, inside an existing session:
#   . F:\embedded\prepare\scripts\activate_project.ps1
#
# Why step 4 is needed:
#   `conda init powershell` writes a block into $PROFILE that re-injects conda's
#   prompt() on EVERY new shell. Removing the env vars alone is not enough,
#   because a new shell loads the profile again. Re-defining prompt() here makes
#   the fix stick for this session.
#   >>> If you ever write your own custom prompt, delete step 4. <<<
#
# NOTE: keep this file ASCII-only. Windows PowerShell 5.1 reads .ps1 as ANSI
#       unless it has a UTF-8 BOM, so non-ASCII comments would be garbled.

$ErrorActionPreference = "Stop"

$VenvDir     = "F:\embedded\prepare\.venv"
$CondaPrefix = "F:\anaconda3"

# --- 1) strip conda/anaconda entries out of PATH -----------------------------
$kept = $env:PATH -split ';' | Where-Object {
    $_ -and ($_ -notlike "$CondaPrefix*") -and ($_ -notmatch 'anaconda|conda')
}
$env:PATH = ($kept -join ';')

# --- 2) drop ALL conda-injected variables -----------------------------------
$named = @('CONDA_PREFIX','CONDA_DEFAULT_ENV','CONDA_SHLVL','CONDA_EXE',
           'CONDA_PYTHON_EXE','CONDA_PROMPT_MODIFIER','_CONDA_EXE','_CE_CONDA',
           '_CE_M','PYTHONPATH','PYTHONHOME')
foreach ($v in $named) {
    if (Test-Path "Env:$v") { Remove-Item "Env:$v" -Force }
}
# sweep anything else conda left behind (CONDA_STACKED_*, etc.)
Get-ChildItem Env: | Where-Object { $_.Name -like 'CONDA*' -or $_.Name -like '_CE_*' } | ForEach-Object {
    Remove-Item "Env:$($_.Name)" -Force -ErrorAction SilentlyContinue
}

# --- 3) activate the venv (plain PATH prepend, no conda activate needed) -----
$env:VIRTUAL_ENV = $VenvDir
$env:PATH = "$VenvDir\Scripts;$env:PATH"

# --- 4) replace conda's prompt wrapper with a clean one ----------------------
function global:prompt {
    $venv = ''
    if ($env:VIRTUAL_ENV) { $venv = '(' + (Split-Path $env:VIRTUAL_ENV -Leaf) + ') ' }
    "$venv$($executionContext.SessionState.Path.CurrentLocation)$('>' * ($nestedPromptLevel + 1)) "
}

# --- 5) project env vars: keep caches and temp off the C: drive --------------
$env:PIP_CACHE_DIR = "F:\data\WB\cache\pip"
$env:TMP  = "F:\data\WB\tmp"
$env:TEMP = "F:\data\WB\tmp"
New-Item -ItemType Directory -Force -Path $env:TMP | Out-Null

# --- 6) verify ---------------------------------------------------------------
Write-Host ""
Write-Host "[env] python      : " -NoNewline
& python -c "import sys; print(sys.executable)"
Write-Host "[env] base_prefix : " -NoNewline
& python -c "import sys; print(sys.base_prefix)"
Write-Host "[env] version     : " -NoNewline
& python --version

$residue = @($env:PATH -split ';' | Where-Object { $_ -match 'anaconda|conda' })
if ($residue.Count -gt 0) {
    Write-Host "[env] WARN  conda residue still in PATH:" -ForegroundColor Yellow
    $residue | ForEach-Object { Write-Host "              $_" -ForegroundColor Yellow }
} else {
    Write-Host "[env] OK    PATH is free of conda/anaconda" -ForegroundColor Green
}

if ($env:CONDA_PROMPT_MODIFIER) {
    Write-Host "[env] WARN  CONDA_PROMPT_MODIFIER still set: [$env:CONDA_PROMPT_MODIFIER]" -ForegroundColor Yellow
} else {
    Write-Host "[env] OK    prompt modifier cleared (no '(base)' prefix)" -ForegroundColor Green
}

$base = (& python -c "import sys; print(sys.base_prefix)").Trim()
if ($base -match 'anaconda') {
    Write-Host "[env] NOTICE venv is built on Anaconda's interpreter." -ForegroundColor Yellow
    Write-Host "             Rebuild with a non-conda Python (see docs/ENV.md)." -ForegroundColor Yellow
}
