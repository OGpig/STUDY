# setup_env.ps1 -- one-shot rebuild of this project's Python environment.
#
# Why this script exists:
#   1. The venv MUST be built from a NON-conda Python, otherwise it inherits
#      anaconda's DLL search path and breaks later (see docs/ENV.md risk 1).
#   2. pip MUST be upgraded FIRST, and via the official index -- a stale pip
#      bundled in a fresh venv cannot read the tuna mirror and reports
#      "from versions: none" for every package (see docs/ENV.md risk 2).
#   3. torch MUST come from the pytorch index (cu128) and be installed in a
#      SEPARATE command from numpy/pandas, because neither index carries both.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File F:\embedded\prepare\scripts\setup_env.ps1
#   powershell -ExecutionPolicy Bypass -File ...\setup_env.ps1 -Force   # wipe & rebuild
#
# NOTE: keep this file ASCII-only (Windows PowerShell 5.1 reads .ps1 as ANSI
#       unless it has a UTF-8 BOM).

param(
    [string]$VenvDir = "F:\embedded\prepare\.venv",
    [string]$PyBase  = "C:\Users\ADMIN\.workbuddy\binaries\python\versions\3.13.12\python.exe",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$VenvPy = Join-Path $VenvDir "Scripts\python.exe"

function Step($msg) { Write-Host ""; Write-Host "=== $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "[FAIL] $msg" -ForegroundColor Red; exit 1 }

# --- 0) sanity ---------------------------------------------------------------
Step "0/5  checking base interpreter"
if (-not (Test-Path $PyBase)) { Fail "base interpreter not found: $PyBase" }
if ($PyBase -match 'anaconda|conda') { Fail "base interpreter must NOT be a conda Python: $PyBase" }
& $PyBase -c "import sys; print('  base python:', sys.version)"
& $PyBase -c "import sys; print('  base prefix:', sys.base_prefix)"

$rebuild = $true
if ((Test-Path $VenvPy) -and (-not $Force)) {
    Write-Host "  venv already exists -- skipping creation (use -Force to wipe and rebuild)" -ForegroundColor Yellow
    $rebuild = $false
}

# --- 1) create venv ----------------------------------------------------------
if ($rebuild) {
    Step "1/5  creating venv (--clear) at $VenvDir"
    & $PyBase -m venv --clear $VenvDir
    if (-not (Test-Path $VenvPy)) { Fail "venv creation failed" }
} else {
    Step "1/5  venv creation skipped"
}

# --- 2) upgrade pip FIRST, via official index --------------------------------
Step "2/5  upgrading pip via https://pypi.org/simple"
Write-Host "  (do NOT use a mirror here -- a stale pip cannot read mirrors)" -ForegroundColor DarkGray
& $VenvPy -m pip install -U pip --index-url https://pypi.org/simple
& $VenvPy -m pip --version

# --- 3) torch (pytorch index, cu128 for sm_120 / RTX 5060) --------------------
Step "3/5  installing torch + torchaudio (cu128)"
& $VenvPy -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# --- 4) the rest (default index, i.e. the mirror configured in pip.ini) -------
Step "4/5  installing numpy pandas soundfile librosa tensorboard"
& $VenvPy -m pip install numpy pandas soundfile librosa tensorboard

# --- 5) verify ---------------------------------------------------------------
Step "5/5  verification"
$base = (& $VenvPy -c "import sys; print(sys.base_prefix)").Trim()
Write-Host "  sys.executable   : " -NoNewline; & $VenvPy -c "import sys; print(sys.executable)"
Write-Host "  sys.base_prefix  : $base"
if ($base -match 'anaconda') {
    Write-Host "[FAIL] venv is still bound to anaconda -- rebuild with a non-conda Python" -ForegroundColor Red
    exit 1
} else {
    Write-Host "  OK: venv is independent of conda" -ForegroundColor Green
}

& $VenvPy -c @"
import torch, torchaudio
print('  torch      :', torch.__version__)
print('  torchaudio :', torchaudio.__version__)
print('  cuda avail :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  device     :', torch.cuda.get_device_name(0))
    print('  capability :', torch.cuda.get_device_capability(0))
    a = torch.randn(1024, 1024, device='cuda')
    b = torch.randn(1024, 1024, device='cuda')
    print('  matmul sum :', float((a @ b).sum()))
    torch.cuda.synchronize()
"@

Write-Host ""
Write-Host "Done. Activate with:" -ForegroundColor Cyan
Write-Host "  powershell -ExecutionPolicy Bypass -File F:\embedded\prepare\scripts\activate_project.ps1"
