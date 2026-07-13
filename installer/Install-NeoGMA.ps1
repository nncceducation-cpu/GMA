<#
    NeoGMA — one-click install. No Docker. No compiler. No terminal knowledge.

    WHAT THIS DOES, IN ORDER
      1. finds or installs Python 3.11
      2. makes a private virtual environment (nothing on your system is touched)
      3. detects whether you have an NVIDIA GPU, and installs the matching PyTorch
      4. downloads the ViTPose-H weights (1.3 GB, once)
      5. puts a "NeoGMA" shortcut on your Desktop

    WHY THERE IS NO mmcv HERE
    The research build of this tool depends on mmcv, which has no prebuilt wheel
    for the PyTorch/CUDA versions we need and therefore compiles from source —
    which requires Visual Studio Build Tools. Asking a clinical centre to install
    a C++ toolchain before it can score a baby is how a deployment dies. So the
    pose network is shipped as a single exported TorchScript file, and the app
    runs on torch + torchvision alone.

    WHAT LEAVES YOUR MACHINE: NOTHING.
    Video, keypoints and labels stay on this computer. Contributing to the
    multi-centre model is a separate, explicit step (see client/contribute.py),
    and even then only de-identified stick figures are sent — never video.
#>

$ErrorActionPreference = "Stop"
$AppName  = "NeoGMA"
$Root     = Join-Path $env:LOCALAPPDATA $AppName
$VenvDir  = Join-Path $Root "venv"
$ModelDir = Join-Path $Root "models"
$ModelTs  = Join-Path $ModelDir "vitpose_h.ts"
$RunsDir  = Join-Path $Root "runs"

# Where the weights live. Replace with your GitHub release asset URL.
$ModelUrl = $env:NEOGMA_MODEL_URL
if (-not $ModelUrl) {
  $ModelUrl = "https://github.com/nncceducation-cpu/GMA/releases/download/v0.1/vitpose_h.ts"
}

function Say($msg, $colour = "Cyan") { Write-Host "  $msg" -ForegroundColor $colour }

Write-Host ""
Write-Host "  ============================================" -ForegroundColor White
Write-Host "   NeoGMA — automated General Movements Assessment" -ForegroundColor White
Write-Host "  ============================================" -ForegroundColor White
Write-Host ""

New-Item -ItemType Directory -Force -Path $Root, $ModelDir, $RunsDir | Out-Null

# ---------------------------------------------------------------- 1. Python
Say "Looking for Python 3.11..."
$py = $null
foreach ($c in @("py -3.11", "python3.11", "python")) {
  try {
    $v = & cmd /c "$c --version" 2>$null
    if ($v -match "Python 3\.(1[0-2])") { $py = $c; Say "found: $v" "Green"; break }
  } catch {}
}
if (-not $py) {
  Say "Python not found. Installing via winget (this may take a few minutes)..." "Yellow"
  winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements --silent
  $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
              [Environment]::GetEnvironmentVariable("Path", "User")
  $py = "py -3.11"
  & cmd /c "$py --version" | Out-Null
  if ($LASTEXITCODE -ne 0) {
    Say "Could not install Python automatically." "Red"
    Say "Install Python 3.11 from python.org, tick 'Add to PATH', then run this again." "Red"
    Read-Host "Press Enter to close"; exit 1
  }
}

# ------------------------------------------------------- 2. isolated venv
Say "Creating a private environment (your system Python is not modified)..."
& cmd /c "$py -m venv `"$VenvDir`"" | Out-Null
$Pip    = Join-Path $VenvDir "Scripts\pip.exe"
$Python = Join-Path $VenvDir "Scripts\python.exe"
& $Pip install --quiet --upgrade pip

# ----------------------------------------------------------- 3. the GPU
$gpu = $false
try {
  $null = & nvidia-smi 2>$null
  if ($LASTEXITCODE -eq 0) { $gpu = $true }
} catch {}

if ($gpu) {
  Say "NVIDIA GPU detected — installing CUDA build of PyTorch (~3 GB)." "Green"
  & $Pip install --quiet torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
} else {
  Say "No NVIDIA GPU found — installing the CPU build." "Yellow"
  Say "It will work, but expect several minutes per clip instead of ~2." "Yellow"
  & $Pip install --quiet torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
}

Say "Installing the analysis stack..."
& $Pip install --quiet numpy scipy pandas scikit-learn matplotlib opencv-python-headless `
    pyarrow fastapi "uvicorn[standard]" python-multipart joblib requests

# ------------------------------------------------------ 4. ViTPose weights
if (Test-Path $ModelTs) {
  Say "Pose model already present." "Green"
} else {
  Say "Downloading ViTPose-H (1.3 GB, once)..." "Yellow"
  try {
    $ProgressPreference = "Continue"
    Invoke-WebRequest -Uri $ModelUrl -OutFile $ModelTs -UseBasicParsing
  } catch {
    Say "Could not download the pose model from:" "Red"
    Say "  $ModelUrl" "Red"
    Say "Download it manually and place it at:" "Red"
    Say "  $ModelTs" "Red"
    Read-Host "Press Enter to close"; exit 1
  }
}
$sizeGb = [math]::Round((Get-Item $ModelTs).Length / 1GB, 2)
Say "Pose model ready ($sizeGb GB)." "Green"

# ---------------------------------------------------------- 5. app + launcher
Say "Installing NeoGMA..."
$src = Split-Path -Parent $PSScriptRoot
Copy-Item -Recurse -Force (Join-Path $src "pipeline") $Root
Copy-Item -Recurse -Force (Join-Path $src "webapp")   $Root
Copy-Item -Recurse -Force (Join-Path $src "client")   $Root -ErrorAction SilentlyContinue

$launcher = Join-Path $Root "NeoGMA.cmd"
@"
@echo off
title NeoGMA
cd /d "$Root"
set NEOGMA_VITPOSE_TS=$ModelTs
set NEOGMA_DATA_DIR=$RunsDir
set NEOGMA_DEVICE=$(if ($gpu) {"cuda"} else {"cpu"})
echo Starting NeoGMA...  (leave this window open)
start "" http://localhost:8001
"$Python" -m uvicorn webapp.app:app --host 127.0.0.1 --port 8001
pause
"@ | Set-Content -Encoding ASCII $launcher

$ws = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut((Join-Path ([Environment]::GetFolderPath("Desktop")) "NeoGMA.lnk"))
$lnk.TargetPath = $launcher
$lnk.WorkingDirectory = $Root
$lnk.Description = "Automated General Movements Assessment"
$lnk.Save()

Write-Host ""
Say "Installed." "Green"
Write-Host ""
Say "  Double-click the NeoGMA icon on your Desktop." "White"
Say "  It opens http://localhost:8001 in your browser." "White"
Write-Host ""
Say "  Everything stays on this computer: video, keypoints and labels are" "Gray"
Say "  written to $RunsDir and are never uploaded." "Gray"
Write-Host ""
Read-Host "Press Enter to close"
