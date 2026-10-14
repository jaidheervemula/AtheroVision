# AtheroVision: one-click setup and start (Windows PowerShell 5.1 or PowerShell 7).
#
#   Double-click start.bat, or run:  powershell -ExecutionPolicy Bypass -File .\start.ps1
#
# First run: creates the Python environment, installs the backend and frontend packages
# (internet needed, a few minutes). Later runs: starts straight away.
# It opens two windows (model service on :8000, web app on :5173) and then the browser.
# Close those two windows to stop AtheroVision.
#
#   -SetupOnly   install everything but do not start the servers

param([switch]$SetupOnly)

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$backend = Join-Path $root 'cardio_ai_hackathon'
$frontend = Join-Path $root 'frontend'
$onWindows = ($PSVersionTable.PSEdition -eq 'Desktop') -or $IsWindows

function Fail($message) {
    Write-Host ''
    Write-Host "  $message" -ForegroundColor Red
    Write-Host ''
    if (-not $SetupOnly) { Read-Host 'Press Enter to close' | Out-Null }
    exit 1
}
function Step($message) { Write-Host "==> $message" -ForegroundColor Cyan }
function Quote($path) { "'" + ($path -replace "'", "''") + "'" }   # for paths inside -Command strings

# ---------------------------------------------------------------- Python
if ($onWindows) { $venvPython = Join-Path $backend '.venv\Scripts\python.exe' }
else { $venvPython = Join-Path $backend '.venv/bin/python' }

if (-not (Test-Path $venvPython)) {
    # Find Python 3.12 or 3.13 (the versions this project is tested with).
    $candidates = @(@('py', '-3.13'), @('py', '-3.12'), @('python'), @('python3'))
    $basePython = $null
    foreach ($c in $candidates) {
        if (-not (Get-Command $c[0] -ErrorAction SilentlyContinue)) { continue }
        $extra = @($c | Select-Object -Skip 1)
        $version = & $c[0] @extra -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and ($version -eq '3.12' -or $version -eq '3.13')) { $basePython = $c; break }
    }
    if (-not $basePython) {
        Fail 'Python 3.12 or 3.13 was not found. Install Python 3.13 from https://www.python.org/downloads/ (tick "Add python.exe to PATH"), then run this again.'
    }
    Step "Creating the Python environment (Python $version)"
    $extra = @($basePython | Select-Object -Skip 1)
    & $basePython[0] @extra -m venv (Join-Path $backend '.venv')
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPython)) { Fail 'Could not create the Python environment.' }
}

& $venvPython -c "import fastapi, uvicorn, sklearn, pandas, openpyxl" 2>$null
if ($LASTEXITCODE -ne 0) {
    Step 'Installing backend packages (first run only; a few minutes)'
    & $venvPython -m pip install --disable-pip-version-check -r (Join-Path $root 'requirements-backend.txt')
    if ($LASTEXITCODE -ne 0) { Fail 'Installing the backend packages failed. Check the internet connection and run this again.' }
}

# ---------------------------------------------------------------- Node
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Fail 'Node.js was not found. Install the LTS version from https://nodejs.org/, then run this again.'
}
$envFile = Join-Path $frontend '.env'
if (-not (Test-Path $envFile)) { Copy-Item (Join-Path $frontend '.env.example') $envFile }
if (-not (Test-Path (Join-Path $frontend 'node_modules'))) {
    Step 'Installing frontend packages (first run only)'
    Push-Location $frontend
    npm install --no-fund --no-audit
    $npmExit = $LASTEXITCODE
    Pop-Location
    if ($npmExit -ne 0) { Fail 'npm install failed. Check the internet connection and run this again.' }
}

Step 'Setup complete'
if ($SetupOnly) { exit 0 }

# ---------------------------------------------------------------- Start
function Test-Backend {
    try { Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 2 | Out-Null; return $true } catch { return $false }
}

if (Test-Backend) {
    Step 'Model service already running on port 8000'
} else {
    Step 'Starting the model service (new window)'
    $cmd = "`$Host.UI.RawUI.WindowTitle = 'AtheroVision model service'; Set-Location -LiteralPath $(Quote $backend); & $(Quote $venvPython) -m uvicorn api.main:app --port 8000"
    Start-Process powershell -ArgumentList @('-NoExit', '-ExecutionPolicy', 'Bypass', '-Command', $cmd)
}

$webUp = $false
try { Invoke-WebRequest -Uri 'http://localhost:5173' -UseBasicParsing -TimeoutSec 2 | Out-Null; $webUp = $true } catch { }
if ($webUp) {
    Step 'Web app already running on port 5173'
} else {
    Step 'Starting the web app (new window)'
    $cmd = "`$Host.UI.RawUI.WindowTitle = 'AtheroVision web app'; Set-Location -LiteralPath $(Quote $frontend); npm run dev -- --strictPort"
    Start-Process powershell -ArgumentList @('-NoExit', '-ExecutionPolicy', 'Bypass', '-Command', $cmd)
}

Step 'Waiting for the models to load'
$ready = $false
for ($i = 0; $i -lt 90; $i++) {
    if (Test-Backend) { $ready = $true; break }
    Start-Sleep -Seconds 1
}
if (-not $ready) {
    Write-Host '  The model service did not answer within 90 seconds. Look at the "AtheroVision model service" window for an error.' -ForegroundColor Yellow
}
Start-Sleep -Seconds 2
Start-Process 'http://localhost:5173'
Write-Host ''
Write-Host '  AtheroVision is running at http://localhost:5173' -ForegroundColor Green
Write-Host '  To stop it, close the two AtheroVision windows.'
Write-Host ''
