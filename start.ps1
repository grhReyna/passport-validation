<#
.SYNOPSIS
    Instala dependencias y levanta la aplicación de validación de pasaportes.
.DESCRIPTION
    Script de inicio rápido para Windows (PowerShell).
    Crea entorno virtual, instala dependencias, crea directorios necesarios
    y arranca el servidor en http://127.0.0.1:9000
.EXAMPLE
    .\start.ps1
    .\start.ps1 -Port 8080
    .\start.ps1 -SkipInstall
#>
param(
    [int]$Port = 9000,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Passport Validation - Setup & Launch" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# --- 1. Verificar Python ---
Write-Host "[1/5] Verificando Python..." -ForegroundColor Yellow
$pythonCmd = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python 3\.(\d+)") {
            $minor = [int]$Matches[1]
            if ($minor -ge 9) {
                $pythonCmd = $cmd
                Write-Host "  OK: $ver" -ForegroundColor Green
                break
            }
        }
    } catch { }
}
if (-not $pythonCmd) {
    Write-Host "  ERROR: Se requiere Python 3.9+. Instala desde https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

# --- 2. Crear entorno virtual ---
Write-Host "[2/5] Configurando entorno virtual..." -ForegroundColor Yellow
$venvPath = Join-Path $ProjectRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"
$venvPip = Join-Path $venvPath "Scripts\pip.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "  Creando .venv..." -ForegroundColor Gray
    & $pythonCmd -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: No se pudo crear el entorno virtual." -ForegroundColor Red
        exit 1
    }
    Write-Host "  OK: Entorno virtual creado" -ForegroundColor Green
} else {
    Write-Host "  OK: Entorno virtual ya existe" -ForegroundColor Green
}

# --- 3. Instalar dependencias ---
if (-not $SkipInstall) {
    Write-Host "[3/5] Instalando dependencias (esto puede tardar unos minutos)..." -ForegroundColor Yellow
    & $venvPip install --upgrade pip --quiet 2>&1 | Out-Null
    & $venvPip install -r requirements.txt --quiet 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ADVERTENCIA: Algunas dependencias fallaron. Intentando instalar las esenciales..." -ForegroundColor DarkYellow
        $essentials = @("torch", "torchvision", "transformers", "opencv-python", "pillow", "numpy", "fastapi", "uvicorn", "python-multipart", "pydantic", "scikit-image", "scipy")
        foreach ($pkg in $essentials) {
            & $venvPip install $pkg --quiet 2>&1 | Out-Null
        }
    }
    Write-Host "  OK: Dependencias instaladas" -ForegroundColor Green
} else {
    Write-Host "[3/5] Saltando instalacion (--SkipInstall)" -ForegroundColor Gray
}

# --- 4. Crear directorios necesarios ---
Write-Host "[4/5] Verificando estructura de directorios..." -ForegroundColor Yellow
$dirs = @(
    "data\raw", "data\processed", "data\results",
    "data\raw\test", "data\raw\train", "data\raw\validation",
    "models", "static"
)
foreach ($d in $dirs) {
    $fullPath = Join-Path $ProjectRoot $d
    if (-not (Test-Path $fullPath)) {
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
    }
}
Write-Host "  OK: Directorios verificados" -ForegroundColor Green

# --- 5. Iniciar servidor ---
Write-Host "[5/5] Iniciando servidor..." -ForegroundColor Yellow
Write-Host ""

# Matar procesos previos en el mismo puerto
$existingPython = Get-Process -Name python -ErrorAction SilentlyContinue
if ($existingPython) {
    Write-Host "  Deteniendo procesos Python previos..." -ForegroundColor Gray
    $existingPython | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep 2
}

Write-Host "================================================" -ForegroundColor Green
Write-Host "  Servidor iniciando en: http://127.0.0.1:$Port" -ForegroundColor Green
Write-Host "  Presiona Ctrl+C para detener" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""

& $venvPython -m uvicorn app:app --host 127.0.0.1 --port $Port
