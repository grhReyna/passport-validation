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

$ErrorActionPreference = "Continue"
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

# Si el venv ya existe, verificar que sea válido
if (Test-Path $venvPython) {
    Write-Host "  Entorno virtual ya existe" -ForegroundColor Gray
    try {
        & $venvPython --version >$null 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  OK: Entorno virtual válido" -ForegroundColor Green
        } else {
            throw "Entorno virtual corrupto"
        }
    } catch {
        Write-Host "  ⚠️ Entorno virtual dañado, recreando..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $venvPath -ErrorAction SilentlyContinue
        Start-Sleep 1
        & $pythonCmd -m venv .venv
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  ERROR: No se pudo crear el entorno virtual." -ForegroundColor Red
            exit 1
        }
        Write-Host "  OK: Entorno virtual recreado" -ForegroundColor Green
    }
} else {
    Write-Host "  Creando .venv..." -ForegroundColor Gray
    & $pythonCmd -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: No se pudo crear el entorno virtual." -ForegroundColor Red
        exit 1
    }
    Write-Host "  OK: Entorno virtual creado" -ForegroundColor Green
}

# --- 3. Instalar dependencias ---
if (-not $SkipInstall) {
    Write-Host "[3/5] Instalando dependencias (esto puede tardar unos minutos)..." -ForegroundColor Yellow
    
    # Verificar si pip está funcionando correctamente
    $pipCheck = & $venvPython -m pip --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ⚠️ Error en pip, recreando entorno virtual..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $venvPath -ErrorAction SilentlyContinue
        Start-Sleep 1
        & $pythonCmd -m venv .venv
    }
    
    # Actualizar pip
    & $venvPython -m pip install --upgrade pip --quiet >$null 2>&1
    
    # Detectar GPU NVIDIA e instalar PyTorch con CUDA si es posible
    $hasNvidiaGpu = $false
    try {
        $nvidiaSmi = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
        if ($LASTEXITCODE -eq 0 -and $nvidiaSmi) {
            $hasNvidiaGpu = $true
            Write-Host "  GPU detectada: $($nvidiaSmi.Trim())" -ForegroundColor Cyan
        }
    } catch { }
    
    if ($hasNvidiaGpu) {
        # Verificar si ya tiene torch con CUDA
        $torchCuda = & $venvPython -c "import torch; print(torch.cuda.is_available())" 2>$null
        if ($torchCuda -ne "True") {
            Write-Host "  Instalando PyTorch con soporte CUDA (GPU)..." -ForegroundColor Yellow
            & $venvPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet >$null 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Host "  OK: PyTorch CUDA instalado" -ForegroundColor Green
            } else {
                Write-Host "  ⚠️ No se pudo instalar PyTorch CUDA, usando CPU" -ForegroundColor Yellow
            }
        } else {
            Write-Host "  OK: PyTorch CUDA ya instalado" -ForegroundColor Green
        }
    }
    
    # Verificar si requirements.txt existe
    if (Test-Path "requirements.txt") {
        # Instalar dependencias (pip se encarga de verificar si ya están)
        & $venvPython -m pip install -r requirements.txt --quiet >$null 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  ⚠️ Error instalando dependencias, recreando..." -ForegroundColor Yellow
            Remove-Item -Recurse -Force $venvPath -ErrorAction SilentlyContinue
            Start-Sleep 1
            
            & $pythonCmd -m venv .venv
            & $venvPython -m pip install --upgrade pip --quiet >$null 2>&1
            & $venvPython -m pip install -r requirements.txt --quiet >$null 2>&1
            
            if ($LASTEXITCODE -ne 0) {
                Write-Host "  ERROR: No se pudieron instalar las dependencias." -ForegroundColor Red
                Write-Host "  Revisa tu conexión a internet y requirements.txt" -ForegroundColor Red
                exit 1
            }
        }
    } else {
        Write-Host "  ⚠️ requirements.txt no encontrado" -ForegroundColor Yellow
    }
    
    Write-Host "  OK: Dependencias listas" -ForegroundColor Green
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

try {
    & $venvPython -m uvicorn app:app --host 127.0.0.1 --port $Port
} catch {
    Write-Host ""
    Write-Host "❌ ERROR: No se pudo iniciar el servidor" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Intentando limpiar y reinstalar..." -ForegroundColor Yellow
    
    Remove-Item -Recurse -Force $venvPath -ErrorAction SilentlyContinue
    Start-Sleep 1
    
    Write-Host "Recreando entorno virtual..." -ForegroundColor Gray
    & $pythonCmd -m venv .venv
    & $venvPython -m pip install --upgrade pip --quiet >$null 2>&1
    & $venvPython -m pip install -r requirements.txt --quiet >$null 2>&1
    
    Write-Host "Reiniciando servidor..." -ForegroundColor Gray
    Write-Host ""
    & $venvPython -m uvicorn app:app --host 127.0.0.1 --port $Port
}
