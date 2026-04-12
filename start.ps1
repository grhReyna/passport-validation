<#
.SYNOPSIS
    Instala dependencias y levanta la aplicacion de validacion de pasaportes.
.DESCRIPTION
    Script de inicio rapido para Windows (PowerShell).
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

# Si el venv ya existe, verificar que sea valido
if (Test-Path $venvPython) {
    Write-Host "  Entorno virtual ya existe" -ForegroundColor Gray
    try {
        & $venvPython --version >$null 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  OK: Entorno virtual valido" -ForegroundColor Green
        } else {
            throw "Entorno virtual corrupto"
        }
    } catch {
        Write-Host "  [!] Entorno virtual danado, recreando..." -ForegroundColor Yellow
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
    
    # Verificar si pip esta funcionando correctamente
    $pipCheck = & $venvPython -m pip --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [!] Error en pip, recreando entorno virtual..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $venvPath -ErrorAction SilentlyContinue
        Start-Sleep 1
        & $pythonCmd -m venv .venv
    }
    
    # Actualizar pip
    & $venvPython -m pip install --upgrade pip --quiet >$null 2>&1

    # ---- Funcion auxiliar para verificar si un modulo ya esta instalado ----
    function Test-PythonModule($module) {
        & $venvPython -c "import $module" >$null 2>&1
        return ($LASTEXITCODE -eq 0)
    }

    # ================================================================
    # Instalacion por ETAPAS (evita timeouts en conexiones lentas)
    # Los paquetes pesados (torch ~2GB, easyocr, transformers) se
    # instalan por separado con feedback visual.
    # ================================================================

    # --- Etapa 1: Paquetes ligeros (API, utilidades) ---
    Write-Host "  [3a] Paquetes ligeros (API, utilidades)..." -ForegroundColor Gray
    $lightPkgs = "fastapi>=0.104.1", "uvicorn>=0.24.0", "python-multipart>=0.0.6",
                  "pydantic>=2.4.2", "pillow>=10.0.0", "pandas>=2.0.3",
                  "huggingface_hub>=0.20.0", "kagglehub>=0.2.1"
    $needLight = $false
    foreach ($pkg in @("fastapi", "uvicorn", "PIL", "pandas")) {
        if (-not (Test-PythonModule $pkg)) { $needLight = $true; break }
    }
    if ($needLight) {
        & $venvPython -m pip install $lightPkgs --quiet 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  [!] Reintentando paquetes ligeros..." -ForegroundColor Yellow
            & $venvPython -m pip install $lightPkgs 2>&1 | Out-Null
        }
    }
    Write-Host "  OK: Paquetes ligeros" -ForegroundColor Green

    # --- Etapa 2: Vision computacional (opencv, numpy, scipy) ---
    Write-Host "  [3b] Vision computacional (opencv, numpy, scipy)..." -ForegroundColor Gray
    if (-not (Test-PythonModule "cv2")) {
        & $venvPython -m pip install "opencv-python-headless>=4.8.0" --quiet 2>&1 | Out-Null
    }
    # numpy <2.0 es requerido por opencv y torch
    # NOTA: version specs con < se guardan en variables para evitar que PS 5.1
    # interprete < como operador de redireccion
    $numpySpec = 'numpy>=1.24.3,<2.0'
    $scipySpec = 'scipy>=1.11.2,<1.14'
    & $venvPython -m pip install $numpySpec --quiet 2>&1 | Out-Null
    if (-not (Test-PythonModule "scipy")) {
        & $venvPython -m pip install $scipySpec --quiet 2>&1 | Out-Null
    }
    if (-not (Test-PythonModule "skimage")) {
        & $venvPython -m pip install "scikit-image>=0.21.0" --quiet 2>&1 | Out-Null
    }
    Write-Host "  OK: Vision computacional" -ForegroundColor Green

    # --- Etapa 3: PyTorch (el mas pesado, ~2GB) ---
    Write-Host "  [3c] PyTorch (puede tardar varios minutos)..." -ForegroundColor Gray
    $hasNvidiaGpu = $false
    try {
        $nvidiaSmi = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
        if ($LASTEXITCODE -eq 0 -and $nvidiaSmi) {
            $hasNvidiaGpu = $true
            Write-Host "  GPU detectada: $($nvidiaSmi.Trim())" -ForegroundColor Cyan
        }
    } catch { }

    $torchOk = $false
    if (Test-PythonModule "torch") {
        if ($hasNvidiaGpu) {
            $torchCuda = & $venvPython -c "import torch; print(torch.cuda.is_available())" 2>$null
            if ($torchCuda -eq "True") {
                Write-Host "  OK: PyTorch CUDA ya activo" -ForegroundColor Green
                $torchOk = $true
            }
        } else {
            Write-Host "  OK: PyTorch ya instalado (CPU)" -ForegroundColor Green
            $torchOk = $true
        }
    }

    if (-not $torchOk) {
        if ($hasNvidiaGpu) {
            Write-Host "  Instalando PyTorch con CUDA (GPU)..." -ForegroundColor Yellow
            & $venvPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>&1 | Out-Null
        } else {
            Write-Host "  Instalando PyTorch (CPU)..." -ForegroundColor Yellow
            & $venvPython -m pip install torch torchvision 2>&1 | Out-Null
        }
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  [!] Error instalando PyTorch. Verifica tu conexion." -ForegroundColor Yellow
        } else {
            Write-Host "  OK: PyTorch instalado" -ForegroundColor Green
        }
        # Corregir numpy (torch puede arrastrar numpy 2.x)
        & $venvPython -m pip install $numpySpec --force-reinstall --no-deps --quiet 2>&1 | Out-Null
    }

    # --- Etapa 4: Modelos NLP/OCR (transformers, easyocr) ---
    Write-Host "  [3d] Modelos OCR (transformers, easyocr)..." -ForegroundColor Gray
    if (-not (Test-PythonModule "transformers")) {
        & $venvPython -m pip install "transformers>=4.35.0" --quiet 2>&1 | Out-Null
    }
    if (-not (Test-PythonModule "easyocr")) {
        & $venvPython -m pip install "easyocr>=1.7.0" --quiet 2>&1 | Out-Null
    }
    Write-Host "  OK: Modelos OCR" -ForegroundColor Green

    # --- Verificacion final ---
    $allOk = & $venvPython -c "import uvicorn, fastapi, cv2, torch, easyocr, transformers; print('OK')" 2>$null
    if ($allOk -eq "OK") {
        Write-Host "  [OK] Todas las dependencias verificadas" -ForegroundColor Green
    } else {
        Write-Host "  [!] Algunas dependencias pueden faltar. El servidor intentara iniciar." -ForegroundColor Yellow
    }
} else {
    Write-Host '[3/5] Saltando instalacion (-SkipInstall)' -ForegroundColor Gray
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
    Write-Host "[X] ERROR: No se pudo iniciar el servidor" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Intentando limpiar y reinstalar..." -ForegroundColor Yellow
    
    Remove-Item -Recurse -Force $venvPath -ErrorAction SilentlyContinue
    Start-Sleep 1
    
    Write-Host "Recreando entorno virtual..." -ForegroundColor Gray
    & $pythonCmd -m venv .venv
    & $venvPython -m pip install --upgrade pip --quiet >$null 2>&1
    & $venvPython -m pip install fastapi uvicorn python-multipart pydantic pillow --quiet 2>&1 | Out-Null
    $npSpec = 'numpy>=1.24.3,<2.0'
    & $venvPython -m pip install "opencv-python-headless>=4.8.0" $npSpec --quiet 2>&1 | Out-Null
    & $venvPython -m pip install torch torchvision 2>&1 | Out-Null
    & $venvPython -m pip install $npSpec --force-reinstall --no-deps --quiet 2>&1 | Out-Null
    & $venvPython -m pip install "transformers>=4.35.0" "easyocr>=1.7.0" --quiet 2>&1 | Out-Null
    
    Write-Host "Reiniciando servidor..." -ForegroundColor Gray
    Write-Host ""
    & $venvPython -m uvicorn app:app --host 127.0.0.1 --port $Port
}
