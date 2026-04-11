#!/usr/bin/env bash
# ============================================================
# Passport Validation - Setup & Launch
# Instala dependencias y levanta la aplicación.
# Uso: ./start.sh [--port 8080] [--skip-install]
# ============================================================
set -e

PORT=9000
SKIP_INSTALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --port) PORT="$2"; shift 2 ;;
        --skip-install) SKIP_INSTALL=true; shift ;;
        *) echo "Uso: $0 [--port PORT] [--skip-install]"; exit 1 ;;
    esac
done

cd "$(dirname "$0")"

echo ""
echo "================================================"
echo "  Passport Validation - Setup & Launch"
echo "================================================"
echo ""

# --- 1. Verificar Python ---
echo "[1/5] Verificando Python..."
PYTHON_CMD=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        ver=$($cmd --version 2>&1)
        if echo "$ver" | grep -qE "Python 3\.(9|1[0-9]|[2-9][0-9])"; then
            PYTHON_CMD="$cmd"
            echo "  OK: $ver"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "  ERROR: Se requiere Python 3.9+."
    echo "  Instala desde https://www.python.org/downloads/"
    exit 1
fi

# --- 2. Crear entorno virtual ---
echo "[2/5] Configurando entorno virtual..."
if [ -f ".venv/bin/python" ]; then
    echo "  Entorno virtual ya existe"
    if .venv/bin/python --version >/dev/null 2>&1; then
        echo "  OK: Entorno virtual válido"
    else
        echo "  ⚠️ Entorno virtual dañado, recreando..."
        rm -rf .venv
        $PYTHON_CMD -m venv .venv
        echo "  OK: Entorno virtual recreado"
    fi
else
    echo "  Creando .venv..."
    $PYTHON_CMD -m venv .venv
    echo "  OK: Entorno virtual creado"
fi

VENV_PYTHON=".venv/bin/python"
VENV_PIP=".venv/bin/pip"

# --- 3. Instalar dependencias ---
if [ "$SKIP_INSTALL" = false ]; then
    echo "[3/5] Instalando dependencias (esto puede tardar unos minutos)..."
    
    # Verificar si pip está funcionando correctamente
    if ! $VENV_PYTHON -m pip --version >/dev/null 2>&1; then
        echo "  ⚠️ Error en pip, recreando entorno virtual..."
        rm -rf .venv
        sleep 1
        $PYTHON_CMD -m venv .venv
    fi
    
    # Actualizar pip
    $VENV_PYTHON -m pip install --upgrade pip --quiet >/dev/null 2>&1
    
    # Verificar si requirements.txt existe e instalar dependencias
    if [ -f "requirements.txt" ]; then
        # Instalar dependencias (pip se encarga de verificar si ya están)
        if ! $VENV_PYTHON -m pip install -r requirements.txt --quiet >/dev/null 2>&1; then
            echo "  ⚠️ Error instalando dependencias, recreando..."
            rm -rf .venv
            sleep 1
            
            $PYTHON_CMD -m venv .venv
            $VENV_PYTHON -m pip install --upgrade pip --quiet >/dev/null 2>&1
            $VENV_PYTHON -m pip install -r requirements.txt --quiet >/dev/null 2>&1
            
            if [ $? -ne 0 ]; then
                echo "  ❌ ERROR: No se pudieron instalar las dependencias."
                echo "  Revisa tu conexión a internet y requirements.txt"
                exit 1
            fi
        fi
    else
        echo "  ⚠️ requirements.txt no encontrado"
    fi
    
    echo "  OK: Dependencias listas"
else
    echo "[3/5] Saltando instalacion (--skip-install)"
fi

# --- 4. Crear directorios ---
echo "[4/5] Verificando estructura de directorios..."
mkdir -p data/raw data/processed data/results data/raw/test data/raw/train data/raw/validation models static
echo "  OK: Directorios verificados"

# --- 5. Iniciar servidor ---
echo "[5/5] Iniciando servidor..."
echo ""
echo "================================================"
echo "  Servidor iniciando en: http://127.0.0.1:$PORT"
echo "  Presiona Ctrl+C para detener"
echo "================================================"
echo ""

# Iniciar servidor con manejo de errores
if ! exec $VENV_PYTHON -m uvicorn app:app --host 127.0.0.1 --port "$PORT"; then
    echo ""
    echo "❌ ERROR: No se pudo iniciar el servidor"
    echo ""
    echo "Intentando limpiar y reinstalar..."
    rm -rf .venv
    sleep 1
    
    echo "Recreando entorno virtual..."
    $PYTHON_CMD -m venv .venv
    $VENV_PYTHON -m pip install --upgrade pip --quiet 2>&1 || true
    $VENV_PYTHON -m pip install -r requirements.txt --quiet 2>&1 || true
    
    echo "Reiniciando servidor..."
    echo ""
    exec $VENV_PYTHON -m uvicorn app:app --host 127.0.0.1 --port "$PORT"
fi
