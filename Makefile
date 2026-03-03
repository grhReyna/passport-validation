# Makefile - Comandos útiles para el proyecto
# Usage: make <command>

.PHONY: help install setup test lint clean docs run run-api format train server

PYTHON := ./venv/Scripts/python
PIP := ./venv/Scripts/pip

help:
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  SISTEMA DE DETECCIÓN DE PASAPORTES FALSOS MEXICANOS       ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  📚 COMANDOS PRINCIPALES:"
	@echo ""
	@echo "  make install      - Instalar dependencias en virtualenv"
	@echo "  make train        - Entrenar modelo TrOCR (3 épocas)"
	@echo "  make server       - Iniciar servidor (http://127.0.0.1:9000)"
	@echo "  make test         - Ejecutar tests"
	@echo ""
	@echo "  📝 COMANDOS OPCIONALES:"
	@echo ""
	@echo "  make train-epochs-5    - Entrenamiento intensivo (5 épocas)"
	@echo "  make train-gpu         - Entrenamiento con GPU (batch_size 8)"
	@echo "  make train-cpu         - Entrenamiento con CPU (batch_size 2)"
	@echo "  make setup             - Preparar dataset"
	@echo "  make lint              - Validar código"
	@echo "  make format            - Formatear código"
	@echo "  make clean             - Limpiar temporales"
	@echo ""

install:
	@echo "Instalando dependencias..."
	$(PIP) install -r requirements.txt
	@echo "✓ Instalación completada"

train:
	@echo "🚀 INICIANDO ENTRENAMIENTO DEL MODELO"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "Dataset: dataset_training/images (44 imágenes)"
	@echo "Épocas: 3 | Batch: 4"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	$(PYTHON) run_training.py --dataset_path dataset_training/images --epochs 3 --batch_size 4

train-epochs-5:
	@echo "🚀 ENTRENAMIENTO INTENSIVO (5 épocas)"
	$(PYTHON) run_training.py --dataset_path dataset_training/images --epochs 5 --batch_size 4

train-gpu:
	@echo "🔥 ENTRENAMIENTO CON GPU (batch_size 8)"
	$(PYTHON) run_training.py --dataset_path dataset_training/images --epochs 3 --batch_size 8

train-cpu:
	@echo "💻 ENTRENAMIENTO CON CPU (batch_size 2)"
	$(PYTHON) run_training.py --dataset_path dataset_training/images --epochs 2 --batch_size 2

server:
	@echo "🌐 Iniciando servidor API..."
	@echo "   URL:     http://127.0.0.1:9000"
	@echo "   Swagger: http://127.0.0.1:9000/docs"
	@echo ""
	$(PYTHON) -m uvicorn app:app --host 127.0.0.1 --port 9000 --reload

setup:
	@echo "Preparando dataset..."
	python setup_dataset.py
	@echo "✓ Dataset listo"

test:
	@echo "Ejecutando tests..."
	pytest tests/ -v --cov=src --cov-report=html
	@echo "✓ Tests completados. Ver htmlcov/index.html para reporte"

test-quick:
	@echo "Ejecutando tests rápido (sin coverage)..."
	pytest tests/ -v

lint:
	@echo "Validando código con flake8..."
	flake8 src/ app.py config.py --max-line-length=120 --ignore=E501,W503
	@echo "✓ Validación completada"

format:
	@echo "Formateando código con black..."
	black src/ app.py config.py setup_dataset.py --line-length=120
	@echo "✓ Formato completado"

format-check:
	@echo "Verificando formato..."
	black src/ app.py config.py setup_dataset.py --line-length=120 --check
	@echo "✓ Verificación completada"

clean:
	@echo "Limpiando archivos temporales..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type f -name '*.egg-info' -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -exec rm -rf {} +
	find . -type d -name '.mypy_cache' -exec rm -rf {} +
	find . -type d -name 'htmlcov' -exec rm -rf {} +
	find . -type f -name '.coverage' -delete
	rm -rf build/ dist/ *.egg-info
	@echo "✓ Limpieza completada"

docs:
	@echo "Generando documentación..."
	mkdocs build 2>/dev/null || echo "mkdocs no disponible, saltando..."
	@echo "✓ Documentación generada"

run:
	@echo "Iniciando API en puerto 8000..."
	@echo "Documentación disponible en: http://localhost:8000/docs"
	python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000

run-api:
	@echo "Iniciando API en modo producción..."
	python -m uvicorn app:app --host 0.0.0.0 --port 8000

notebook:
	@echo "Iniciando Jupyter..."
	jupyter notebook notebooks/

lab:
	@echo "Iniciando JupyterLab..."
	jupyter lab notebooks/

info:
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  Sistema de Detección de Falsos Pasaportes Mexicanos      ║"
	@echo "║  Proyecto de Maestría - Seminario de Innovación           ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "📋 INFORMACIÓN DEL PROYECTO"
	@echo ""
	@echo "Objetivo: Verificar autenticidad de pasaportes usando IA"
	@echo "Stack:    Python, Transformers (TrOCR), FastAPI, OpenCV"
	@echo "Dataset:  Synthetic Printed Mexican Passports (Kaggle)"
	@echo ""
	@echo "📁 ESTRUCTURA"
	@echo ""
	@echo "  src/                 - Código principal"
	@echo "  app.py               - API REST (FastAPI)"
	@echo "  config.py            - Configuración centralizada"
	@echo "  data/                - Datos (raw, processed, results)"
	@echo "  notebooks/           - Jupyter notebooks"
	@echo "  tests/               - Tests unitarios"
	@echo ""
	@echo "🚀 COMANDOS ÚTILES"
	@echo ""
	@echo "  make setup           - Descargar dataset"
	@echo "  make run             - Iniciar API"
	@echo "  make test            - Ejecutar tests"
	@echo "  make notebook        - Abrir Jupyter"
	@echo ""
	@echo "📖 DOCUMENTACIÓN"
	@echo ""
	@echo "  README.md                        - Inicio rápido"
	@echo "  PLAN_DETECCION_PASAPORTES.md    - Plan detallado del proyecto"
	@echo ""
	@echo ""

# Variables por defecto
.DEFAULT_GOAL := help
