"""
Configuración centralizada del proyecto
Detección de Falsos Pasaportes Mexicanos con Score de Confianza
"""

import os
from pathlib import Path

# ============================================================================
# PATHS Y DIRECTORIOS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / "data"
DATA_RAW_PATH = DATA_PATH / "raw"
DATA_PROCESSED_PATH = DATA_PATH / "processed"
DATA_RESULTS_PATH = DATA_PATH / "results"
MODELS_PATH = PROJECT_ROOT / "models"
NOTEBOOKS_PATH = PROJECT_ROOT / "notebooks"
TESTS_PATH = PROJECT_ROOT / "tests"

# Crear directorios si no existen
for path in [DATA_RAW_PATH, DATA_PROCESSED_PATH, DATA_RESULTS_PATH, MODELS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PREPROCESAMIENTO DE IMÁGENES
# ============================================================================

# Dimensiones objetivo para imágenes de pasaporte
# IMPORTANTE: ViT redimensiona NUEVAMENTE a 384x384, así que mantener
# una relación cercana a eso para no destruir información de texto
# Pasaportes: enfoque en región de número (izq-centro, parte superior)
IMAGE_SIZE = (640, 480)  # 4:3 ratio, optimizado para números de pasaporte
IMAGE_HEIGHT, IMAGE_WIDTH = IMAGE_SIZE[1], IMAGE_SIZE[0]

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)

# Noise removal
DENOISE_H = 10
DENOISE_TEMPLATE_WINDOW = 7
DENOISE_SEARCH_WINDOW = 21

# ============================================================================
# OCR - TROCR
# ============================================================================

# Modelo TrOCR preentrenado para documentos impresos
TROCR_MODEL_NAME = "microsoft/trocr-base-printed"

# Umbral de confianza para tokens OCR
OCR_CONFIDENCE_THRESHOLD = 0.65

# Campos críticos para OCR (mayor peso en scoring)
CRITICAL_FIELDS = ["nombre", "apellido", "numero_pasaporte", "nacionalidad"]

# ============================================================================
# MRZ (Machine Readable Zone) - PASAPORTE MEXICANO
# ============================================================================

# Estructura MRZ para pasaportes
MRZ_LINE_LENGTH = 88  # Típicamente 88 caracteres para 3 líneas

# Código de país pasaporte mexicano
PASSPORT_COUNTRY_CODE = "MEX"

# Longitud del número de pasaporte mexicano
PASSPORT_NUMBER_LENGTH = 9

# Fechas en formato YYMMDD
# Rango válido: pasaportes pueden estar 10 años vigentes
DATE_MIN_YEAR = 80  # 1980
DATE_MAX_YEAR = 50  # 2050

# Pesos para validación MRZ
MRZ_CHECKSUM_WEIGHT = 0.35
MRZ_COHERENCE_WEIGHT = 0.25

# ============================================================================
# SCORING Y DECISIÓN
# ============================================================================

# Pesos del score final de confianza
OCR_WEIGHT = 0.40      # Extracción de texto OCR
MRZ_WEIGHT = 0.60      # Validación de MRZ

# Umbrales de decisión
PASS_THRESHOLD = 0.90      # Score >= 90% → PASS (aceptar)
REVIEW_THRESHOLD = 0.70    # 70% <= Score < 90% → REVIEW (manual)
# Score < 70% → REJECT (rechazar)

# Estados del documento
STATUS_PASS = "PASS"
STATUS_REVIEW = "REVIEW"
STATUS_REJECT = "REJECT"

# ============================================================================
# API REST
# ============================================================================

API_HOST = "0.0.0.0"
API_PORT = 8000
API_DEBUG = True
API_RELOAD = True

# Límite de tamaño de archivo (en bytes) - 10 MB
MAX_FILE_SIZE = 10 * 1024 * 1024

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================================================
# PROCESSING
# ============================================================================

# Timeout para procesamiento de imagen (segundos)
PROCESSING_TIMEOUT = 30

# Número de workers para procesamiento paralelo
NUM_WORKERS = 2

# ============================================================================
# ESPECIFICACIONES TÉCNICAS - PASAPORTE MEXICANO
# ============================================================================

# Línea 1: Tipo de documento y código de país
# Formato: P<[CODE_PAÍS]
MRZ_LINE1_START = 0
MRZ_LINE1_LENGTH = 24

# Línea 2: Número de pasaporte, fecha nacimiento, etc
# Formato variable

# Línea 3: Información personal

MEXICAN_PASSPORT_FIELDS = {
    "documento_type": "P",  # P para Passport
    "country_code": "MEX",
    "nationality": "MEX",
    "mrz_length": 88,
}

# ============================================================================
# CONFIGURACIÓN DE EXPERIMENTACIÓN
# ============================================================================

# Porcentaje de split para train/val/test
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Seed para reproducibilidad
RANDOM_SEED = 42

# ============================================================================
# VALIDACIÓN Y CALIDAD
# ============================================================================

# Métricas target
TARGET_CER = 0.05  # Character Error Rate < 5%
TARGET_MRZ_PRECISION = 0.95  # > 95% MRZ correctos
TARGET_FP_RATE = 0.05  # < 5% falsos positivos
TARGET_FN_RATE = 0.10  # < 10% falsos negativos
TARGET_PROCESSING_TIME = 2.0  # < 2 segundos por imagen
