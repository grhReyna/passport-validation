"""
Paquete: src - Sistema de Detección de Pasaportes Falsos

Módulos:
  - preprocessing: Preprocesamiento de imágenes
  - ocr_engine: Motor de OCR con TrOCR
  - mrz_validator: Validador de MRZ
  - confidence_scorer: Cálculo de scores de confianza
  - pipeline: Pipeline principal de procesamiento
"""

__version__ = "0.1.0"
__author__ = "Tu Nombre"

# Importaciones de conveniencia
try:
    from .preprocessing import preprocess_pipeline
    from .ocr_engine import extract_text_with_confidence
    from .mrz_validator import validate_mrz
    from .confidence_scorer import compute_final_score
    from .pipeline import verify_passport
except ImportError:
    # Los módulos no existen aún, eso está bien
    pass

__all__ = [
    "preprocess_pipeline",
    "extract_text_with_confidence",
    "validate_mrz",
    "compute_final_score",
    "verify_passport",
]
