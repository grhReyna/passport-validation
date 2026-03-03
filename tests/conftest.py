"""
conftest.py - Configuración de pytest

Define fixtures compartidas para todos los tests
"""

import pytest
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import config


@pytest.fixture
def sample_image_path():
    """Fixture para path de imagen de prueba"""
    # En desarrollo, usar una imagen del dataset si está disponible
    sample_dir = config.DATA_RAW_PATH / "test"
    if sample_dir.exists():
        images = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
        if images:
            return images[0]
    return None


@pytest.fixture
def test_config():
    """Fixture para acceder a valores de configuración"""
    return {
        "image_size": config.IMAGE_SIZE,
        "ocr_threshold": config.OCR_CONFIDENCE_THRESHOLD,
        "pass_threshold": config.PASS_THRESHOLD,
        "review_threshold": config.REVIEW_THRESHOLD,
    }


@pytest.fixture
def mock_ocr_result():
    """Fixture con resultado OCR de ejemplo"""
    return {
        "full_text": "APELLIDO NOMBRE\nP<MEXPASAPORT1234567890<<<<<<<>>>",
        "tokens": [
            {"text": "APELLIDO", "confidence": 0.98},
            {"text": "NOMBRE", "confidence": 0.95},
        ],
        "mrz_detected": True,
        "mrz_lines": [
            "P<MEXPASAPORTE<<<<<<APELLIDO<<NOMBRE",
            "A12345678MEX2000010112345678<<<<<<<<<<<<<<<",
            "9<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        ],
        "ocr_avg_confidence": 0.92
    }


@pytest.fixture
def mock_mrz_validation():
    """Fixture con resultado de validación MRZ de ejemplo"""
    return {
        "mrz_valid": True,
        "checksum_errors": [],
        "coherence_errors": [],
        "mrz_confidence_score": 1.0,
        "details": {
            "country_code": "MEX",
            "passport_number": "A12345678",
            "nationality": "MEX"
        }
    }


# ============================================================================
# HOOKS Y CONFIGURACIÓN GLOBAL
# ============================================================================

def pytest_configure(config):
    """Hook que se ejecuta antes de empezar los tests"""
    # Crear marcadores personalizados
    config.addinivalue_line(
        "markers", "slow: marca tests como lentos"
    )
    config.addinivalue_line(
        "markers", "integration: marca tests de integración"
    )
    config.addinivalue_line(
        "markers", "requires_dataset: requiere dataset descargado"
    )


def pytest_collection_modifyitems(config, items):
    """Hook para modificar items de prueba"""
    for item in items:
        # Agregar marcado automático
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        if "test_with_dataset" in item.nodeid or "requires_dataset" in item.nodeid:
            item.add_marker(pytest.mark.requires_dataset)
