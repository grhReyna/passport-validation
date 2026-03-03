"""
Detector alternativo para pasaportes mexicanos
Busca: número de pasaporte, en lugar de MRZ ICAO

Los pasaportes mexicanos modernos pueden NO tener MRZ en formato ICAO
Pero SÍ tienen un número de pasaporte que podemos validar
"""

import re
import logging
from typing import Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)

def find_passport_number_region(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detectar región donde está el número de pasaporte mexicano.
    
    Los pasaportes mexicanos tienen el número típicamente:
    - En los datos personales (primeras páginas)
    - Formato: 8-12 caracteres alfanuméricos (ej: G77536498)
    
    Returns:
        Tupla (x, y, w, h) o None si no se detecta
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Buscar en la región superior donde están datos personales (primeros 60% de altura)
        search_height = int(h * 0.6)
        gray_upper = gray[:search_height, :]
        
        # Binarizar
        _, binary = cv2.threshold(gray_upper, 120, 255, cv2.THRESH_BINARY_INV)
        
        # Buscar bloques de texto denso (donde típicamente está el número)
        # El número de pasaporte está cerca de "NÚMERO DE PASAPORTE:" o similar
        vertical_hist = np.sum(binary == 255, axis=1)
        
        if np.max(vertical_hist) == 0:
            return None
        
        # Buscar regiones densas
        vert_norm = (vertical_hist / np.max(vertical_hist)) * 100
        dense_rows = np.where(vert_norm > 5)[0]
        
        if len(dense_rows) < 10:
            return None
        
        # Tomar la región más densa (donde están los datos)
        y_start = dense_rows[0]
        y_end = dense_rows[-1]
        
        # Buscar horizontalmente
        region = binary[y_start:y_end+1, :]
        horizontal_hist = np.sum(region == 255, axis=0)
        
        if np.max(horizontal_hist) == 0:
            return None
        
        horiz_norm = (horizontal_hist / np.max(horizontal_hist)) * 100
        dense_cols = np.where(horiz_norm > 2)[0]
        
        if len(dense_cols) == 0:
            return None
        
        x_start = dense_cols[0]
        x_end = dense_cols[-1]
        
        roi_w = x_end - x_start + 1
        roi_h = y_end - y_start + 1
        
        logger.debug(f"Región de datos detectada: ({x_start}, {y_start}, {roi_w}, {roi_h})")
        
        return (x_start, y_start, roi_w, roi_h)
        
    except Exception as e:
        logger.debug(f"Error en find_passport_number_region: {e}")
        return None


def validate_mexican_passport_number(number: str) -> Tuple[bool, str]:
    """
    Validar formato de número de pasaporte mexicano.
    
    Formato: 
    - Mayúscula (tipo de documento)
    - 8 dígitos (serie)
    - Ejemplo: G77536498 (G = Pasaporte, 77536498 = número)
    
    Args:
        number: Cadena a validar
        
    Returns:
        Tupla (es_válido, razón)
    """
    # Limpiar
    number = number.strip().upper()
    
    # Patrón: letra seguida de números (8-10 dígitos)
    pattern = r'^[A-Z]\d{8,10}$'
    
    if re.match(pattern, number):
        return True, f"Número mexicano válido: {number}"
    else:
        return False, f"Formato inválido: {number} (esperado: Letra + 8-10 dígitos)"


def find_text_regions(image: np.ndarray, grid_size: int = 10) -> list:
    """
    Dividir imagen en grid y encontrar dónde hay texto.
    
    Útil para identificar dónde están los datos principales.
    
    Returns:
        Lista de tuplas (x, y, w, h) de regiones con texto
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Divid ir en grid
    grid_w = w // grid_size
    grid_h = h // grid_size
    
    regions = []
    
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    
    for gy in range(grid_size):
        for gx in range(grid_size):
            y1 = gy * grid_h
            y2 = (gy + 1) * grid_h
            x1 = gx * grid_w
            x2 = (gx + 1) * grid_w
            
            cell = binary[y1:y2, x1:x2]
            density = np.sum(cell == 255) / cell.size * 100
            
            if density > 3:  # Celda tiene texto
                regions.append({
                    'grid_pos': (gx, gy),
                    'pixel_pos': (x1, y1, grid_w, grid_h),
                    'density': density
                })
    
    return regions
