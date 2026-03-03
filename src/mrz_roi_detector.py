#!/usr/bin/env python
"""
mrz_roi_detector.py - Detector optimizado de zona de lectura mecánica (MRZ)

Detecta específicamente la zona de MRZ en pasaportes usando:
- Búsqueda de líneas horizontales oscuras
- Patrones de caracteres OCR
- Histogramas de densidad vertical
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def _find_mrz_region_upright(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detectar la Zona de Lectura Mecánica (MRZ) en pasaporte.
    
    Características del MRZ:
    - Está típicamente en los últimos 150px pero puede variar
    - Tiene altura de 30-60px (2 líneas de texto)
    - Densidad ALTA de caracteres oscuros
    - Está al fondo del documento
    
    ESTRATEGIA: 
    1. Buscar CUALQUIER región densa de caracteres
    2. Preferentemente hacia el fondo
    3. Ser muy tolerante con los parámetros
    
    Args:
        image: Imagen en BGR
        
    Returns:
        Tupla (x, y, w, h) de región MRZ, o None si no se detecta
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        logger.debug(f"find_mrz_region: Imagen {width}x{height}")
        
        # 1. Binarizar con threshold adaptativo (más robusto)
        _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
        
        # 2. Análisis vertical: densidad por fila
        vertical_hist = np.sum(binary == 255, axis=1)
        
        if np.max(vertical_hist) == 0:
            logger.debug("MRZ: No hay píxeles oscuros detectados")
            return None
        
        vertical_hist_norm = (vertical_hist / np.max(vertical_hist)) * 100
        
        # 3. ESTRATEGIA: Buscar en TODO el documento pero PRIORIZANDO zona inferior
        # Dividir en 3 regiones: superior, media, inferior
        third_height = height // 3
        
        # Priorizar: inferior (peso 3x), media (peso 2x), superior (peso 1x)
        # El MRZ casi siempre está en el tercio inferior
        
        # Buscar en región inferior PRIMERO (últimos 180px)
        search_lower_bound = max(0, height - 180)
        search_upper_bound = height
        
        region_hist = vertical_hist_norm[search_lower_bound:search_upper_bound]
        
        # Buscar filas con densidad MODERADA (no muy estricta)
        min_density = 8  # Reducido de 15 para ser más tolerante
        dense_indices = np.where(region_hist > min_density)[0]
        
        logger.debug(f"MRZ: Filas densas encontradas en región inferior: {len(dense_indices)}")
        
        if len(dense_indices) < 20:  # Mínimo 20px (muy tolerante)
            logger.debug(f"MRZ: Buscando en TODA la imagen...")
            
            # Fallback: buscar en toda la imagen
            dense_indices = np.where(vertical_hist_norm > min_density)[0]
            
            if len(dense_indices) < 20:
                logger.debug(f"MRZ: No se encontraron regiones densas")
                return None
            
            dense_rows = dense_indices
        else:
            # Convertir índices a imagen original
            dense_rows = dense_indices + search_lower_bound
        
        # 4. Encontrar bloques contiguos densos
        # Separar bloques por gaps
        blocks = []
        current_block = [dense_rows[0]]
        
        for i in range(1, len(dense_rows)):
            gap = dense_rows[i] - dense_rows[i-1]
            
            if gap <= 3:  # Tolerancia: gaps pequeños no rompen el bloque
                current_block.append(dense_rows[i])
            else:
                # Fin de bloque
                if len(current_block) >= 20:  # Bloque válido
                    blocks.append(current_block)
                current_block = [dense_rows[i]]
        
        # Agregar último bloque
        if len(current_block) >= 20:
            blocks.append(current_block)
        
        if not blocks:
            logger.debug(f"MRZ: No se encontraron bloques válidos")
            return None
        
        # 5. Seleccionar el BLOQUE MÁS BAJO (más cerca del final)
        # El MRZ está casi siempre al final del documento
        mrz_block = blocks[-1]  # Último bloque = más abajo
        
        mrz_top = mrz_block[0]
        mrz_bottom = mrz_block[-1]
        mrz_height = mrz_bottom - mrz_top + 1
        
        logger.debug(f"MRZ: Bloque seleccionado - top={mrz_top}, bottom={mrz_bottom}, height={mrz_height}")
        
        # Validar altura de forma más flexible
        if mrz_height < 18 or mrz_height > 120:
            logger.debug(f"MRZ: Altura fuera de rango ({mrz_height}px, esperado 18-120)")
            
            # Si el único bloque no cumple, intentar con cualquier bloque
            for block in reversed(blocks):
                h = block[-1] - block[0] + 1
                if 18 <= h <= 120:
                    mrz_block = block
                    mrz_top = block[0]
                    mrz_bottom = block[-1]
                    mrz_height = h
                    logger.debug(f"MRZ: Usando bloque alternativo - height={mrz_height}")
                    break
            else:
                logger.debug(f"MRZ: Ningún bloque con altura válida")
                return None
        
        # 6. Encontrar límites horizontales
        # Buscar en registro horizontal dentro de la región MRZ
        mrz_region = binary[mrz_top:mrz_bottom+1, :]
        horizontal_hist = np.sum(mrz_region == 255, axis=0)
        
        if np.max(horizontal_hist) == 0:
            logger.debug(f"MRZ: Columnas sin píxeles")
            return None
        
        # Encontrar columnas con densidad mínima
        min_col_density = np.max(horizontal_hist) * 0.05  # 5% de la densidad máxima
        valid_cols = np.where(horizontal_hist > min_col_density)[0]
        
        if len(valid_cols) < 20:
            logger.debug(f"MRZ: Muy pocas columnas válidas")
            return None
        
        mrz_left = valid_cols[0]
        mrz_right = valid_cols[-1]
        mrz_width = mrz_right - mrz_left + 1
        
        logger.debug(f"MRZ: Detectado exitosamente - x={mrz_left}, y={mrz_top}, w={mrz_width}, h={mrz_height}")
        
        return (mrz_left, mrz_top, mrz_width, mrz_height)
        
    except Exception as e:
        logger.debug(f"Error en find_mrz_region: {e}")
        return None


def _map_rotated_roi_to_original(
    roi: Tuple[int, int, int, int],
    rotation_flag: Optional[int],
    original_width: int,
    original_height: int,
) -> Tuple[int, int, int, int]:
    """Mapear ROI detectado en imagen rotada a coordenadas de imagen original."""
    x, y, w, h = roi
    corners = [
        (x, y),
        (x + w - 1, y),
        (x, y + h - 1),
        (x + w - 1, y + h - 1),
    ]

    mapped = []
    for x_r, y_r in corners:
        if rotation_flag is None:
            x_o, y_o = x_r, y_r
        elif rotation_flag == cv2.ROTATE_90_CLOCKWISE:
            x_o = y_r
            y_o = original_height - 1 - x_r
        elif rotation_flag == cv2.ROTATE_90_COUNTERCLOCKWISE:
            x_o = original_width - 1 - y_r
            y_o = x_r
        elif rotation_flag == cv2.ROTATE_180:
            x_o = original_width - 1 - x_r
            y_o = original_height - 1 - y_r
        else:
            x_o, y_o = x_r, y_r

        mapped.append((x_o, y_o))

    xs = [p[0] for p in mapped]
    ys = [p[1] for p in mapped]

    x_min = max(0, min(xs))
    y_min = max(0, min(ys))
    x_max = min(original_width - 1, max(xs))
    y_max = min(original_height - 1, max(ys))

    return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)


def find_mrz_region(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detectar MRZ intentando múltiples orientaciones (0°, 90°, 270°, 180°).
    """
    try:
        if image is None or image.size == 0:
            return None

        original_height, original_width = image.shape[:2]

        orientation_candidates = [
            (None, image, "0°"),
            (cv2.ROTATE_90_CLOCKWISE, cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), "90° CW"),
            (cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), "90° CCW"),
            (cv2.ROTATE_180, cv2.rotate(image, cv2.ROTATE_180), "180°"),
        ]

        for rotation_flag, oriented_img, label in orientation_candidates:
            roi_oriented = _find_mrz_region_upright(oriented_img)
            if roi_oriented is None:
                continue

            roi_original = _map_rotated_roi_to_original(
                roi_oriented,
                rotation_flag,
                original_width,
                original_height,
            )
            logger.debug(f"MRZ detectado en orientación {label}: {roi_original}")
            return roi_original

        logger.debug("MRZ no detectado en ninguna orientación")
        return None

    except Exception as e:
        logger.warning(f"Error detectando MRZ multi-orientación: {e}")
        return None


def detect_document_roi(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detectar región principal del documento (sin fondos).
    
    Estrategia:
    1. Convertir a binaria buscando contenido
    2. Encontrar el bounding box del documento completo
    3. Validar tamaño mínimo
    
    Args:
        image: Imagen en BGR
        
    Returns:
        Tupla (x, y, w, h) del documento
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Estrategia 1: Buscar contenido oscuro (texto del documento)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Contorno más grande (probablemente el documento)
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            
            # Validar tamaño (documento debe ocupar >15% de imagen)
            area_percent = (w * h) / (width * height) * 100
            if area_percent > 15:
                logger.debug(f"Documento detectado: {w}x{h} ({area_percent:.1f}%)")
                return (x, y, w, h)
        
        # Estrategia 2: Si contornos fallaron, buscar por densidad
        logger.debug("Documento: fallback a detección por densidad")
        
        # Buscar region densa verticalemente
        vertical_hist = np.sum(binary == 255, axis=1)
        
        # Encontrar top y bottom del contenido
        nonzero_rows = np.where(vertical_hist > 0)[0]
        
        if len(nonzero_rows) > 100:  # Mínimo altura
            y_start = nonzero_rows[0]
            y_end = nonzero_rows[-1]
            
            # Buscar horizontalmente
            horizontal_hist = np.sum(binary == 255, axis=0)
            nonzero_cols = np.where(horizontal_hist > 0)[0]
            
            if len(nonzero_cols) > 100:  # Mínimo ancho
                x_start = nonzero_cols[0]
                x_end = nonzero_cols[-1]
                
                x = max(0, x_start - 5)
                y = max(0, y_start - 5)
                w = min(width - x, x_end - x_start + 10)
                h = min(height - y, y_end - y_start + 10)
                
                area_percent = (w * h) / (width * height) * 100
                if area_percent > 10:
                    logger.info(f"✓ Documento detectado (fallback): {w}x{h}")
                    return (x, y, w, h)
        
        logger.warning("Documento no detectado, usando imagen completa")
        return None
        
    except Exception as e:
        logger.warning(f"Error detectando documento: {e}")
        return None


def extract_region(image: np.ndarray, 
                  roi: Tuple[int, int, int, int],
                  expand_percent: float = 10) -> np.ndarray:
    """
    Extraer región con margen opcional.
    
    Args:
        image: Imagen
        roi: Tupla (x, y, w, h)
        expand_percent: % para expandir región (para contexto)
        
    Returns:
        Región recortada
    """
    x, y, w, h = roi
    height, width = image.shape[:2]
    
    # Expandir con margen
    margin_x = int(w * expand_percent / 100)
    margin_y = int(h * expand_percent / 100)
    
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(width, x + w + margin_x)
    y2 = min(height, y + h + margin_y)
    
    return image[y1:y2, x1:x2]
