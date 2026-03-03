"""
passport_number_extractor.py - Extrae el número del pasaporte mexicano

Estrategia:
1. El número del pasaporte mexicano está siempre en el lado IZQUIERDO de la segunda página
   o en el lado DERECHO de una página simple
2. Buscamos una región con el patrón: Letra (G, A, etc.) + 8-9 dígitos
3. Extraemos solo esa región y amplificamos para OCR
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def extract_passport_number_region(image: np.ndarray) -> Tuple[Optional[np.ndarray], Tuple]:
    """
    Extrae la región donde va el número del pasaporte.
    
    El número mexicano típicamente está en formato: G77536498 (letra + 8 dígitos)
    Ubicación típica: parte central-derecha del documento
    
    Args:
        image: Imagen preprocesada (512x320 o similar)
        
    Returns:
        Tupla (región_extraída, coords) o (None, ()) si no encuentra
    """
    try:
        h, w = image.shape[:2]
        
        # El número del pasaporte está típicamente en cierta región
        # Pasaportes Mexicanos: en la página 2, lado izquierdo, parte central
        # O en documentos de una página: lado derecho, parte central
        
        # Dividimos en 3 zonas verticales (izq, centro, der)
        # y 3 zonas horizontales (sup, medio, inf)
        
        zone_w = w // 3
        zone_h = h // 3
        
        # Buscar número en el cuadrante central-derecho
        # (donde típicamente está el número)
        regions_to_check = [
            ("CENTRO-DERECHA", (zone_w, zone_h, 2*zone_w, 2*zone_h)),
            ("DERECHA", (2*zone_w, zone_h, w, 2*zone_h)),
            ("DERECHA-INF", (2*zone_w, h//4, w, 3*h//4)),
            ("CENTRO", (zone_w//2, h//4, 2*zone_w, 3*h//4)),
        ]
        
        for region_name, (x1, y1, x2, y2) in regions_to_check:
            region = image[y1:y2, x1:x2]
            
            # Amplificar la región para que el OCR vea mejor los números
            enlarged = cv2.resize(region, (384, 384), interpolation=cv2.INTER_LINEAR)
            
            yield enlarged, (x1, y1, x2, y2), region_name
            
    except Exception as e:
        logger.error(f"Error extrayendo región de número: {e}")
        return None, ()


def extract_text_from_regions(image, ocr_func, confidence_threshold=0.7):
    """
    Intenta extraer número del pasaporte probando diferentes regiones.
    
    Args:
        image: imagen preprocesada
        ocr_func: función de OCR (de ocr_engine)
        confidence_threshold: mínima confianza aceptada
        
    Returns:
        dict con {'numero': str, 'confianza': float, 'region': str}
    """
    best_result = {'numero': None, 'confianza': 0, 'region': 'NONE'}
    
    for enlarged_region, coords, region_name in extract_passport_number_region(image):
        try:
            # OCR en esta región
            ocr_result = ocr_func(enlarged_region)
            text = ocr_result.get('full_text', '').strip()
            confidence = ocr_result.get('ocr_avg_confidence', 0)
            
            logger.debug(f"Región {region_name}: '{text}' (conf={confidence:.1%})")
            
            # Buscar patrón Letra + dígitos
            import re
            pattern = r'[A-Z]?\d{8,9}'
            match = re.search(pattern, text.replace('O', '0').replace('L', '1'))
            
            if match:
                numero = match.group()
                if confidence > best_result['confianza']:
                    best_result = {
                        'numero': numero,
                        'confianza': confidence,
                        'region': region_name
                    }
                    logger.debug(f"  -> Numero encontrado: {numero}")
                    
        except Exception as e:
            logger.debug(f"Error procesando región {region_name}: {e}")
            continue
    
    return best_result
