"""
ocr_passport_specialist.py - OCR especializado en números de pasaporte

Estrategia específica para extraer el número del pasaporte mexicano
Enfocarse en la región del número, no en todo el documento
"""

import cv2
import numpy as np
import torch
from PIL import Image
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def extract_passport_number_region(image: np.ndarray) -> Tuple[np.ndarray, Tuple]:
    """
    Extrae SOLO la región donde va el número del pasaporte mexicano.
    
    En pasaportes mexicanos, el número está:
    - LADO DERECHO de la págima
    - PARTE SUPERIOR-MEDIA en la información personal
    - Patrón: LETRA (G, A, B, etc) + 8 dígitos
    
    Args:
        image: Imagen preprocesada
        
    Returns:
        (región_extraída, coords) para OCR especializado
    """
    try:
        h, w = image.shape[:2]
        
        # El número típicamente está en el lado derecho, sector superior-medio
        # Rango X: 45%-95% del ancho
        # Rango Y: 30%-60% del alto
        
        x1 = int(w * 0.45)
        y1 = int(h * 0.25)
        x2 = int(w * 0.95)
        y2 = int(h * 0.55)
        
        region = image[y1:y2, x1:x2]
        
        if region.size == 0:
            logger.warning("Región de número vacía")
            return image, (0, 0, w, h)
        
        logger.debug(f"Región del número extraída: {region.shape}")
        return region, (x1, y1, x2, y2)
        
    except Exception as e:
        logger.error(f"Error extrayendo región: {e}")
        return image, (0, 0, h, w)


def extract_mrz_region(image: np.ndarray) -> Tuple[np.ndarray, Tuple]:
    """
    Extrae región del MRZ (3 líneas al pie del documento).
    
    En pasaportes mexicanos:
    - Ubicada en la parte INFERIOR
    - Ocupada por líneas de < > y caracteres
    
    Args:
        image: Imagen preprocesada
        
    Returns:
        (región_mrz, coords)
    """
    try:
        h, w = image.shape[:2]
        
        # MRZ está tipicamente en los últimos 15-20% de la altura
        y1 = int(h * 0.80)
        y2 = h
        x1 = int(w * 0.05)
        x2 = int(w * 0.95)
        
        region = image[y1:y2, x1:x2]
        
        if region.size == 0:
            logger.warning("Región MRZ vacía")
            return image, (0, 0, w, h)
        
        logger.debug(f"Región MRZ extraída: {region.shape}")
        return region, (x1, y1, x2, y2)
        
    except Exception as e:
        logger.error(f"Error extrayendo MRZ: {e}")
        return image, (0, 0, w, h)


def ocr_passport_region(image_region: np.ndarray, 
                        ocr_model, processor, tokenizer,
                        region_type: str = "number") -> Dict:
    """
    OCR especializado para región de número o MRZ.
    
    Args:
        image_region: Región extraída (640x?, 480x?, etc)
        ocr_model: Modelo ViT-TrOCR cargado
        processor: Processor de ViT
        tokenizer: Tokenizer
        region_type: "number" o "mrz"
        
    Returns:
        {'text': str, 'confidence': float}
    """
    try:
        # Redimensionar región a tamaño óptimo para OCR
        if region_type == "number":
            # Región de número: mantener aspecto, max 256px alto
            h, w = image_region.shape[:2]
            target_h = 256
            target_w = int(w * target_h / h)
            target_w = min(target_w, 512)  # Max 512px ancho
        else:  # mrz
            # Región MRZ: 1 línea alta, ancho máximo
            h, w = image_region.shape[:2]
            target_h = 128
            target_w = int(w * target_h / h)
            target_w = min(target_w, 640)  # Max 640px ancho
        
        resized = cv2.resize(image_region, (target_w, target_h), 
                             interpolation=cv2.INTER_LANCZOS4)
        
        # Mejorar contraste para OCR
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Convertir a RGB para modelo
        rgb_region = cv2.cvtColor(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 
                                   cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_region)
        
        # OCR
        pixel_values = processor(pil_img, return_tensors="pt").pixel_values
        
        with torch.no_grad():
            generated_ids = ocr_model.generate(pixel_values, max_length=128)
        
        text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        text = text.strip()
        
        logger.debug(f"OCR {region_type}: '{text}'")
        
        return {
            'text': text,
            'confidence': 0.95,  # Placeholder
            'region_size': (target_w, target_h)
        }
        
    except Exception as e:
        logger.error(f"Error en OCR {region_type}: {e}")
        return {'text': '', 'confidence': 0.0, 'error': str(e)}


def extract_passport_data(image: np.ndarray, 
                          ocr_model=None, processor=None, tokenizer=None) -> Dict:
    """
    Extrae TODOS los datos del pasaporte usando OCR especializado.
    
    Procesa:
    1. Región de número (cuadro verde con número)
    2. Región de nombre
    3. Región de MRZ (si existe)
    
    Args:
        image: Imagen preprocesada
        ocr_model, processor, tokenizer: Modelos de OCR
        
    Returns:
        {
            'numero': str,
            'nombre': str, 
            'mrz': str,
            'all_text': str
        }
    """
    try:
        results = {}
        
        # 1. OCR NÚMERO
        number_region, number_coords = extract_passport_number_region(image)
        
        if ocr_model is not None:
            number_result = ocr_passport_region(number_region, ocr_model, 
                                               processor, tokenizer, "number")
            results['numero'] = number_result.get('text', '')
            logger.debug(f"Número extraído: {results['numero']}")
        
        # 2. OCR MRZ (si está disponible)
        mrz_region, mrz_coords = extract_mrz_region(image)
        
        if ocr_model is not None:
            mrz_result = ocr_passport_region(mrz_region, ocr_model, 
                                            processor, tokenizer, "mrz")
            results['mrz'] = mrz_result.get('text', '')
            logger.debug(f"MRZ extraído: {results['mrz']}")
        
        # 3. OCR GENERAL (todo el documento, para nombre y otros datos)
        # Usar imagen completa per con mejor preprocesamiento
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        rgb_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        pil_img = Image.fromarray(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        
        if ocr_model is not None:
            pixel_values = processor(pil_img, return_tensors="pt").pixel_values
            
            with torch.no_grad():
                generated_ids = ocr_model.generate(pixel_values, max_length=256)
            
            all_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            results['all_text'] = all_text
            logger.debug(f"Texto completo: {len(all_text)} chars")
        
        return results
        
    except Exception as e:
        logger.error(f"Error extrayendo datos: {e}")
        return {'error': str(e)}
