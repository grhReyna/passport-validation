"""
ocr_engine.py - Motor de OCR con TrOCR

Extrae texto de imágenes de pasaportes usando el modelo
TrOCR (Transformer-based OCR) de Microsoft/Hugging Face.

Características:
- Extracción de texto con confianza por token
- Detección automática de MRZ
- Caché de modelo para eficiencia
- Manejo de múltiples idiomas

Versión: 0.1.0
Referencia: Li et al. (2023) - TrOCR: Transformer-based OCR
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple, List, Dict, Optional
import logging
import re
from PIL import Image
import io

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
except ImportError:
    raise ImportError("Requiere: transformers, torch, pillow")

import config

logger = logging.getLogger(__name__)


# ============================================================================
# CACHÉ GLOBAL DE MODELO
# ============================================================================

_model_cache = {
    "model": None,
    "processor": None,
    "device": None,
}

_easyocr_cache = {
    "reader": None,
}


def _get_easyocr_reader():
    """Cargar EasyOCR bajo demanda para fallback de MRZ."""
    if _easyocr_cache["reader"] is not None:
        return _easyocr_cache["reader"]

    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)
        _easyocr_cache["reader"] = reader
        logger.info("✓ EasyOCR cargado para fallback MRZ")
        return reader
    except Exception as e:
        logger.debug(f"EasyOCR no disponible para fallback MRZ: {e}")
        return None


def _extract_mrz_lines_with_easyocr(region_bgr: np.ndarray) -> List[str]:
    """Extraer posibles líneas MRZ con EasyOCR (fallback)."""
    reader = _get_easyocr_reader()
    if reader is None or region_bgr is None or region_bgr.size == 0:
        return []

    try:
        gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
        up = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        _, th = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        easy_results = reader.readtext(th, detail=0, paragraph=False)
        candidates = []

        for text in easy_results:
            normalized = str(text).upper().strip()
            normalized = ''.join(c for c in normalized if c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789< ')
            normalized = normalized.replace(' ', '<')
            if len(normalized) >= 10:
                candidates.append(normalized)

        return candidates

    except Exception as e:
        logger.debug(f"Fallback EasyOCR MRZ falló: {e}")
        return []


def _mrz_line_likelihood(line: str) -> float:
    """Estimar qué tan probable es que una línea sea MRZ real."""
    if not line:
        return 0.0

    text = line.strip().upper()
    if not text:
        return 0.0

    valid_chars = sum(1 for c in text if c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<')
    digit_count = sum(1 for c in text if c.isdigit())
    filler_count = text.count('<')

    char_ratio = valid_chars / max(1, len(text))
    digit_ratio = digit_count / max(1, len(text))
    filler_ratio = filler_count / max(1, len(text))

    score = (0.45 * char_ratio) + (0.35 * digit_ratio) + (0.20 * filler_ratio)
    return float(max(0.0, min(1.0, score)))


def extract_passport_number_fallback(image_bgr: np.ndarray) -> Dict:
    """
    Extraer número de pasaporte mexicano con EasyOCR en múltiples orientaciones.

    Busca patrón típico: Letra + 8/9 dígitos (ej. G12345678).
    """
    result = {
        "passport_number": None,
        "confidence": 0.0,
        "raw_text": "",
        "engine": "none",
    }

    reader = _get_easyocr_reader()
    if reader is None or image_bgr is None or image_bgr.size == 0:
        return result

    try:
        orientations = [
            image_bgr,
            cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE),
        ]

        strict_pattern = re.compile(r'[A-Z]\d{8,9}')
        loose_pattern = re.compile(r'[A-Z]\d{7,10}')

        best_candidate = None
        best_conf = 0.0
        best_quality = 0.0
        collected_text = []

        for oriented in orientations:
            gray = cv2.cvtColor(oriented, cv2.COLOR_BGR2GRAY)
            up = cv2.resize(gray, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
            detections = reader.readtext(up, detail=1, paragraph=False)

            for det in detections:
                if len(det) < 3:
                    continue

                text = str(det[1]).upper().strip()
                conf = float(det[2])
                normalized = re.sub(r'[^A-Z0-9]', '', text)
                normalized = normalized.replace('O', '0')

                if normalized:
                    collected_text.append(normalized)

                digit_count = sum(1 for c in normalized if c.isdigit())
                quality = 0.0
                if normalized:
                    quality = (digit_count / max(1, len(normalized)))

                strict_match = strict_pattern.search(normalized)
                if strict_match and (conf + quality) > (best_conf + best_quality):
                    best_candidate = strict_match.group(0)
                    best_conf = max(conf, 0.65)
                    best_quality = quality
                    continue

                if best_candidate is None:
                    loose_match = loose_pattern.search(normalized)
                    if loose_match and (conf + quality) > (best_conf + best_quality):
                        best_candidate = loose_match.group(0)
                        best_conf = max(conf, 0.45)
                        best_quality = quality

        if best_candidate:
            result["passport_number"] = best_candidate
            result["confidence"] = best_conf
            result["engine"] = "easyocr"

        result["raw_text"] = " ".join(collected_text[:30])
        return result

    except Exception as e:
        logger.debug(f"Fallback número pasaporte falló: {e}")
        result["raw_text"] = str(e)
        return result


def _estimate_text_confidence(text: str) -> float:
    """Estimar confianza OCR basada en calidad real del texto extraído."""
    if not text:
        return 0.05

    cleaned = text.strip()
    if not cleaned:
        return 0.05

    text_len = len(cleaned)
    alnum_count = sum(1 for c in cleaned if c.isalnum())
    alpha_ratio = alnum_count / max(1, text_len)

    length_factor = min(1.0, text_len / 40.0)
    confidence = (0.20 + (0.55 * alpha_ratio) + (0.25 * length_factor))

    if text_len <= 2 and alnum_count <= 1:
        confidence = min(confidence, 0.20)

    return float(max(0.05, min(0.97, confidence)))


def _text_quality_score(text: str) -> float:
    """Score auxiliar para elegir mejor resultado OCR entre intentos."""
    if not text:
        return 0.0

    normalized = text.strip()
    if not normalized:
        return 0.0

    alnum = sum(1 for c in normalized if c.isalnum())
    unique_chars = len(set(normalized))
    score = (alnum * 1.5) + min(20, unique_chars)
    return float(score)


def _prepare_ocr_variants(image_bgr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """Generar variantes de preprocesamiento para OCR robusto."""
    variants = [("orig", image_bgr)]

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    variants.append(("gray", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    variants.append(("clahe", cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2BGR)))

    _, binary = cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("binary", cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)))

    upscaled = cv2.resize(image_bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    variants.append(("up2x", upscaled))

    return variants


def _get_device() -> str:
    """
    Determinar dispositivo para inferencia (CUDA si disponible, senó CPU).
    
    Returns:
        str: "cuda" o "cpu"
    """
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"✓ GPU disponible: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.warning("GPU no disponible, usando CPU (inferencia lenta)")
    
    return device


def load_trocr_model(model_name: str = config.TROCR_MODEL_NAME,
                     force_reload: bool = False) -> Tuple[VisionEncoderDecoderModel, TrOCRProcessor, str]:
    """
    Cargar modelo TrOCR y processor (con caché).
    
    Args:
        model_name: Nombre del modelo en Hugging Face
        force_reload: Si forzar recarga (ignorar caché)
        
    Returns:
        Tupla (modelo, processor, dispositivo)
        
    Raises:
        RuntimeError: Si no se puede cargar el modelo
    """
    try:
        # Verificar caché
        if not force_reload and _model_cache["model"] is not None:
            logger.debug("Usando modelo cacheado")
            return (_model_cache["model"], 
                   _model_cache["processor"], 
                   _model_cache["device"])
        
        # Determinar dispositivo
        device = _get_device()
        
        # Intentar cargar modelo local finetuned primero
        loaded = False
        from pathlib import Path as _Path
        if _Path(model_name).exists():
            try:
                logger.info(f"Cargando modelo finetuned local: {model_name}")
                processor = TrOCRProcessor.from_pretrained(model_name)
                model = VisionEncoderDecoderModel.from_pretrained(model_name)
                loaded = True
                logger.info("✓ Modelo finetuned local cargado exitosamente")
            except Exception as e:
                logger.warning(f"No se pudo cargar modelo local: {e}")
        
        if not loaded:
            # Fallback al modelo base de HuggingFace
            fallback = getattr(config, 'TROCR_BASE_MODEL', model_name)
            logger.info(f"Cargando modelo TrOCR: {fallback}")
            processor = TrOCRProcessor.from_pretrained(fallback)
            model = VisionEncoderDecoderModel.from_pretrained(fallback)
            logger.info("✓ Modelo TrOCR base cargado exitosamente")
        
        model.to(device)
        model.eval()  # Modo evaluación
        
        # Guardar en caché
        _model_cache["model"] = model
        _model_cache["processor"] = processor
        _model_cache["device"] = device
        
        return model, processor, device
    
    except Exception as e:
        logger.error(f"Error cargando modelo TrOCR: {str(e)}")
        raise RuntimeError(f"No se pudo cargar modelo: {str(e)}")


def image_to_pil(image_input: Union[np.ndarray, str, Path, bytes]) -> Image.Image:
    """
    Convertir imagen a formato PIL para TrOCR.
    
    Args:
        image_input: Imagen en varios formatos
        
    Returns:
        Image.Image: Imagen PIL en modo RGB
    """
    try:
        if isinstance(image_input, Image.Image):
            if image_input.mode != 'RGB':
                return image_input.convert('RGB')
            return image_input
        
        elif isinstance(image_input, np.ndarray):
            # Convertir de BGR (OpenCV) a RGB
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                rgb_image = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb_image)
            else:
                return Image.fromarray(image_input).convert('RGB')
        
        elif isinstance(image_input, (str, Path)):
            img = Image.open(image_input)
            return img.convert('RGB')
        
        elif isinstance(image_input, bytes):
            img = Image.open(io.BytesIO(image_input))
            return img.convert('RGB')
        
        else:
            raise ValueError(f"Tipo no soportado: {type(image_input)}")
    
    except Exception as e:
        logger.error(f"Error convirtiendo imagen a PIL: {str(e)}")
        raise


def extract_text_from_image(image_input: Union[np.ndarray, str, Path, bytes],
                           model=None,
                           processor=None,
                           device: str = None) -> str:
    """
    Extraer texto de imagen usando TrOCR.
    
    Función simple que retorna solo el texto (sin confianzas).
    
    Args:
        image_input: Imagen en múltiples formatos
        model: Modelo TrOCR (se carga si no se proporciona)
        processor: Processor de TrOCR
        device: Dispositivo de ejecución
        
    Returns:
        str: Texto extraído
    """
    try:
        # Cargar modelo si no se proporciona
        if model is None or processor is None:
            model, processor, device = load_trocr_model()
        
        logger.debug("Extrayendo texto con TrOCR")
        
        # Convertir a PIL
        pil_image = image_to_pil(image_input)
        
        # Procesar imagen
        pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        
        # Generar texto
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_length=128)
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        logger.debug(f"Texto extraído ({len(generated_text)} caracteres)")
        
        return generated_text
    
    except Exception as e:
        logger.error(f"Error extrayendo texto: {str(e)}")
        raise


def extract_text_with_confidence(image_input: Union[np.ndarray, str, Path, bytes],
                                model=None,
                                processor=None,
                                device: str = None,
                                return_all_beams: bool = False,
                                num_beams: int = 2) -> Dict:
    """
    Extraer texto con scores de confianza por token.
    
    Genera múltiples hipótesis (beam search) para estimar confianza.
    
    Args:
        image_input: Imagen en múltiples formatos
        model: Modelo TrOCR (se carga si no se proporciona)
        processor: Processor de TrOCR
        device: Dispositivo de ejecución
        return_all_beams: Si retornar todas las hipótesis beam search
        
    Returns:
        dict: Resultado con texto y confianzas
        
        Estructura:
        {
            "full_text": "TEXTO COMPLETO",
            "tokens": [
                {"text": "PALABRA", "confidence": 0.95},
                ...
            ],
            "mrz_detected": bool,
            "mrz_lines": ["línea1", "línea2", "línea3"],
            "ocr_avg_confidence": 0.92,
            "all_hypotheses": [...]  # Si return_all_beams=True
        }
    """
    result = {
        "full_text": "",
        "tokens": [],
        "mrz_detected": False,
        "mrz_lines": [],
        "ocr_avg_confidence": 0.0,
    }
    
    try:
        # Cargar modelo si no se proporciona
        if model is None or processor is None:
            model, processor, device = load_trocr_model()
        
        logger.info("Extrayendo texto con confianzas (beam search)")
        
        # Convertir a PIL
        pil_image = image_to_pil(image_input)
        
        # Procesar imagen
        pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        
        # Generar con beam search (para obtener múltiples hipótesis)
        num_beams = max(1, int(num_beams))
        
        with torch.no_grad():
            sequences = model.generate(
                pixel_values,
                max_length=128,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                output_scores=True,
                return_dict_in_generate=True,
            )
        
        # Decodificar hipótesis
        hypotheses = processor.batch_decode(
            sequences.sequences,
            skip_special_tokens=True
        )
        
        # Usar mejor hipótesis (primera)
        best_text = hypotheses[0]
        
        result["full_text"] = best_text
        
        # Tokenizar y asignar confianzas
        # Confianza basada en calidad del texto real, no en posición del beam
        words = best_text.split()
        base_confidence = _estimate_text_confidence(best_text)
        
        for word in words:
            result["tokens"].append({
                "text": word,
                "confidence": base_confidence
            })
        
        # Calcular confianza promedio
        result["ocr_avg_confidence"] = base_confidence
        
        # Detectar MRZ (buscar líneas que empiezan con P< o tienen patrones MRZ)
        lines = best_text.split('\n')
        mrz_candidates = []
        
        for line in lines:
            line_upper = line.strip().upper()
            # Buscar patrones MRZ
            if (line_upper.startswith('P<') or 
                (len(line_upper) > 20 and 
                 all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<' for c in line_upper[:20]))):
                mrz_candidates.append(line_upper)
        
        if len(mrz_candidates) >= 3:
            result["mrz_detected"] = True
            result["mrz_lines"] = mrz_candidates[:3]
            logger.debug(f"MRZ detectada: {len(mrz_candidates)} líneas")
        elif mrz_candidates:
            result["mrz_detected"] = True
            result["mrz_lines"] = mrz_candidates
            logger.debug(f"MRZ parcialmente detectada: {len(mrz_candidates)} línea(s)")
        else:
            logger.debug("MRZ no detectada en OCR")
        
        # Opcionalmente, retornar todas las hipótesis
        if return_all_beams:
            result["all_hypotheses"] = [
                {
                    "text": hyp,
                    "rank": i,
                }
                for i, hyp in enumerate(hypotheses)
            ]
        
        logger.info(f"✓ OCR completado: {len(words)} palabras, "
                   f"confianza promedio {result['ocr_avg_confidence']:.2%}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error extrayendo texto con confianza: {str(e)}")
        raise


def extract_text_with_confidence_robust(
    image_input: Union[np.ndarray, str, Path, bytes],
    model=None,
    processor=None,
    device: str = None,
    max_attempts: int = 4,
) -> Dict:
    """
    OCR robusto: prueba orientaciones y variantes de preprocesamiento,
    y selecciona el mejor resultado por calidad de texto.
    """
    try:
        if model is None or processor is None:
            model, processor, device = load_trocr_model()

        if isinstance(image_input, np.ndarray):
            base_image = image_input.copy()
        else:
            base_image = cv2.cvtColor(np.array(image_to_pil(image_input)), cv2.COLOR_RGB2BGR)

        orientation_candidates = [
            ("0", base_image),
            ("90cw", cv2.rotate(base_image, cv2.ROTATE_90_CLOCKWISE)),
            ("90ccw", cv2.rotate(base_image, cv2.ROTATE_90_COUNTERCLOCKWISE)),
            ("180", cv2.rotate(base_image, cv2.ROTATE_180)),
        ]

        best_result = None
        best_score = -1.0
        attempts = 0

        for orientation_name, oriented_img in orientation_candidates:
            for variant_name, variant_img in _prepare_ocr_variants(oriented_img):
                attempts += 1
                current = extract_text_with_confidence(
                    variant_img,
                    model=model,
                    processor=processor,
                    device=device,
                    return_all_beams=False,
                    num_beams=1,
                )
                quality = _text_quality_score(current.get("full_text", ""))

                if quality > best_score:
                    best_score = quality
                    best_result = current
                    best_result["orientation"] = orientation_name
                    best_result["preprocess_variant"] = variant_name
                    best_result["quality_score"] = quality

                if attempts >= max_attempts:
                    break

            if attempts >= max_attempts:
                break

        if best_result is None:
            return {
                "full_text": "",
                "tokens": [],
                "mrz_detected": False,
                "mrz_lines": [],
                "ocr_avg_confidence": 0.05,
                "orientation": "0",
                "preprocess_variant": "orig",
                "quality_score": 0.0,
            }

        return best_result

    except Exception as e:
        logger.error(f"Error en OCR robusto: {e}")
        return {
            "full_text": "",
            "tokens": [],
            "mrz_detected": False,
            "mrz_lines": [],
            "ocr_avg_confidence": 0.05,
            "error": str(e),
        }


def extract_mrz_text_from_roi(
    mrz_roi: np.ndarray,
    model=None,
    processor=None,
    device: str = None,
) -> Dict:
    """
    Extraer texto MRZ dedicado desde ROI detectada.
    Divide en 2 líneas e intenta OCR robusto por línea.
    """
    result = {
        "full_text": "",
        "mrz_lines": [],
        "mrz_detected": False,
        "ocr_avg_confidence": 0.0,
    }

    try:
        if mrz_roi is None or mrz_roi.size == 0:
            return result

        region = mrz_roi.copy()
        h, w = region.shape[:2]

        if h > w:
            region = cv2.rotate(region, cv2.ROTATE_90_CLOCKWISE)
            h, w = region.shape[:2]

        # 1) Intento rápido con EasyOCR (más robusto en fotos reales)
        easy_lines = _extract_mrz_lines_with_easyocr(region)
        if easy_lines:
            ranked = sorted(easy_lines, key=_mrz_line_likelihood, reverse=True)
            selected = [line for line in ranked if _mrz_line_likelihood(line) >= 0.30][:3]
            if not selected:
                selected = ranked[:2]

            if selected:
                result["mrz_detected"] = True
                result["mrz_lines"] = selected
                result["full_text"] = "\n".join(selected)
                easy_conf = np.mean([_mrz_line_likelihood(line) for line in selected])
                result["ocr_avg_confidence"] = float(max(0.35, easy_conf))
                result["ocr_engine"] = "easyocr"
                return result

        # 2) Respaldo TrOCR liviano por líneas
        if model is None or processor is None:
            model, processor, device = load_trocr_model()

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        prepared = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        split_y = h // 2
        line_regions = [
            prepared[:split_y, :],
            prepared[split_y:, :],
        ]

        confidences = []
        cleaned_lines = []

        for line_img in line_regions:
            line_up = cv2.resize(line_img, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            line_result = extract_text_with_confidence(
                line_up,
                model=model,
                processor=processor,
                device=device,
                num_beams=1,
            )
            line_text = line_result.get("full_text", "").upper().strip()
            line_text = ''.join(c for c in line_text if c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789< ')
            line_text = line_text.replace(' ', '<')

            if len(line_text) >= 10:
                likelihood = _mrz_line_likelihood(line_text)
                if likelihood >= 0.25:
                    cleaned_lines.append(line_text)
                    base_conf = float(line_result.get("ocr_avg_confidence", 0.0))
                    confidences.append(base_conf * likelihood)

        if cleaned_lines:
            result["mrz_detected"] = True
            result["mrz_lines"] = cleaned_lines
            result["full_text"] = "\n".join(cleaned_lines)
            result["ocr_avg_confidence"] = float(np.mean(confidences)) if confidences else 0.25

        return result

    except Exception as e:
        logger.error(f"Error OCR MRZ dedicado: {e}")
        result["error"] = str(e)
        return result


def recognize_text_batch(image_list: List[Union[np.ndarray, str, Path, bytes]],
                        batch_size: int = 4) -> List[Dict]:
    """
    Reconocer texto en lote de imágenes (más eficiente).
    
    Args:
        image_list: Lista de imágenes
        batch_size: Tamaño de lote
        
    Returns:
        Lista de resultados OCR
    """
    try:
        logger.info(f"Procesando lote de {len(image_list)} imágenes")
        
        model, processor, device = load_trocr_model()
        
        results = []
        
        for i in range(0, len(image_list), batch_size):
            batch = image_list[i:i+batch_size]
            
            # Convertir a PIL
            pil_images = [image_to_pil(img) for img in batch]
            
            # Procesar lote
            pixel_values = processor(images=pil_images, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)
            
            # Generar
            with torch.no_grad():
                generated_ids = model.generate(pixel_values, max_length=128)
            
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            for text in generated_texts:
                results.append({
                    "full_text": text,
                    "tokens": text.split(),
                    "ocr_avg_confidence": 0.85  # Valor por defecto para lote
                })
        
        logger.info(f"✓ Lote procesado: {len(results)} imágenes")
        
        return results
    
    except Exception as e:
        logger.error(f"Error procesando lote: {str(e)}")
        raise


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def clear_model_cache():
    """Limpiar caché del modelo (liberar memoria)."""
    global _model_cache
    
    if _model_cache["model"] is not None:
        logger.info("Limpiando caché del modelo")
        del _model_cache["model"]
        del _model_cache["processor"]
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    _model_cache = {
        "model": None,
        "processor": None,
        "device": None,
    }


def get_model_info() -> Dict:
    """
    Obtener información del modelo cargado.
    
    Returns:
        dict: Información del modelo
    """
    return {
        "model_name": config.TROCR_MODEL_NAME,
        "model_loaded": _model_cache["model"] is not None,
        "device": _model_cache.get("device", "not_loaded"),
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    }


if __name__ == "__main__":
    # Ejemplo de uso (requiere imagen real)
    logging.basicConfig(level=logging.DEBUG)
    
    print("Módulo ocr_engine.py cargado correctamente")
    print("Funciones principales:")
    print("  - extract_text_from_image() ← Extracción simple")
    print("  - extract_text_with_confidence() ← Con confianzas (recomendado)")
    print("  - recognize_text_batch() ← Procesamiento en lote")
    print("")
    print("Info del modelo:")
    info = get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
