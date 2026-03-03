"""
preprocessing.py - Preprocesamiento de Imágenes de Pasaportes

Este módulo se encarga de:
1. Normalizar tamaño de imágenes
2. Detectar región de interés (ROI) del pasaporte
3. Mejorar contraste usando CLAHE
4. Reducir ruido
5. Corregir rotación

Versión: 0.1.0
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional
import logging

import config
from . import mrz_roi_detector

# ============================================================================
# SETUP LOGGING
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# FUNCIONES DE PREPROCESAMIENTO
# ============================================================================

def load_image(image_input: Union[str, Path, bytes, np.ndarray]) -> np.ndarray:
    """
    Cargar imagen desde múltiples formatos.
    
    Args:
        image_input: Ruta, bytes o array numpy de imagen
        
    Returns:
        np.ndarray: Imagen en formato BGR (OpenCV)
        
    Raises:
        ValueError: Si imagen no puede ser cargada
    """
    try:
        if isinstance(image_input, np.ndarray):
            # Ya es un array
            if len(image_input.shape) == 2:
                # Convertir grayscale a BGR
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            return image_input
        
        elif isinstance(image_input, (str, Path)):
            # Cargar desde archivo
            img = cv2.imread(str(image_input))
            if img is None:
                raise ValueError(f"No se pudo cargar imagen: {image_input}")
            return img
        
        elif isinstance(image_input, bytes):
            # Cargar desde bytes
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("No se pudo decodificar imagen desde bytes")
            return img
        
        else:
            raise ValueError(f"Tipo no soportado: {type(image_input)}")
    
    except Exception as e:
        logger.error(f"Error cargando imagen: {str(e)}")
        raise


def resize_image(image: np.ndarray, 
                size: Tuple[int, int] = config.IMAGE_SIZE) -> np.ndarray:
    """
    Redimensionar imagen a tamaño estándar.
    
    Args:
        image: Imagen OpenCV
        size: Tupla (ancho, alto)
        
    Returns:
        np.ndarray: Imagen redimensionada
    """
    try:
        height, width = size[1], size[0]
        resized = cv2.resize(image, (width, height), 
                            interpolation=cv2.INTER_LANCZOS4)
        logger.debug(f"Imagen redimensionada a {width}x{height}")
        return resized
    
    except Exception as e:
        logger.error(f"Error redimensionando imagen: {str(e)}")
        raise


def detect_passport_roi(image: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Detectar región de interés (ROI) del pasaporte.
    
    Para imágenes 2-page, detecta automáticamente cuál mitad contiene
    la foto y datos (mayor variación tonal) y la retorna.
    """
    try:
        h_img, w_img = image.shape[:2]
        aspect_ratio = h_img / w_img if w_img > 0 else 1
        
        logger.warning(f"🔍 DETECT_ROI: Aspecto={aspect_ratio:.3f}, Dim={w_img}x{h_img}")
        
        # CASO 1: DOS PÁGINAS LADO A LADO (aspect muy pequeño)
        if aspect_ratio < 0.5:
            logger.warning(f"✓ DETECTADO: 2 PÁGINAS LADO A LADO")
            
            half_width = w_img // 2
            
            # Convertir a escala de grises para análisis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Analizar variación tonal en MITAD IZQUIERDA
            left_half = gray[:, :half_width]
            left_std = np.std(left_half)  # Mayor std = más detalle (foto)
            
            # Analizar variación tonal en MITAD DERECHA
            right_half = gray[:, half_width:]
            right_std = np.std(right_half)
            
            logger.warning(f"  Variación tonal izquierda: {left_std:.1f}")
            logger.warning(f"  Variación tonal derecha: {right_std:.1f}")
            
            # Seleccionar la mitad con MÁS variación (= con foto/detalles)
            if left_std > right_std:
                logger.warning(f"  → PÁGINA PRINCIPAL EN IZQUIERDA ✓ (mayor detalle)")
                roi = (0, 0, half_width, h_img)
            else:
                logger.warning(f"  → PÁGINA PRINCIPAL EN DERECHA ✓ (mayor detalle)")
                roi = (half_width, 0, half_width, h_img)
            
            x, y, w, h = roi
            logger.warning(f"  ROI FINAL: x={x}, y={y}, w={w}, h={h}")
            return roi

        # CASO 1B: PROBABLE DOBLE PÁGINA HORIZONTAL (una mitad con mucho más detalle)
        # Ejemplo típico: página de datos + página en blanco/semiblanco
        elif w_img > h_img:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            half_width = w_img // 2

            left_half = gray[:, :half_width]
            right_half = gray[:, half_width:]

            left_std = float(np.std(left_half))
            right_std = float(np.std(right_half))
            std_ratio = (max(left_std, right_std) / max(1.0, min(left_std, right_std)))

            logger.warning(
                f"  Análisis horizontal: std_left={left_std:.1f}, std_right={right_std:.1f}, ratio={std_ratio:.2f}"
            )

            if std_ratio >= 1.25:
                logger.warning("✓ DETECTADO: probable doble página por asimetría tonal")

                if left_std > right_std:
                    logger.warning("  → PÁGINA PRINCIPAL EN IZQUIERDA ✓")
                    roi = (0, 0, half_width, h_img)
                else:
                    logger.warning("  → PÁGINA PRINCIPAL EN DERECHA ✓")
                    roi = (half_width, 0, half_width, h_img)

                x, y, w, h = roi
                logger.warning(f"  ROI FINAL: x={x}, y={y}, w={w}, h={h}")
                return roi
        
        # CASO 2: DOS PÁGINAS VERTICALES (aspect grande)
        elif aspect_ratio > 1.2:
            logger.warning(f"✓ DETECTADO: 2 PÁGINAS VERTICALES")
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Dividir en mitad superior e inferior
            half_height = h_img // 2
            
            top_half = gray[:half_height, :]
            top_std = np.std(top_half)
            
            bottom_half = gray[half_height:, :]
            bottom_std = np.std(bottom_half)
            
            logger.warning(f"  Variación tonal superior: {top_std:.1f}")
            logger.warning(f"  Variación tonal inferior: {bottom_std:.1f}")
            
            # La mitad con MÁS variación tiene los datos
            if top_std > bottom_std:
                logger.warning(f"  → PÁGINA PRINCIPAL ARRIBA ✓ (mayor detalle)")
                new_height = half_height
                new_y = 0
            else:
                logger.warning(f"  → PÁGINA PRINCIPAL ABAJO ✓ (mayor detalle)")
                new_height = int(h_img * 0.6)
                new_y = h_img - new_height
            
            roi = (0, new_y, w_img, new_height)
            logger.warning(f"  ROI FINAL: x=0, y={new_y}, w={w_img}, h={new_height}")
            return roi
        
        # CASO 3: IMAGEN NORMAL (1 página)
        else:
            logger.warning(f"✓ DETECTADO: Imagen NORMAL (1 página)")
            return (0, 0, w_img, h_img)
        
    except Exception as e:
        logger.error(f"❌ Error en detect_passport_roi: {e}")
        h, w = image.shape[:2]
        return (0, 0, w, h)


def crop_roi(image: np.ndarray, 
            roi: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """
    Recortar imagen a región de interés.
    
    Args:
        image: Imagen OpenCV
        roi: Tupla (x, y, w, h). Si es None, usa detect_passport_roi()
        
    Returns:
        np.ndarray: Imagen recortada
    """
    try:
        if roi is None:
            roi = detect_passport_roi(image)
        
        x, y, w, h = roi
        cropped = image[y:y+h, x:x+w]
        
        if cropped.size == 0:
            logger.warning("ROI vacío, retornando imagen original")
            return image
        
        logger.debug(f"Imagen recortada a {cropped.shape}")
        return cropped
    
    except Exception as e:
        logger.error(f"Error recortando imagen: {str(e)}")
        return image


def ensure_landscape_orientation(image: np.ndarray) -> np.ndarray:
    """
    Asegurar orientación horizontal (landscape) para OCR de pasaporte.

    Si la imagen viene en orientación vertical (alto > ancho), rota 90° CW.
    """
    try:
        h, w = image.shape[:2]
        if h > w:
            logger.warning(f"↻ Rotando imagen a landscape: {w}x{h} -> {h}x{w}")
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return image
    except Exception as e:
        logger.warning(f"Error asegurando orientación landscape: {str(e)}")
        return image


def _order_quad_points(points: np.ndarray) -> np.ndarray:
    """Ordenar puntos de un cuadrilátero: TL, TR, BR, BL."""
    rect = np.zeros((4, 2), dtype="float32")
    point_sums = points.sum(axis=1)
    point_diffs = np.diff(points, axis=1)

    rect[0] = points[np.argmin(point_sums)]
    rect[2] = points[np.argmax(point_sums)]
    rect[1] = points[np.argmin(point_diffs)]
    rect[3] = points[np.argmax(point_diffs)]
    return rect


def correct_document_perspective(image: np.ndarray) -> np.ndarray:
    """
    Corregir perspectiva del documento usando contorno principal.

    Si no se detecta un cuadrilátero confiable, retorna la imagen original.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image

        h, w = image.shape[:2]
        image_area = float(h * w)

        best_quad = None
        best_area = 0.0

        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) != 4:
                continue

            quad = approx.reshape(4, 2).astype("float32")
            area = cv2.contourArea(quad)
            area_ratio = area / image_area

            if area_ratio < 0.25:
                continue

            if area > best_area:
                best_area = area
                best_quad = quad

        if best_quad is None:
            return image

        rect = _order_quad_points(best_quad)
        (tl, tr, br, bl) = rect

        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        max_width = int(max(width_top, width_bottom))

        height_right = np.linalg.norm(br - tr)
        height_left = np.linalg.norm(bl - tl)
        max_height = int(max(height_right, height_left))

        if max_width < 100 or max_height < 100:
            return image

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ], dtype="float32")

        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(
            image,
            matrix,
            (max_width, max_height),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        logger.warning(f"✓ Perspectiva corregida: {w}x{h} -> {max_width}x{max_height}")
        return warped

    except Exception as e:
        logger.warning(f"Error corrigiendo perspectiva: {str(e)}")
        return image


def adjust_brightness_contrast(image: np.ndarray) -> np.ndarray:
    """
    Ajustar brillo y contraste usando CLAHE.
    
    CLAHE (Contrast Limited Adaptive Histogram Equalization)
    mejora el contraste localmente sin crear artefactos.
    
    Args:
        image: Imagen OpenCV (BGR)
        
    Returns:
        np.ndarray: Imagen con contraste mejorado
    """
    try:
        # Convertir a LAB para procesar luminosidad por separado
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Aplicar CLAHE solo al canal L (luminosidad)
        clahe = cv2.createCLAHE(
            clipLimit=config.CLAHE_CLIP_LIMIT,
            tileGridSize=config.CLAHE_TILE_SIZE
        )
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convertir de vuelta a BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        logger.debug("Contraste ajustado con CLAHE")
        return enhanced
    
    except Exception as e:
        logger.warning(f"Error ajustando contraste: {str(e)}")
        return image


def remove_noise(image: np.ndarray) -> np.ndarray:
    """
    Reducir ruido usando filtro bilateral.
    
    El filtro bilateral es excelente para denoising porque
    preserva bordes mientras reduce ruido.
    
    Args:
        image: Imagen OpenCV
        
    Returns:
        np.ndarray: Imagen sin ruido
    """
    try:
        # Bilateral filter: excelente para preservar bordes
        denoised = cv2.bilateralFilter(
            image, 
            d=9,  # Diámetro del pixel vecindario
            sigmaColor=75,  # Rango de color
            sigmaSpace=75   # Rango espacial
        )
        
        logger.debug("Ruido removido con bilateral filter")
        return denoised
    
    except Exception as e:
        logger.warning(f"Error removiendo ruido: {str(e)}")
        return image


def correct_rotation(image: np.ndarray) -> np.ndarray:
    """
    Detectar y corregir rotación de imagen.
    
    Usa detección de líneas Hough para encontrar ángulo
    de rotación y corregir si es necesario.
    
    Args:
        image: Imagen OpenCV
        
    Returns:
        np.ndarray: Imagen corregida de rotación
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detectar bordes
        edges = cv2.Canny(gray, 50, 150)
        
        # Usar Hough para detectar líneas
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        
        if lines is None or len(lines) < 5:
            # No hay suficientes líneas para detectar rotación
            logger.debug("No se detectó rotación significativa")
            return image
        
        # Calcular ángulo promedio de las líneas
        angles = []
        for line in lines[:20]:  # Usar las primeras 20 líneas
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90
            angles.append(angle)
        
        avg_angle = np.median(angles)
        
        # Si ángulo es pequeño, es probable que sea ruido
        if abs(avg_angle) < 5:
            logger.debug(f"Ángulo detectado: {avg_angle:.1f}° (< 5°, ignorado)")
            return image
        
        # Corregir rotación
        logger.debug(f"Corrigiendo rotación: {avg_angle:.1f}°")
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Matriz de rotación
        M = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
        
        # Aplicar rotación
        rotated = cv2.warpAffine(image, M, (w, h),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))
        
        return rotated
    
    except Exception as e:
        logger.warning(f"Error corrigiendo rotación: {str(e)}")
        return image


def validate_image_quality(image: np.ndarray) -> Tuple[bool, str]:
    """
    Validar calidad de imagen.
    
    Verifica:
    - Dimensiones mínimas
    - Brillo promedio (no muy oscura, no muy clara)
    - Contraste
    
    Args:
        image: Imagen OpenCV
        
    Returns:
        Tupla (es_válida, mensaje)
    """
    try:
        h, w = image.shape[:2]
        
        # Verificar dimensiones
        if w < 400 or h < 300:
            return False, f"Imagen muy pequeña: {w}x{h}"
        
        # Verificar brillo
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        if brightness < 30:
            return False, f"Imagen muy oscura: {brightness:.1f}"
        elif brightness > 245:  # Relajado de 225 para permitir fotos claras de pasaportes
            return False, f"Imagen muy clara: {brightness:.1f}"
        
        # Verificar contraste (desviación estándar)
        contrast = np.std(gray)
        if contrast < 10:
            return False, f"Contraste muy bajo: {contrast:.1f}"
        
        return True, "Imagen válida"
    
    except Exception as e:
        logger.warning(f"Error validando calidad: {str(e)}")
        return False, str(e)


# ============================================================================
# PIPELINE COMPLETO
# ============================================================================

def preprocess_pipeline(image_input: Union[str, Path, bytes, np.ndarray],
                       validate: bool = True) -> Tuple[np.ndarray, dict]:
    """
    Pipeline completo de preprocesamiento.
    
    Secuencia:
    1. Cargar imagen
    2. Validar calidad
    3. Redimensionar
    4. Detectar y recortar ROI
    5. Ajustar brillo/contraste (CLAHE)
    6. Remover ruido
    7. Corregir rotación
    
    Args:
        image_input: Imagen en múltiples formatos
        validate: Si validar calidad de imagen original
        
    Returns:
        Tupla (imagen_procesada, metadatos)
        
        metadatos contiene:
        - "original_shape": forma de imagen original
        - "is_valid": si pasó validación de calidad
        - "quality_message": mensaje de validación
        - "roi": coordenadas del ROI
        - "rotation_corrected": si fue rotación corregida
    """
    metadata = {
        "original_shape": None,
        "is_valid": True,
        "quality_message": "OK",
        "roi": None,
        "rotation_corrected": False,
        "perspective_corrected": False,
    }
    
    try:
        logger.info("Iniciando pipeline de preprocesamiento")
        
        # 1. Cargar imagen
        logger.debug("Paso 1: Cargando imagen")
        image = load_image(image_input)
        metadata["original_shape"] = image.shape
        h_original, w_original = image.shape[:2]
        logger.warning(f"IMAGEN ORIGINAL: {w_original}x{h_original}, aspect={h_original/w_original:.3f}")
        
        # 2. Validar calidad original
        if validate:
            logger.debug("Paso 2: Validando calidad de imagen")
            is_valid, message = validate_image_quality(image)
            metadata["is_valid"] = is_valid
            metadata["quality_message"] = message
            
            if not is_valid:
                logger.warning(f"Validación fallida: {message}")
        
        # 3. **DETECTAR ROI EN IMAGEN ORIGINAL** (ANTES de redimensionar)
        logger.debug("Paso 3: Detectando ROI del pasaporte (IMAGEN ORIGINAL)")
        roi = detect_passport_roi(image)
        metadata["roi"] = roi
        logger.warning(f"ROI DETECTADO: {roi}")
        
        # 4. Recortar ROI
        logger.debug("Paso 4: Recortando a ROI")
        image = crop_roi(image, roi)
        
        # 5. Corregir orientación a landscape antes de OCR
        logger.debug("Paso 5: Asegurando orientación landscape")
        image = ensure_landscape_orientation(image)

        # 6. Corregir perspectiva del documento
        logger.debug("Paso 6: Corrigiendo perspectiva del documento")
        perspective_input_shape = image.shape
        image = correct_document_perspective(image)
        metadata["perspective_corrected"] = image.shape != perspective_input_shape

        # 7. AHORA redimensionar región recortada
        logger.debug("Paso 7: Redimensionando región recortada")
        image = resize_image(image)
        
        # 8. Ajustar contraste
        logger.debug("Paso 8: Ajustando brillo y contraste")
        image = adjust_brightness_contrast(image)
        
        # 9. Remover ruido
        logger.debug("Paso 9: Removiendo ruido")
        image = remove_noise(image)
        
        # 10. Corregir rotación fina
        logger.debug("Paso 10: Corrigiendo rotación")
        # (el método ya registra si corrigió algo)
        image = correct_rotation(image)
        
        logger.info("✓ Pipeline de preprocesamiento completado exitosamente")
        return image, metadata
    
    except Exception as e:
        logger.error(f"Error en pipeline de preprocesamiento: {str(e)}")
        raise


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def save_processed_image(image: np.ndarray, 
                        output_path: Union[str, Path]) -> bool:
    """
    Guardar imagen procesada.
    
    Args:
        image: Imagen OpenCV
        output_path: Ruta donde guardar
        
    Returns:
        bool: Si se guardó exitosamente
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        success = cv2.imwrite(str(output_path), image)
        if success:
            logger.debug(f"Imagen guardada en {output_path}")
            return True
        else:
            logger.error(f"Error guardando imagen en {output_path}")
            return False
    
    except Exception as e:
        logger.error(f"Error guardando imagen: {str(e)}")
        return False


def get_preprocessing_stats(image_original: np.ndarray, 
                           image_processed: np.ndarray) -> dict:
    """
    Calcular estadísticas de preprocesamiento.
    
    Compara imagen original vs procesada para ver mejoras.
    
    Args:
        image_original: Imagen original
        image_processed: Imagen procesada
        
    Returns:
        dict: Estadísticas de mejora
    """
    try:
        gray_orig = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
        gray_proc = cv2.cvtColor(image_processed, cv2.COLOR_BGR2GRAY)
        
        stats = {
            "brightness_original": float(np.mean(gray_orig)),
            "brightness_processed": float(np.mean(gray_proc)),
            "contrast_original": float(np.std(gray_orig)),
            "contrast_processed": float(np.std(gray_proc)),
            "brightness_change": float(np.mean(gray_proc) - np.mean(gray_orig)),
            "contrast_improvement": float(np.std(gray_proc) - np.std(gray_orig)),
        }
        
        return stats
    
    except Exception as e:
        logger.error(f"Error calculando estadísticas: {str(e)}")
        return {}


if __name__ == "__main__":
    # Ejemplo de uso
    logging.basicConfig(level=logging.DEBUG)
    
    print("Módulo preprocessing.py cargado correctamente")
    print("Funciones disponibles:")
    print("  - load_image()")
    print("  - resize_image()")
    print("  - detect_passport_roi()")
    print("  - adjust_brightness_contrast()")
    print("  - remove_noise()")
    print("  - correct_rotation()")
    print("  - preprocess_pipeline()  ← Función principal")
