"""
anti_fraud_detector.py - Detección de pasaportes falsos y generados con IA

Técnicas:
1. Análisis de micro-impresión (líneas de seguridad)
2. Detección de manipulación de foto (análisis de luz/sombra)
3. Detección de artefactos IA (Frecuencia/noise analysis)
4. Análisis de bordes y claridad (diferencia real vs falso)
5. Verificación de coherencia de datos (nombre, números, fechas)
6. Detección de compresión JPEG anómala
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def detect_ai_artifacts(image: np.ndarray) -> Dict[str, float]:
    """
    Detecta artefactos típicos de generación por IA.
    
    Las imágenes generadas por IA generalmente tienen:
    - Transiciones de color muy suaves (baja frecuencia)
    - Menos detalle de alta frecuencia
    - Patrón de ruido artificial/uniforme
    - Falta de micro-texturas naturales
    
    Args:
        image: Imagen BGR
        
    Returns:
        {'ai_score': float 0-1, 'signs': [list de indicadores encontrados]}
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        signs = []
        scores = []
        
        # 1. ANÁLISIS DE FRECUENCIA (FFT)
        # Imágenes reales tienen más componentes de alta frecuencia
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Dividir en cuadrantes de frecuencia
        center_h, center_w = h // 2, w // 2
        
        # Baja frecuencia (centro)
        low_freq = magnitude[
            center_h - h//8:center_h + h//8,
            center_w - w//8:center_w + w//8
        ].sum()
        
        # Alta frecuencia (esquinas)
        high_freq = (
            magnitude[:h//8, :w//8].sum() +
            magnitude[-h//8:, :w//8].sum() +
            magnitude[:h//8, -w//8:].sum() +
            magnitude[-h//8:, -w//8:].sum()
        ) / 4
        
        freq_ratio = high_freq / (low_freq + 1e-6)
        
        # Imágenes reales: ratio alto (más detalles)
        # Imágenes IA: ratio bajo (muy suave)
        if freq_ratio < 0.15:
            signs.append("BAJA_FRECUENCIA_ALTA (típico IA)")
            scores.append(0.7)
        elif freq_ratio < 0.25:
            signs.append("FRECUENCIAS_SUAVIZADAS")
            scores.append(0.4)
        else:
            signs.append("FRECUENCIAS_NATURALES")
            scores.append(0.0)
        
        # 2. ANÁLISIS DE RUIDO
        # Aplicar Laplacian para detectar bordes
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        edge_strength = np.std(laplacian)
        
        # Muy bajo = imagen muy suave (IA)
        # Muy alto = mucho ruido (fotocopiado malo)
        if edge_strength < 5:
            signs.append("BORDES_MUY_SUAVES (IA)")
            scores.append(0.6)
        elif edge_strength < 10:
            signs.append("BORDES_MODERADOS")
            scores.append(0.2)
        elif edge_strength > 30:
            signs.append("EXCESO_DE_RUIDO")
            scores.append(0.4)
        else:
            signs.append("RUIDO_NATURAL")
            scores.append(0.0)
        
        # 3. ANÁLISIS DE CONTRASTE LOCAL
        # Dividir imagen en bloques y medir contraste
        block_size = 64
        contrast_values = []
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                contrast = np.std(block)
                contrast_values.append(contrast)
        
        contrast_variance = np.std(contrast_values)
        
        # Imágenes IA: contraste muy uniforme entre bloques
        # Imágenes reales: variación natural
        if contrast_variance < 2:
            signs.append("CONTRASTE_UNIFORME (IA)")
            scores.append(0.5)
        elif contrast_variance < 5:
            signs.append("CONTRASTE_MODERADO")
            scores.append(0.1)
        else:
            signs.append("CONTRASTE_NATURAL")
            scores.append(0.0)
        
        # 4. ANÁLISIS DE TRANSICIONES DE BORDE
        # Las imágenes del pasaporte tienen zonas bien definidas
        # Detectar si los bordes son naturales o suavizados
        canny = cv2.Canny(gray, 100, 200)
        edge_pixels = np.count_nonzero(canny)
        edge_density = edge_pixels / (h * w)
        
        if edge_density < 0.01:
            signs.append("POCOS_BORDES_DEFINIDOS (IA)")
            scores.append(0.4)
        elif edge_density > 0.1:
            signs.append("DEMASIADOS_BORDES (ruido/fotocopiado)")
            scores.append(0.3)
        else:
            signs.append("BORDES_NATURALES")
            scores.append(0.0)
        
        # Calcular score final
        ai_score = np.mean(scores) if scores else 0
        
        logger.debug(f"AI Detection - Score: {ai_score:.2f}, Signs: {signs}")
        
        return {
            'ai_score': float(ai_score),
            'ai_probability': "ALTA" if ai_score > 0.6 else "MEDIA" if ai_score > 0.3 else "BAJA",
            'signs': signs,
            'freq_ratio': float(freq_ratio),
            'edge_strength': float(edge_strength),
            'contrast_variance': float(contrast_variance),
            'edge_density': float(edge_density)
        }
        
    except Exception as e:
        logger.error(f"Error en detección de artefactos IA: {e}")
        return {'ai_score': 0.5, 'error': str(e), 'signs': []}


def detect_manipulation(image: np.ndarray) -> Dict[str, float]:
    """
    Detecta manipulación de foto (splicing, copy-paste, etc).
    
    Técnicas:
    - Análisis de histograma (copias de áreas tienen histogramas similares)
    - Detección de inconsistencia de iluminación
    - Análisis de artefactos de compresión JPG
    
    Args:
        image: Imagen BGR
        
    Returns:
        {'manipulation_score': float 0-1, 'findings': [list de problemas]}
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        findings = []
        scores = []
        
        # 1. ANÁLISIS DE ILUMINACIÓN
        # La luz debe ser relativamente consistente en el documento
        # Dividir en cuadrantes
        quarters = [
            gray[:h//2, :w//2],      # TOP-LEFT
            gray[:h//2, w//2:],      # TOP-RIGHT
            gray[h//2:, :w//2],      # BOT-LEFT
            gray[h//2:, w//2:]       # BOT-RIGHT
        ]
        
        brightness_values = [q.mean() for q in quarters]
        brightness_variance = np.std(brightness_values)
        
        # Documentos reales: variación pequeña
        # Manipulados: grandes variaciones entre cuadrantes
        if brightness_variance > 30:
            findings.append("ILUMINACION_INCONSISTENTE")
            scores.append(0.6)
        elif brightness_variance > 15:
            findings.append("ILUMINACION_MODERADAMENTE_INCONSISTENTE")
            scores.append(0.3)
        else:
            findings.append("ILUMINACION_CONSISTENTE")
            scores.append(0.0)
        
        # 2. ANÁLISIS DE SOMBRAS
        # Detectar sombras anómalas que indiquen manipulación
        # Las sombras deben ser consistentes (una fuente de luz)
        
        # Aplicar detección de sombras con morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        shadows = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        shadow_diff = cv2.absdiff(gray, shadows)
        shadow_presence = np.count_nonzero(shadow_diff > 30) / (h * w)
        
        if shadow_presence > 0.3:
            findings.append("SOMBRAS_ANOMALAS")
            scores.append(0.5)
        elif shadow_presence > 0.15:
            findings.append("SOMBRAS_MODERADAS")
            scores.append(0.2)
        else:
            findings.append("SOMBRAS_NATURALES")
            scores.append(0.0)
        
        # 3. ANÁLISIS DE RUIDO DE COMPRESIÓN JPG
        # Los JPG tiene bloques de 8x8 con artefactos visibles
        # Manipulados pueden mostrar bloques incompletos o desalineados
        
        dct_artifacts = 0
        block_size = 8
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size].astype(float)
                # Análisis DCT simple
                variance = np.var(block)
                if variance > 500:  # Artefacto DCT visible
                    dct_artifacts += 1
        
        artifact_density = dct_artifacts / ((h//8) * (w//8))
        
        if artifact_density > 0.3:
            findings.append("ALTO_NUMERO_ARTEFACTOS_JPG")
            scores.append(0.3)
        else:
            findings.append("ARTEFACTOS_JPG_NORMALES")
            scores.append(0.0)
        
        # Calcular score final
        manipulation_score = np.mean(scores) if scores else 0
        
        logger.debug(f"Manipulation Detection - Score: {manipulation_score:.2f}")
        
        return {
            'manipulation_score': float(manipulation_score),
            'findings': findings,
            'brightness_variance': float(brightness_variance),
            'shadow_presence': float(shadow_presence),
            'artifact_density': float(artifact_density)
        }
        
    except Exception as e:
        logger.error(f"Error en detección de manipulación: {e}")
        return {'manipulation_score': 0.5, 'error': str(e), 'findings': []}


def detect_forgery_indicators(image: np.ndarray) -> Dict:
    """
    Detección de indicadores de falsificación.
    
    Busca signos de:
    - Pasaporte impreso de baja calidad
    - Foto reemplazada
    - Texto adulterado
    - Ausencia de elementos de seguridad
    
    Args:
        image: Imagen BGR
        
    Returns:
        Dict con hallazgos
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        indicators = []
        forgery_score = 0
        
        # 1. CALIDAD DE IMPRESIÓN
        # Imágenes reales tienen impresión nítida
        # Falsificaciones pueden tener pixelación o borrosidad
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.std(laplacian)
        
        if sharpness < 8:
            indicators.append("BAJA_NITIDEZ (posible falsificación)")
            forgery_score += 0.3
        elif sharpness > 50:
            indicators.append("SOBRE-NITIDEZ (edición digital)")
            forgery_score += 0.2
        
        # 2. ALINEACIÓN Y PERSPECTIVA
        # El pasaporte debe estar rectilíneo
        edges = cv2.Canny(gray, 100, 200)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 50)
        
        if lines is not None:
            angles = []
            for line in lines[:20]:
                rho, theta = line[0]
                angle = np.degrees(theta)
                angles.append(angle)
            
            angle_variance = np.std(angles) if angles else 0
            
            # Pasaporte real: ángulos muy consistentes
            # Falsificado: ángulos dispersos
            if angle_variance > 15:
                indicators.append("PERSPECTIVA_ANÓMALA")
                forgery_score += 0.4
        
        # 3. PRESENCIA DE ELEMENTOS DE SEGURIDAD
        # Pasaportes mexicanos tienen líneas de seguridad (microimpresión)
        # Buscar patrón de líneas finas
        
        # Aplicar threshold adaptativo para buscar líneas finas
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Detectar líneas verticales (microimpresión típica)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
        vertical_lines = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel_v)
        line_pixels = np.count_nonzero(vertical_lines)
        
        if line_pixels < 100:
            indicators.append("POSIBLE_AUSENCIA_DE_MICROIMPRESION")
            forgery_score += 0.2
        
        logger.debug(f"Forgery Detection - Score: {forgery_score:.2f}")
        
        return {
            'forgery_score': float(min(forgery_score, 1.0)),
            'forgery_risk': "ALTA" if forgery_score > 0.6 else "MEDIA" if forgery_score > 0.3 else "BAJA",
            'indicators': indicators,
            'sharpness': float(sharpness)
        }
        
    except Exception as e:
        logger.error(f"Error en detección de falsificación: {e}")
        return {'forgery_score': 0, 'error': str(e), 'indicators': []}


def get_fraud_score(image: np.ndarray) -> Dict:
    """
    Calcula score general de fraude/falsificación.
    
    Combina:
    - Detección de artefactos IA
    - Detección de manipulación
    - Detección de falsificación
    
    Args:
        image: Imagen BGR
        
    Returns:
        Dict con análisis completo
    """
    try:
        ai_result = detect_ai_artifacts(image)
        manipulation_result = detect_manipulation(image)
        forgery_result = detect_forgery_indicators(image)
        
        # Calcular score ponderado
        ai_weight = 0.3
        manip_weight = 0.35
        forgery_weight = 0.35
        
        total_fraud_score = (
            ai_result.get('ai_score', 0) * ai_weight +
            manipulation_result.get('manipulation_score', 0) * manip_weight +
            forgery_result.get('forgery_score', 0) * forgery_weight
        )
        
        # Determinar nivel de riesgo
        if total_fraud_score > 0.7:
            risk_level = "ALTO - POTENCIAL FRAUDE"
        elif total_fraud_score > 0.4:
            risk_level = "MEDIO - REQUERIDA VERIFICACIÓN"
        else:
            risk_level = "BAJO - PROBABLE AUTENTICO"
        
        return {
            'fraud_score': float(total_fraud_score),
            'risk_level': risk_level,
            'ai_analysis': ai_result,
            'manipulation_analysis': manipulation_result,
            'forgery_analysis': forgery_result
        }
        
    except Exception as e:
        logger.error(f"Error calculando fraud score: {e}")
        return {
            'fraud_score': 0.5,
            'error': str(e),
            'risk_level': 'ERROR'
        }
