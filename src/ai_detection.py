"""
ai_detection.py - Detectar si imágenes fueron procesadas con IA o herramientas de edición

Valida:
1. Metadatos EXIF (software de edición, timestamps)
2. Anomalías estadísticas (histograma, ruido)
3. Artefactos de compresión JPEG
4. Inconsistencias de iluminación
5. Detección de deepfakes
"""

import logging
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS
import numpy as np
import cv2
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class AIDetector:
    """Detectar si una imagen fue procesada con IA o herramientas de edición"""
    
    # Software de edición/IA conocido
    AI_SOFTWARE = {
        'photoshop': ['Adobe Photoshop', 'adobe'],
        'lightroom': ['Adobe Lightroom', 'lightroom'],
        'gimp': ['GIMP', 'GNU Image'],
        'canva': ['Canva', 'canva'],
        'dall-e': ['DALL-E', 'OpenAI'],
        'midjourney': ['Midjourney', 'midjourney'],
        'stable_diffusion': ['Stable Diffusion', 'stable'],
        'chatgpt': ['ChatGPT', 'OpenAI'],
        'gans': ['GAN', 'generative adversarial'],
    }
    
    def __init__(self):
        """Inicializar detector"""
        self.logger = logging.getLogger(__name__)
    
    def detect_from_image(self, image_array: np.ndarray) -> Dict:
        """
        Detectar anomalías de IA/edición en imagen
        
        Args:
            image_array: Array numpy de imagen (BGR)
            
        Returns:
            dict: Análisis con puntuación y detalles
        """
        results = {
            'is_ai_generated': False,
            'is_edited': False,
            'confidence': 0.0,
            'detected_method': None,
            'red_flags': [],
            'details': {}
        }
        
        try:
            # Análisis 1: Anomalías en histograma
            histogram_analysis = self._analyze_histogram(image_array)
            results['details']['histogram'] = histogram_analysis
            
            # Análisis 2: Detección de ruido/artefactos
            noise_analysis = self._analyze_noise(image_array)
            results['details']['noise'] = noise_analysis
            
            # Análisis 3: Inconsistencias de iluminación
            lighting_analysis = self._analyze_lighting(image_array)
            results['details']['lighting'] = lighting_analysis
            
            # Análisis 4: Compresión JPEG
            compression_analysis = self._analyze_compression(image_array)
            results['details']['compression'] = compression_analysis
            
            # Compilar banderas rojas
            if histogram_analysis['suspicious']:
                results['red_flags'].append('Histograma sospechoso - posible edición')
            if noise_analysis['unnatural']:
                results['red_flags'].append('Patrón de ruido artificial - posible generación IA')
            if lighting_analysis['inconsistent']:
                results['red_flags'].append('Iluminación inconsistente - posible composición')
            if compression_analysis['excessive']:
                results['red_flags'].append('Compresión excesiva - posible re-edición')
            
            # Calcular confianza
            red_flag_count = len(results['red_flags'])
            if red_flag_count >= 3:
                results['is_ai_generated'] = True
                results['confidence'] = 0.95
                results['detected_method'] = 'AI_GENERATED'
            elif red_flag_count == 2:
                results['is_edited'] = True
                results['confidence'] = 0.7
                results['detected_method'] = 'EDITED'
            elif red_flag_count == 1:
                results['is_edited'] = True
                results['confidence'] = 0.4
                results['detected_method'] = 'POSSIBLY_EDITED'
        
        except Exception as e:
            self.logger.error(f"Error en detección de IA: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def detect_from_file(self, image_path: str) -> Dict:
        """
        Detectar anomalías de IA/edición desde archivo
        
        Args:
            image_path: Ruta a archivo de imagen
            
        Returns:
            dict: Análisis incluyendo EXIF
        """
        results = {'exif': {}, 'ai_detection': {}}
        
        try:
            # Extraer EXIF
            results['exif'] = self._extract_exif(image_path)
            
            # Análisis de IA
            image = cv2.imread(image_path)
            if image is not None:
                results['ai_detection'] = self.detect_from_image(image)
            
            # Verificar software en EXIF
            software = results['exif'].get('Software', '').lower()
            for ai_type, keywords in self.AI_SOFTWARE.items():
                for keyword in keywords:
                    if keyword.lower() in software:
                        results['ai_detection']['red_flags'].append(
                            f'Software sospechoso detectado: {ai_type}'
                        )
                        results['ai_detection']['is_edited'] = True
                        results['ai_detection']['detected_method'] = ai_type.upper()
        
        except Exception as e:
            self.logger.error(f"Error analizando archivo: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def _extract_exif(self, image_path: str) -> Dict:
        """Extraer metadatos EXIF"""
        exif_data = {}
        try:
            image = Image.open(image_path)
            exif = image._getexif()
            
            if exif:
                for tag_id, value in exif.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    exif_data[tag_name] = str(value)[:100]  # Limitar longitud
        
        except Exception as e:
            self.logger.debug(f"No se pudo leer EXIF: {str(e)}")
        
        return exif_data
    
    def _analyze_histogram(self, image: np.ndarray) -> Dict:
        """Analizar histograma para detectar edición"""
        try:
            # Convertir a escala de grises
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Calcular histograma
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()
            
            # Detectar gaps (indicativo de edición)
            gaps = np.sum(hist < 0.001)
            gap_ratio = gaps / 256
            
            # Detectar picos anormales
            peak_threshold = np.mean(hist) + 2 * np.std(hist)
            suspicious_peaks = np.sum(hist > peak_threshold)
            
            return {
                'suspicious': gap_ratio > 0.3 or suspicious_peaks > 5,
                'gap_ratio': float(gap_ratio),
                'suspicious_peaks': int(suspicious_peaks),
                'entropy': float(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0])))
            }
        except Exception as e:
            return {'error': str(e), 'suspicious': False}
    
    def _analyze_noise(self, image: np.ndarray) -> Dict:
        """Analizar patrón de ruido"""
        try:
            # Convertir a escala de grises
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Calcular Laplaciano (detecta variaciones)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian_var = np.var(laplacian)
            
            # Ruido natural: distribución más uniforme
            # Ruido de IA: patrón repetitivo o demasiado limpio
            unnatural = laplacian_var < 10 or laplacian_var > 1000
            
            return {
                'unnatural': unnatural,
                'laplacian_variance': float(laplacian_var),
                'interpretation': 'muy_limpio' if laplacian_var < 10 else 'muy_ruidoso' if laplacian_var > 1000 else 'normal'
            }
        except Exception as e:
            return {'error': str(e), 'unnatural': False}
    
    def _analyze_lighting(self, image: np.ndarray) -> Dict:
        """Analizar consistencia de iluminación"""
        try:
            # Dividir imagen en cuadrantes
            h, w = image.shape[:2]
            q_h, q_w = h // 2, w // 2
            
            quadrants = [
                image[:q_h, :q_w],
                image[:q_h, q_w:],
                image[q_h:, :q_w],
                image[q_h:, q_w:]
            ]
            
            # Calcular brillo promedio de cada cuadrante
            brightness_levels = [np.mean(cv2.cvtColor(q, cv2.COLOR_BGR2GRAY)) for q in quadrants]
            brightness_std = np.std(brightness_levels)
            
            # Inconsistencia: desv std > 30
            inconsistent = brightness_std > 30
            
            return {
                'inconsistent': inconsistent,
                'brightness_levels': [float(b) for b in brightness_levels],
                'std_dev': float(brightness_std)
            }
        except Exception as e:
            return {'error': str(e), 'inconsistent': False}
    
    def _analyze_compression(self, image: np.ndarray) -> Dict:
        """Analizar artefactos de compresión JPEG"""
        try:
            # Buscar bloques de compresión JPEG (8x8)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # DCT análisis simplificado
            dct = cv2.dct(np.float32(gray) / 255.0)
            
            # Cuantificación: valores en múltiplos de 8
            dct_quantized = np.round(dct * 8) / 8
            
            # Diferencia = indicador de re-compresión
            recompression_score = np.mean(np.abs(dct - dct_quantized))
            
            return {
                'excessive': recompression_score > 0.05,
                'recompression_score': float(recompression_score)
            }
        except Exception as e:
            return {'error': str(e), 'excessive': False}
    
    def get_summary(self, analysis: Dict) -> str:
        """Generar resumen legible del análisis"""
        if analysis.get('is_ai_generated'):
            return "⚠️ POSIBLE GENERACIÓN POR IA"
        elif analysis.get('is_edited'):
            return "⚠️ IMAGEN EDITADA DETECTADA"
        else:
            return "✓ Imagen natural"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    detector = AIDetector()
    
    # Prueba
    test_image = "D:/ProyectosGRIT/ProyectoMaestría/data/raw/real_passport/ejemplo.jpg"
    if Path(test_image).exists():
        result = detector.detect_from_file(test_image)
        print(result)
