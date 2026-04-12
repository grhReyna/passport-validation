"""
ai_detection.py - Detectar si imágenes fueron procesadas con IA o herramientas de edición

Valida:
1. Metadatos EXIF (software de edición, timestamps)
2. Anomalías estadísticas (histograma, ruido)
3. Artefactos de compresión JPEG
4. Inconsistencias de iluminación
5. Detección de deepfakes
6. ELA (Error Level Analysis) - detecta regiones manipuladas/generadas
7. Análisis de textura facial - detecta caras IA demasiado perfectas
8. Análisis de frecuencia - detecta patrones de generadores IA
9. Correlación de canales de color - detecta anomalías en RGB
"""

import logging
import io
import struct
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional

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
    
    def detect_from_image(self, image_array: np.ndarray, raw_bytes: Optional[bytes] = None) -> Dict:
        """
        Detectar anomalías de IA/edición en imagen.
        
        Usa un sistema de scoring continuo en lugar de banderas binarias
        para reducir falsos positivos en pasaportes reales.
        
        Args:
            image_array: Array numpy de imagen (BGR)
            raw_bytes: Bytes crudos del archivo (para análisis de chunks/metadata)
            
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
            
            # Análisis 5: ELA (Error Level Analysis)
            ela_analysis = self._analyze_ela(image_array)
            results['details']['ela'] = ela_analysis
            
            # Análisis 6: Textura facial (caras IA demasiado perfectas)
            face_analysis = self._analyze_face_texture(image_array)
            results['details']['face'] = face_analysis
            
            # Análisis 7: Frecuencia (patrones de generadores IA)
            frequency_analysis = self._analyze_frequency(image_array)
            results['details']['frequency'] = frequency_analysis
            
            # Análisis 8: Correlación de canales de color
            color_analysis = self._analyze_color_channels(image_array)
            results['details']['color'] = color_analysis
            
            # Análisis 9: Chunk data / metadata binaria
            chunk_analysis = self._analyze_chunks(raw_bytes) if raw_bytes else {'suspicious': False, 'ai_markers': []}
            results['details']['chunks'] = chunk_analysis
            
            # ====== SCORING CONTINUO ======
            # Cada análisis aporta un score de sospecha (0.0 = limpio, 1.0 = muy sospechoso)
            suspicion_score = 0.0
            
            # --- Histograma ---
            peaks = histogram_analysis.get('suspicious_peaks', 0)
            gap = histogram_analysis.get('gap_ratio', 0)
            
            # La combinación de gap_ratio bajo + peaks altos es el indicador más fuerte.
            # Pasaportes reales: gap > 0.20 (distribución amplia).
            # Imágenes IA/falsas: gap < 0.15 con peaks >= 27 (distribución concentrada).
            if gap < 0.15 and peaks >= 27:
                suspicion_score += 0.50
                results['red_flags'].append('Histograma sospechoso - distribución artificial')
            elif gap < 0.10 and peaks >= 25:
                suspicion_score += 0.45
                results['red_flags'].append('Histograma con distribución concentrada')
            elif gap < 0.15 and peaks >= 25:
                suspicion_score += 0.35
                results['red_flags'].append('Histograma con anomalías moderadas')
            # peaks >= 20 con gap > 0.20 = normal para fotos reales con detalle
            # No suma sospecha
            
            # --- Ruido / Laplacian ---
            lap_var = noise_analysis.get('laplacian_variance', 1500)
            
            # IMPORTANTE: El laplacian varía enormemente según cómo se tome la foto.
            # Una foto limpia de página sola puede tener laplacian = 30-150 y ser REAL.
            # Por eso el laplacian SOLO no debe ser un indicador fuerte.
            # Solo pesa fuerte cuando se COMBINA con histograma sospechoso.
            
            histogram_is_suspicious = (gap < 0.15 and peaks >= 25)  # Histograma apunta a IA
            
            if lap_var < 8:
                # Extremadamente limpio — solo imágenes sintéticas perfectas
                suspicion_score += 0.30 if histogram_is_suspicious else 0.10
                if histogram_is_suspicious:
                    results['red_flags'].append('Imagen demasiado limpia + histograma artificial')
            elif lap_var < 350:
                if histogram_is_suspicious:
                    # Combo fuerte: laplacian bajo + histograma artificial = IA
                    suspicion_score += 0.35
                    results['red_flags'].append('Patrón de ruido artificial + histograma sospechoso')
                # Sin histograma sospechoso, laplacian bajo solo = foto limpia, NO suma
            elif lap_var < 800:
                if histogram_is_suspicious:
                    suspicion_score += 0.15
                # Sin histograma sospechoso, zona gris NO suma
            elif lap_var > 4500:
                suspicion_score += 0.20 if histogram_is_suspicious else 0.10
                results['red_flags'].append('Ruido excesivo - posible manipulación')
            # 800-4500 = rango normal para fotos reales, no suma sospecha
            
            # --- Iluminación ---
            if lighting_analysis.get('inconsistent'):
                suspicion_score += 0.15
                results['red_flags'].append('Iluminación inconsistente - posible composición')
            
            # --- Compresión ---
            if compression_analysis.get('excessive'):
                suspicion_score += 0.10
                results['red_flags'].append('Compresión excesiva - posible re-edición')
            
            # --- ELA (Error Level Analysis) ---
            # NOTA: Documentos impresos (pasaportes) tienen ELA naturalmente uniforme
            # porque son superficies planas con impresión consistente.
            # Solo es fuerte indicador en combinación con otros análisis.
            ela_uniformity = ela_analysis.get('uniformity', 0.0)
            
            # Solo activar ELA si hay señales corroborativas
            has_face_signal = face_analysis.get('suspicious', False)
            has_other_signal = (histogram_is_suspicious or 
                               frequency_analysis.get('suspicious', False) or
                               color_analysis.get('suspicious', False))
            
            # NOTA CLAVE: Documentos impresos/escaneados (pasaportes) tienen ELA
            # naturalmente muy uniforme (0.85-0.95) porque son superficies planas.
            # ELA solo NO es indicador confiable para documentos. Requiere corroboración.
            if ela_uniformity > 0.92 and has_other_signal:
                # Extremadamente uniforme + otra señal fuerte = probable IA
                suspicion_score += 0.30
                results['red_flags'].append('ELA: niveles de error extremadamente uniformes + anomalías corroborativas')
            elif ela_uniformity > 0.92:
                # Solo ELA alto, sin corroboración = puede ser documento real escaneado
                suspicion_score += 0.05
            elif ela_uniformity > 0.88 and has_other_signal:
                suspicion_score += 0.20
                results['red_flags'].append('ELA: niveles de error muy uniformes + otras anomalías')
            # Por debajo de 0.92 sin corroboración = normal para documentos impresos
            
            # --- Textura facial ---
            # NOTA: Fotos de pasaporte son tomadas en estudio con iluminación uniforme,
            # lo que produce rostros naturalmente suaves. Solo considerar sospechoso
            # con smoothness alto (>0.70) ya que fotos reales de estudio llegan a 0.55-0.65.
            face_suspicious = face_analysis.get('suspicious', False)
            face_smoothness = face_analysis.get('smoothness_score', 0.0)
            if face_suspicious:
                if face_smoothness > 0.85:
                    suspicion_score += 0.35
                    results['red_flags'].append('Rostro con textura artificial - probable IA')
                elif face_smoothness > 0.70:
                    suspicion_score += 0.15
                    results['red_flags'].append('Rostro demasiado suave - posible IA')
                # smoothness 0.55-0.70 = normal para fotos de estudio, NO suma
            
            # --- Frecuencia ---
            freq_suspicious = frequency_analysis.get('suspicious', False)
            if freq_suspicious:
                suspicion_score += 0.20
                results['red_flags'].append('Patrones de frecuencia anómalos - posible generación IA')
            
            # --- Color ---
            color_suspicious = color_analysis.get('suspicious', False)
            if color_suspicious:
                suspicion_score += 0.15
                results['red_flags'].append('Correlación de canales de color anómala')
            
            # --- Chunks / Metadata binaria ---
            chunk_markers = chunk_analysis.get('ai_markers', [])
            if chunk_markers:
                # Evidencia directa en metadata = señal muy fuerte
                marker_score = min(0.60, len(chunk_markers) * 0.30)
                suspicion_score += marker_score
                for marker in chunk_markers:
                    results['red_flags'].append(f'Metadata: {marker}')
            
            # ====== DECISIÓN FINAL ======
            # Umbral continuo en vez de contar banderas
            if suspicion_score >= 0.65:
                results['is_ai_generated'] = True
                results['confidence'] = min(0.95, suspicion_score)
                results['detected_method'] = 'AI_GENERATED'
            elif suspicion_score >= 0.50:
                results['is_ai_generated'] = True
                results['confidence'] = suspicion_score
                results['detected_method'] = 'AI_GENERATED_LIKELY'
            elif suspicion_score >= 0.40:
                results['is_edited'] = True
                results['confidence'] = suspicion_score
                results['detected_method'] = 'POSSIBLY_EDITED'
            else:
                results['confidence'] = suspicion_score
                results['detected_method'] = 'NATURAL'
            
            results['details']['suspicion_score'] = round(suspicion_score, 3)
        
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
                'suspicious': False,  # Decisión se toma en detect_from_image con scoring
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
            
            # Clasificación de rangos:
            # < 8: demasiado limpio (artificial)
            # 50-350: IA generativa típica
            # 350-800: zona gris
            # 800-3500: foto real de celular (normal)
            # > 4500: muy ruidoso (artificial)
            if laplacian_var < 8:
                interpretation = 'muy_limpio'
            elif laplacian_var < 350:
                interpretation = 'sospechoso_ia'
            elif laplacian_var < 800:
                interpretation = 'zona_gris'
            elif laplacian_var <= 4500:
                interpretation = 'normal'
            else:
                interpretation = 'muy_ruidoso'
            
            return {
                'unnatural': False,  # Decisión se toma en detect_from_image con scoring
                'laplacian_variance': float(laplacian_var),
                'interpretation': interpretation
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
    
    def _analyze_ela(self, image: np.ndarray) -> Dict:
        """
        Error Level Analysis (ELA).
        
        Re-comprime la imagen a un nivel JPEG conocido y compara con el original.
        Imágenes reales muestran niveles de error variados (bordes vs áreas planas).
        Imágenes IA muestran niveles de error uniformes en toda la imagen.
        """
        try:
            # Convertir BGR a RGB para PIL
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            
            # Re-comprimir a JPEG calidad 90
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=90)
            buffer.seek(0)
            recompressed = Image.open(buffer)
            
            # Calcular diferencia (ELA)
            original_arr = np.array(pil_image, dtype=np.float64)
            recompressed_arr = np.array(recompressed, dtype=np.float64)
            ela_diff = np.abs(original_arr - recompressed_arr)
            
            # Amplificar diferencias para análisis
            ela_amplified = ela_diff * 10.0
            ela_gray = np.mean(ela_amplified, axis=2) if len(ela_amplified.shape) == 3 else ela_amplified
            
            # Dividir en bloques y medir varianza entre ellos
            h, w = ela_gray.shape
            block_size = max(h // 8, w // 8, 16)
            block_means = []
            for y in range(0, h - block_size + 1, block_size):
                for x in range(0, w - block_size + 1, block_size):
                    block = ela_gray[y:y+block_size, x:x+block_size]
                    block_means.append(np.mean(block))
            
            if len(block_means) < 4:
                return {'suspicious': False, 'uniformity': 0.0}
            
            block_means = np.array(block_means)
            mean_ela = np.mean(block_means)
            std_ela = np.std(block_means)
            
            # Coeficiente de variación: bajo = uniforme = IA
            # Fotos reales: CV > 0.4 (variación alta entre bloques)
            # IA generada: CV < 0.25 (todos los bloques similares)
            cv = std_ela / (mean_ela + 1e-10)
            uniformity = max(0.0, 1.0 - cv)
            
            suspicious = uniformity > 0.65
            
            return {
                'suspicious': suspicious,
                'uniformity': float(uniformity),
                'mean_ela': float(mean_ela),
                'std_ela': float(std_ela),
                'coefficient_of_variation': float(cv),
            }
        except Exception as e:
            self.logger.debug(f"Error en ELA: {str(e)}")
            return {'suspicious': False, 'uniformity': 0.0, 'error': str(e)}
    
    def _analyze_face_texture(self, image: np.ndarray) -> Dict:
        """
        Analizar textura de la región facial.
        
        Caras generadas por IA tienden a tener:
        - Piel demasiado suave/uniforme
        - Falta de poros y micro-texturas
        - Gradientes de color demasiado perfectos
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Detectar cara con Haar Cascade
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
            
            if len(faces) == 0:
                return {'suspicious': False, 'face_detected': False}
            
            # Tomar la cara más grande
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_roi = gray[y:y+h, x:x+w]
            
            # Región central de la cara (mejillas/frente - donde se nota la IA)
            margin_x = w // 4
            margin_y = h // 4
            face_center = gray[y+margin_y:y+h-margin_y, x+margin_x:x+w-margin_x]
            
            if face_center.size == 0:
                return {'suspicious': False, 'face_detected': True}
            
            # 1. Laplacian de la cara (micro-textura)
            face_lap = cv2.Laplacian(face_center, cv2.CV_64F)
            face_lap_var = np.var(face_lap)
            
            # 2. Análisis de bordes finos (Canny con umbrales altos)
            edges = cv2.Canny(face_center, 100, 200)
            edge_density = np.sum(edges > 0) / edges.size
            
            # 3. Gradiente local (detecta suavizado excesivo)
            sobelx = cv2.Sobel(face_center, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(face_center, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(sobelx**2 + sobely**2)
            gradient_mean = np.mean(gradient_mag)
            gradient_std = np.std(gradient_mag)
            
            # Score de suavidad: combina múltiples indicadores
            # Caras reales (foto de pasaporte): laplacian 50-300, edge_density 0.03-0.15
            # Caras IA: laplacian < 30, edge_density < 0.02, gradiente uniforme
            smoothness = 0.0
            
            if face_lap_var < 20:
                smoothness += 0.40  # Muy suave
            elif face_lap_var < 50:
                smoothness += 0.25  # Suave
            elif face_lap_var < 100:
                smoothness += 0.10
            
            if edge_density < 0.01:
                smoothness += 0.30  # Casi sin bordes finos
            elif edge_density < 0.03:
                smoothness += 0.15
            
            # Gradiente demasiado uniforme (std bajo relativo a mean)
            grad_cv = gradient_std / (gradient_mean + 1e-10)
            if grad_cv < 0.8:
                smoothness += 0.20  # Gradientes muy uniformes
            elif grad_cv < 1.2:
                smoothness += 0.10
            
            suspicious = smoothness > 0.55
            
            return {
                'suspicious': suspicious,
                'face_detected': True,
                'smoothness_score': float(smoothness),
                'face_laplacian': float(face_lap_var),
                'edge_density': float(edge_density),
                'gradient_mean': float(gradient_mean),
                'gradient_cv': float(grad_cv),
            }
        except Exception as e:
            self.logger.debug(f"Error en análisis facial: {str(e)}")
            return {'suspicious': False, 'face_detected': False, 'error': str(e)}
    
    def _analyze_frequency(self, image: np.ndarray) -> Dict:
        """
        Análisis de dominio de frecuencia.
        
        Generadores IA (GANs, Diffusion) dejan artefactos en el espectro de frecuencia:
        - Picos en frecuencias específicas
        - Espectro demasiado suave (falta de ruido de alta frecuencia natural)
        - Patrones periódicos no naturales
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Redimensionar a tamaño estándar para comparación consistente
            resized = cv2.resize(gray, (256, 256))
            
            # FFT 2D
            f_transform = np.fft.fft2(resized.astype(np.float64))
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.log1p(np.abs(f_shift))
            
            # Análisis radial del espectro
            center = (128, 128)
            Y, X = np.ogrid[:256, :256]
            r = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
            
            # Dividir en bandas de frecuencia
            low_freq = magnitude[r < 30].mean()
            mid_freq = magnitude[(r >= 30) & (r < 80)].mean()
            high_freq = magnitude[r >= 80].mean()
            
            # Ratio alta/media frecuencia - Fotos reales tienen más alta frecuencia
            # (ruido del sensor, textura de impresión, etc.)
            # IA: ratio bajo (falta detalle de alta frecuencia)
            hm_ratio = high_freq / (mid_freq + 1e-10)
            lm_ratio = low_freq / (mid_freq + 1e-10)
            
            # Azimuthal analysis: desviación angular del espectro
            # IA tiende a tener espectro más isótropo (uniforme en todas direcciones)
            angles = np.arctan2(Y - center[1], X - center[0])
            angular_bins = 12
            angular_means = []
            for i in range(angular_bins):
                a_start = -np.pi + i * (2 * np.pi / angular_bins)
                a_end = a_start + (2 * np.pi / angular_bins)
                mask = (angles >= a_start) & (angles < a_end) & (r > 30) & (r < 100)
                if np.any(mask):
                    angular_means.append(float(magnitude[mask].mean()))
            
            angular_std = np.std(angular_means) if angular_means else 0.0
            angular_mean = np.mean(angular_means) if angular_means else 1.0
            angular_cv = angular_std / (angular_mean + 1e-10)
            
            # IA: hm_ratio < 0.65 Y angular_cv < 0.05
            # Real: hm_ratio > 0.70 O angular_cv > 0.06
            suspicious = (hm_ratio < 0.60 and angular_cv < 0.04)
            
            return {
                'suspicious': suspicious,
                'high_mid_ratio': float(hm_ratio),
                'low_mid_ratio': float(lm_ratio),
                'angular_cv': float(angular_cv),
                'low_freq': float(low_freq),
                'mid_freq': float(mid_freq),
                'high_freq': float(high_freq),
            }
        except Exception as e:
            self.logger.debug(f"Error en análisis de frecuencia: {str(e)}")
            return {'suspicious': False, 'error': str(e)}
    
    def _analyze_color_channels(self, image: np.ndarray) -> Dict:
        """
        Analizar correlación entre canales de color.
        
        Imágenes IA tienen patrones de color anómalos:
        - Correlación entre canales demasiado alta o baja
        - Distribución de saturación poco natural
        - Anomalías en el espacio de color HSV
        """
        try:
            if len(image.shape) != 3 or image.shape[2] != 3:
                return {'suspicious': False}
            
            b, g, r = cv2.split(image)
            
            # 1. Correlación entre canales
            # Flatten
            r_flat = r.flatten().astype(np.float64)
            g_flat = g.flatten().astype(np.float64)
            b_flat = b.flatten().astype(np.float64)
            
            # Correlación de Pearson
            rg_corr = np.corrcoef(r_flat, g_flat)[0, 1]
            rb_corr = np.corrcoef(r_flat, b_flat)[0, 1]
            gb_corr = np.corrcoef(g_flat, b_flat)[0, 1]
            
            # IA tiende a tener correlaciones MUY altas (> 0.97) entre todos los canales
            mean_corr = (abs(rg_corr) + abs(rb_corr) + abs(gb_corr)) / 3
            
            # 2. Análisis de saturación en HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1].flatten().astype(np.float64)
            sat_std = np.std(saturation)
            sat_mean = np.mean(saturation)
            
            # IA: saturación muy uniforme (std bajo) o media baja
            sat_cv = sat_std / (sat_mean + 1e-10)
            
            # 3. Análisis de rango dinámico por canal
            ranges = []
            for ch in [r, g, b]:
                p5 = np.percentile(ch, 5)
                p95 = np.percentile(ch, 95)
                ranges.append(p95 - p5)
            range_std = np.std(ranges)
            
            # Sospechoso: correlación muy alta + saturación uniforme + rangos similares
            suspicious = (mean_corr > 0.97 and sat_cv < 0.5 and range_std < 15)
            
            return {
                'suspicious': suspicious,
                'mean_correlation': float(mean_corr),
                'rg_corr': float(rg_corr),
                'rb_corr': float(rb_corr),
                'gb_corr': float(gb_corr),
                'saturation_cv': float(sat_cv),
                'range_std': float(range_std),
            }
        except Exception as e:
            self.logger.debug(f"Error en análisis de color: {str(e)}")
            return {'suspicious': False, 'error': str(e)}
    
    def _analyze_chunks(self, raw_bytes: bytes) -> Dict:
        """
        Analizar chunks/metadata binaria del archivo de imagen.
        
        Herramientas de IA dejan rastros en:
        - PNG tEXt/iTXt chunks: "parameters", "Dream", "Stable Diffusion", "ComfyUI"
        - JPEG APP markers: XMP con "ai", "generated", "midjourney"
        - JPEG COM markers: comentarios con herramientas IA
        - EXIF UserComment, ImageDescription
        - PNG Software chunk
        - WebP metadata
        """
        result = {
            'suspicious': False,
            'ai_markers': [],
            'metadata_found': [],
            'format': 'unknown',
        }
        
        if not raw_bytes or len(raw_bytes) < 8:
            return result
        
        try:
            # Detectar formato
            if raw_bytes[:8] == b'\x89PNG\r\n\x1a\n':
                result['format'] = 'PNG'
                self._analyze_png_chunks(raw_bytes, result)
            elif raw_bytes[:2] == b'\xff\xd8':
                result['format'] = 'JPEG'
                self._analyze_jpeg_chunks(raw_bytes, result)
            elif raw_bytes[:4] == b'RIFF' and raw_bytes[8:12] == b'WEBP':
                result['format'] = 'WEBP'
                self._analyze_webp_chunks(raw_bytes, result)
            
            # Búsqueda genérica de strings IA en todo el archivo
            self._search_ai_strings(raw_bytes, result)
            
            result['suspicious'] = len(result['ai_markers']) > 0
            
        except Exception as e:
            self.logger.debug(f"Error en análisis de chunks: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def _analyze_png_chunks(self, raw_bytes: bytes, result: Dict):
        """Leer chunks PNG y buscar metadata de IA"""
        try:
            offset = 8  # Skip PNG signature
            while offset < len(raw_bytes) - 12:
                # Cada chunk: 4 bytes length + 4 bytes type + data + 4 bytes CRC
                chunk_len = struct.unpack('>I', raw_bytes[offset:offset+4])[0]
                chunk_type = raw_bytes[offset+4:offset+8].decode('ascii', errors='replace')
                chunk_data_start = offset + 8
                chunk_data_end = chunk_data_start + chunk_len
                
                if chunk_data_end > len(raw_bytes):
                    break
                
                chunk_data = raw_bytes[chunk_data_start:chunk_data_end]
                
                # Chunks de texto: tEXt, iTXt, zTXt
                if chunk_type in ('tEXt', 'iTXt', 'zTXt'):
                    try:
                        text_content = chunk_data.decode('utf-8', errors='replace').lower()
                        result['metadata_found'].append({
                            'type': chunk_type, 
                            'preview': text_content[:200]
                        })
                        self._check_ai_text(text_content, chunk_type, result)
                    except Exception:
                        pass
                
                offset = chunk_data_end + 4  # Skip CRC
                
                if chunk_type == 'IEND':
                    break
                    
        except Exception as e:
            self.logger.debug(f"Error leyendo PNG chunks: {e}")
    
    def _analyze_jpeg_chunks(self, raw_bytes: bytes, result: Dict):
        """Leer markers JPEG y buscar metadata de IA"""
        try:
            offset = 2  # Skip SOI (FF D8)
            while offset < len(raw_bytes) - 4:
                if raw_bytes[offset] != 0xFF:
                    offset += 1
                    continue
                
                marker = raw_bytes[offset + 1]
                
                # SOS (Start of Scan) = fin de headers
                if marker == 0xDA:
                    break
                
                # Markers sin payload
                if marker in (0xD8, 0xD9, 0x00):
                    offset += 2
                    continue
                
                # Leer longitud del segmento
                if offset + 4 > len(raw_bytes):
                    break
                seg_len = struct.unpack('>H', raw_bytes[offset+2:offset+4])[0]
                seg_data_start = offset + 4
                seg_data_end = offset + 2 + seg_len
                
                if seg_data_end > len(raw_bytes):
                    break
                
                seg_data = raw_bytes[seg_data_start:seg_data_end]
                
                # APP0-APP15 markers (0xE0-0xEF)
                if 0xE0 <= marker <= 0xEF:
                    try:
                        text_content = seg_data.decode('utf-8', errors='replace').lower()
                        marker_name = f'APP{marker - 0xE0}'
                        
                        # APP1 = EXIF/XMP, APP13 = IPTC
                        if len(text_content) > 10:
                            result['metadata_found'].append({
                                'type': marker_name,
                                'preview': text_content[:200]
                            })
                        self._check_ai_text(text_content, marker_name, result)
                    except Exception:
                        pass
                
                # COM marker (0xFE) - Comment
                elif marker == 0xFE:
                    try:
                        comment = seg_data.decode('utf-8', errors='replace').lower()
                        result['metadata_found'].append({
                            'type': 'COM',
                            'preview': comment[:200]
                        })
                        self._check_ai_text(comment, 'COM', result)
                    except Exception:
                        pass
                
                offset = seg_data_end
                
        except Exception as e:
            self.logger.debug(f"Error leyendo JPEG markers: {e}")
    
    def _analyze_webp_chunks(self, raw_bytes: bytes, result: Dict):
        """Leer chunks WebP y buscar metadata de IA"""
        try:
            offset = 12  # Skip RIFF header + WEBP
            file_size = min(struct.unpack('<I', raw_bytes[4:8])[0] + 8, len(raw_bytes))
            
            while offset < file_size - 8:
                chunk_id = raw_bytes[offset:offset+4].decode('ascii', errors='replace')
                chunk_size = struct.unpack('<I', raw_bytes[offset+4:offset+8])[0]
                chunk_data = raw_bytes[offset+8:offset+8+chunk_size]
                
                if chunk_id in ('EXIF', 'XMP '):
                    try:
                        text = chunk_data.decode('utf-8', errors='replace').lower()
                        result['metadata_found'].append({
                            'type': f'WEBP_{chunk_id.strip()}',
                            'preview': text[:200]
                        })
                        self._check_ai_text(text, f'WEBP_{chunk_id}', result)
                    except Exception:
                        pass
                
                # Align to even byte
                offset += 8 + chunk_size + (chunk_size % 2)
                
        except Exception as e:
            self.logger.debug(f"Error leyendo WebP chunks: {e}")
    
    def _check_ai_text(self, text: str, source: str, result: Dict):
        """Buscar indicadores de IA en texto de metadata"""
        text_lower = text.lower()
        
        ai_indicators = {
            'stable diffusion': 'Stable Diffusion detectado',
            'comfyui': 'ComfyUI (generador IA) detectado',
            'automatic1111': 'Automatic1111 (Stable Diffusion UI) detectado',
            'midjourney': 'Midjourney detectado',
            'dall-e': 'DALL-E detectado',
            'dall·e': 'DALL-E detectado',
            'openai': 'Herramienta OpenAI detectada',
            'chatgpt': 'ChatGPT detectado',
            'ai_generated': 'Marcador ai_generated encontrado',
            'ai generated': 'Marcador AI generated encontrado',
            'dream\x00': 'DreamStudio/Stable Diffusion detectado',
            'invoke-ai': 'InvokeAI detectado',
            'novelai': 'NovelAI detectado',
            'nai diffusion': 'NAI Diffusion detectado',
            'flux': 'FLUX (modelo IA) detectado',
            'kandinsky': 'Kandinsky (modelo IA) detectado',
            'deepfloyd': 'DeepFloyd IF detectado',
            'imagen': 'Google Imagen detectado',
            'firefly': 'Adobe Firefly detectado',
            'bing image creator': 'Bing Image Creator detectado',
            'playground ai': 'Playground AI detectado',
            'leonardo.ai': 'Leonardo AI detectado',
            'generated by': 'Marcador "generated by" encontrado',
            'made with': 'Marcador "made with" en metadata',
            'samplereuler': 'Sampler de difusión (Euler) detectado',
            'samplerdpm': 'Sampler de difusión (DPM) detectado',
            'cfg scale': 'Parámetro CFG Scale (difusión) detectado',
            'negative prompt': 'Prompt negativo (difusión) detectado',
            'lora:': 'LoRA adaptador detectado',
            'controlnet': 'ControlNet detectado',
        }
        
        for pattern, description in ai_indicators.items():
            if pattern in text_lower:
                marker = f'{description} [en {source}]'
                if marker not in result['ai_markers']:
                    result['ai_markers'].append(marker)
    
    def _search_ai_strings(self, raw_bytes: bytes, result: Dict):
        """Búsqueda genérica de strings IA en los bytes crudos del archivo"""
        try:
            # Buscar solo en primeros 64KB (headers/metadata) para eficiencia
            search_region = raw_bytes[:65536]
            text = search_region.decode('utf-8', errors='replace').lower()
            
            # Patrones muy específicos que NO aparecen en fotos normales
            critical_patterns = {
                'parameters\x00': 'Chunk "parameters" de Stable Diffusion A1111',
                'prompt\x00': 'Chunk "prompt" de generador IA',
                'aesthetic_score': 'Parámetro aesthetic_score de SDXL',
                'score_9': 'Parámetro score_9 de Pony Diffusion',
                'steps:': 'Parámetro "steps" de difusión',
                'sampler:': 'Parámetro "sampler" de difusión',
                'model hash': 'Hash de modelo de difusión',
                'clip skip': 'Parámetro CLIP skip de difusión',
            }
            
            for pattern, description in critical_patterns.items():
                if pattern in text:
                    marker = f'{description} [binario]'
                    if marker not in result['ai_markers']:
                        result['ai_markers'].append(marker)
        except Exception:
            pass

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
