"""
authenticity_validator.py - Validador completo de autenticidad

Combina múltiples técnicas:
1. Scoring OCR/MRZ (confidence_scorer)
2. Detección de IA/Edición (ai_detection)
3. Análisis de metadatos
4. Validaciones adicionales de pasaporte

Resultado final: PASS/REVIEW/REJECT + razón específica
"""

import logging
import numpy as np
from typing import Dict, Tuple
from datetime import datetime

from .ai_detection import AIDetector

logger = logging.getLogger(__name__)


class AuthenticityValidator:
    """Validador completo de autenticidad de pasaportes"""
    
    def __init__(self):
        """Inicializar validador"""
        self.ai_detector = AIDetector()
    
    def validate(
        self, 
        image_array: np.ndarray,
        ocr_score: float,
        mrz_score: float,
        ocr_confidence: float,
        mrz_result: Dict,
        original_confidence: float
    ) -> Dict:
        """
        Validación completa de autenticidad
        
        Args:
            image_array: Imagen preprocesada (BGR)
            ocr_score: Score de OCR (0-1)
            mrz_score: Score de MRZ (0-1)
            ocr_confidence: Confianza OCR (0-1)
            mrz_result: Resultado de validación MRZ
            original_confidence: Score de confianza original (%)
            
        Returns:
            dict: Resultado final con estado y justificación
        """
        
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'autenticidad_score': 0.0,
            'estado': 'REJECT',
            'razon': '',
            'metodo_deteccion': '',
            'confianza': {
                'ocr': ocr_confidence,
                'mrz': mrz_score,
                'ia': 0.0,
                'final': original_confidence / 100.0
            },
            'analisis': {
                'ocr_score': float(ocr_score),
                'mrz_score': float(mrz_score),
                'ai_analysis': {},
                'metadata_warnings': []
            }
        }
        
        try:
            # 1. ANÁLISIS DE IA/EDICIÓN
            ai_analysis = self.ai_detector.detect_from_image(image_array)
            results['analisis']['ai_analysis'] = ai_analysis
            
            # 2. DETERMINAR SI FUE POR IA
            if ai_analysis.get('is_ai_generated'):
                results['estado'] = 'REJECT'
                results['razon'] = f'❌ FALSIFICADO POR IA - Confianza: {ai_analysis.get("confidence", 0):.0%}'
                results['metodo_deteccion'] = 'AI_GENERATED'
                results['autenticidad_score'] = 5  # Score muy bajo
                return results
            
            # 3. ANÁLISIS COMBINADO DE SCORES
            final_score = (ocr_score * 0.40) + (mrz_score * 0.60)
            results['autenticidad_score'] = final_score * 100

            ai_confidence = ai_analysis.get('confidence', 0.0)
            is_strong_edit_signal = ai_analysis.get('is_edited') and ai_confidence >= 0.75
            is_weak_edit_signal = ai_analysis.get('is_edited') and ai_confidence < 0.75

            # Regla de estabilización:
            # Si MRZ es consistente pero OCR general es bajo, evitar rechazo automático.
            if (mrz_score >= 0.65 and ocr_score < 0.40 and
                    not ai_analysis.get('is_ai_generated') and not is_strong_edit_signal):
                results['estado'] = 'REVIEW'
                results['razon'] = (
                    f'⚠️ REVISAR - MRZ consistente ({mrz_score:.0%}) '
                    f'pero OCR general bajo ({ocr_score:.0%}).'
                )
                results['metodo_deteccion'] = 'MRZ_STRONG_OCR_WEAK'
                results['confianza']['ia'] = ai_confidence
                return results
            
            # 4. DECISIÓN BASADA EN MÚLTIPLES FACTORES
            if is_strong_edit_signal and final_score < 0.8:
                results['estado'] = 'REJECT'
                results['razon'] = f'❌ FALSIFICADO - Imagen editada + scores bajos (OCR: {ocr_score:.0%}, MRZ: {mrz_score:.0%})'
                results['metodo_deteccion'] = 'EDITED_WITH_LOW_SCORES'
            
            elif is_strong_edit_signal and 0.8 <= final_score < 0.9:
                results['estado'] = 'REVIEW'
                results['razon'] = f'⚠️ REVISAR - Signos de edición detectados. Score: {final_score:.0%}'
                results['metodo_deteccion'] = 'POSSIBLY_EDITED'

            elif is_weak_edit_signal and final_score < 0.7:
                results['estado'] = 'REVIEW'
                results['razon'] = (
                    f'⚠️ REVISAR - Señales leves de edición (conf {ai_confidence:.0%}) '
                    f'y score bajo ({final_score:.0%}).'
                )
                results['metodo_deteccion'] = 'WEAK_EDIT_SIGNAL'
            
            elif final_score >= 0.90:
                results['estado'] = 'PASS'
                results['razon'] = f'✓ AUTÉNTICO - Score de confianza: {final_score:.0%}'
                results['metodo_deteccion'] = 'AUTHENTIC'
            
            elif final_score >= 0.70:
                results['estado'] = 'REVIEW'
                results['razon'] = f'⚠️ REVISAR - Score intermedio: {final_score:.0%}'
                results['metodo_deteccion'] = 'UNCERTAIN'
            
            else:
                results['estado'] = 'REJECT'
                results['razon'] = f'❌ FALSIFICADO - Score bajo: {final_score:.0%}'
                results['metodo_deteccion'] = 'LOW_SCORE'
            
            # 5. AGREGAR ADVERTENCIAS DE METADATOS
            for red_flag in ai_analysis.get('red_flags', []):
                results['analisis']['metadata_warnings'].append(red_flag)
            
            # 6. AGREGAR ANOMALÍAS TÉCNICAS
            if ai_analysis.get('details', {}).get('histogram', {}).get('suspicious'):
                results['analisis']['metadata_warnings'].append('Anomalía en histograma')
            
            if ai_analysis.get('details', {}).get('noise', {}).get('unnatural'):
                results['analisis']['metadata_warnings'].append('Patrón de ruido artificial')
            
            if ai_analysis.get('details', {}).get('lighting', {}).get('inconsistent'):
                results['analisis']['metadata_warnings'].append('Iluminación inconsistente')
            
            # 7. VALIDACIONES ADICIONALES DE PASAPORTE MEXICANO
            if mrz_result and not mrz_result.get('valid'):
                results['analisis']['metadata_warnings'].append('MRZ inválido o mal leído')
            
            # Confianza en detección de IA
            results['confianza']['ia'] = ai_analysis.get('confidence', 0)
        
        except Exception as e:
            logger.error(f"Error en validación de autenticidad: {str(e)}")
            results['error'] = str(e)
            results['estado'] = 'REJECT'
            results['razon'] = f'❌ Error en procesamiento: {str(e)}'
        
        return results
    
    def get_recommendation(self, validation_result: Dict) -> str:
        """Generar recomendación textual"""
        estado = validation_result.get('estado', 'UNKNOWN')
        razon = validation_result.get('razon', '')
        
        if estado == 'PASS':
            return f"✓ Pasaporte auténtico. {razon}"
        
        elif estado == 'REVIEW':
            if 'IA' in razon or 'editada' in razon.lower():
                return f"⚠️ Posibles signos de falsificación detectados. {razon} Requiere verificación manual por experto."
            else:
                return f"⚠️ Confianza media. {razon} Se recomienda verificación manual."
        
        elif estado == 'REJECT':
            if 'IA' in razon:
                return f"❌ ALERTA: Pasaporte probablemente generado por IA. {razon} Rechazado."
            elif 'editada' in razon.lower():
                return f"❌ Evidencia de edición/falsificación detectada. {razon} Rechazado."
            else:
                return f"❌ Pasaporte rechazado. {razon}"
        
        else:
            return "❓ Estado desconocido. Contactar con administrador."


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    validator = AuthenticityValidator()
    print("✓ Validador de autenticidad inicializado")
