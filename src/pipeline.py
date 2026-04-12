"""
pipeline.py - Pipeline Principal de Verificación de Pasaportes

Orquesta el flujo completo de verificación:
1. Preprocesamiento de imagen
2. OCR con TrOCR
3. Validación de MRZ
4. Cálculo de score de confianza
5. Decisión de autenticidad

Versión: 0.1.0
"""

import logging
import uuid
import re
from datetime import datetime
from typing import Union, Dict, Tuple, Optional
from pathlib import Path

# Importar módulos del proyecto
from . import preprocessing
from . import ocr_engine
from . import mrz_validator
from . import mrz_roi_detector
from . import confidence_scorer
from .authenticity_validator import AuthenticityValidator

import config

logger = logging.getLogger(__name__)


# ============================================================================
# CLASE PRINCIPAL - VERIFICADOR DE PASAPORTES
# ============================================================================

class PassportVerifier:
    """
    Verificador de autenticidad de pasaportes.
    
    Realiza el pipeline completo de verificación usando
    preprocesamiento, OCR y validación de MRZ.
    """
    
    def __init__(self):
        """Inicializar verificador."""
        self.session_id = str(uuid.uuid4())[:8]
        self.logger = logging.getLogger(f"PassportVerifier-{self.session_id}")
        self.logger.info(f"Verificador inicializado: {self.session_id}")
    
    def _update_analysis_status(self, step: str, progress: int, total: int):
        """Reportar progreso del análisis al endpoint /model-status."""
        try:
            import app as _app
            _app._model_status = {"status": "analyzing", "step": step, "progress": progress, "total": total}
        except Exception:
            pass
    
    def verify(self, image_input: Union[str, Path, bytes],
              verbose: bool = False) -> Dict:
        """
        Verificar autenticidad de un pasaporte.
        
        Args:
            image_input: Imagen del pasaporte en múltiples formatos
            verbose: Si mostrar logs detallados
            
        Returns:
            dict: Resultado completo de verificación
        """
        start_time = datetime.utcnow()
        
        result = {
            "id": str(uuid.uuid4()),
            "session_id": self.session_id,
            "timestamp": start_time.isoformat(),
            "image_input_type": type(image_input).__name__,
            "processing_stages": {},
            "final_result": None,
            "error": None,
            "processing_time_ms": 0,
        }

        highres_document_image = None
        original_full_image = None
        
        try:
            self.logger.info("=" * 60)
            self.logger.info("INICIANDO VERIFICACIÓN DE PASAPORTE")
            self.logger.info("=" * 60)
            
            # ====== ETAPA 1: PREPROCESAMIENTO ======
            self.logger.info("Etapa 1/4: Preprocesamiento de imagen")
            self._update_analysis_status("Preprocesando imagen...", 1, 4)
            
            try:
                processed_image, preprocessing_metadata = preprocessing.preprocess_pipeline(
                    image_input,
                    validate=True
                )

                # Preparar versión de alta resolución para MRZ dedicado
                try:
                    original_image = preprocessing.load_image(image_input)
                    original_full_image = original_image.copy()  # Guardar copia original para detección IA
                    highres_roi = preprocessing.detect_passport_roi(original_image)
                    highres_document_image = preprocessing.crop_roi(original_image, highres_roi)
                    highres_document_image = preprocessing.ensure_landscape_orientation(highres_document_image)
                    highres_document_image = preprocessing.correct_document_perspective(highres_document_image)
                except Exception as prep_e:
                    self.logger.debug(f"No se pudo preparar alta resolución MRZ: {prep_e}")
                    highres_document_image = None
                
                result["processing_stages"]["preprocessing"] = {
                    "status": "success",
                    "image_shape": processed_image.shape,
                    "is_valid": preprocessing_metadata.get("is_valid"),
                    "quality_message": preprocessing_metadata.get("quality_message"),
                    "perspective_corrected": preprocessing_metadata.get("perspective_corrected", False),
                }
                
                self.logger.info(f"✓ Preprocesamiento completado")
                
            except Exception as e:
                self.logger.error(f"✗ Error en preprocesamiento: {str(e)}")
                result["processing_stages"]["preprocessing"] = {
                    "status": "error",
                    "error": str(e),
                }
                raise
            
            # ====== ETAPA 2: OCR ======
            self.logger.info("Etapa 2/4: Extracción de texto (OCR)")
            self._update_analysis_status("Extrayendo texto (OCR)...", 2, 4)
            
            try:
                # OCR general robusto (orientación + variantes)
                ocr_result = ocr_engine.extract_text_with_confidence_robust(processed_image)

                # OCR dedicado en ROI MRZ (preferentemente en alta resolución)
                mrz_source_image = highres_document_image if highres_document_image is not None else processed_image
                mrz_roi = mrz_roi_detector.find_mrz_region(mrz_source_image)
                mrz_ocr_result = {
                    "full_text": "",
                    "mrz_lines": [],
                    "mrz_detected": False,
                    "ocr_avg_confidence": 0.0,
                }

                if mrz_roi is not None:
                    x, y, w, h = mrz_roi
                    mrz_crop = mrz_source_image[y:y+h, x:x+w]
                    mrz_ocr_result = ocr_engine.extract_mrz_text_from_roi(mrz_crop)
                    mrz_ocr_result["roi"] = mrz_roi
                
                result["processing_stages"]["ocr"] = {
                    "status": "success",
                    "text_length": len(ocr_result.get("full_text", "")),
                    "tokens": len(ocr_result.get("tokens", [])),
                    "mrz_detected": ocr_result.get("mrz_detected"),
                    "ocr_confidence": ocr_result.get("ocr_avg_confidence"),
                    "orientation": ocr_result.get("orientation"),
                    "variant": ocr_result.get("preprocess_variant"),
                    "mrz_roi_detected": mrz_roi is not None,
                    "mrz_source": "highres" if highres_document_image is not None else "processed",
                    "mrz_ocr_confidence": mrz_ocr_result.get("ocr_avg_confidence", 0.0),
                    "mrz_lines": len(mrz_ocr_result.get("mrz_lines", [])),
                }
                
                self.logger.info(f"✓ OCR completado: {len(ocr_result.get('tokens', []))} tokens, "
                                f"confianza {ocr_result.get('ocr_avg_confidence', 0):.1%}")
                
            except Exception as e:
                self.logger.error(f"✗ Error en OCR: {str(e)}")
                result["processing_stages"]["ocr"] = {
                    "status": "error",
                    "error": str(e),
                }
                raise
            
            # ====== ETAPA 3: VALIDACIÓN DE DATOS (MRZ o Mexicano) ======
            self.logger.info("Etapa 3/4: Validación de datos de pasaporte")
            self._update_analysis_status("Validando MRZ...", 3, 4)
            
            try:
                full_text = ocr_result.get("full_text", "")
                mrz_text = mrz_ocr_result.get("full_text", "") if "mrz_ocr_result" in locals() else ""

                combined_text = full_text
                if mrz_text:
                    combined_text = f"{full_text}\n{mrz_text}" if full_text else mrz_text

                mrz_ocr_confidence = max(
                    float(ocr_result.get("ocr_avg_confidence", 0)),
                    float(mrz_ocr_result.get("ocr_avg_confidence", 0)) if "mrz_ocr_result" in locals() else 0.0
                )
                
                # Extraer líneas o números de pasaporte (flexible para ambos formatos)
                mrz_lines = mrz_validator.extract_mrz_lines(combined_text)

                # Si OCR dedicado MRZ tiene líneas válidas, priorizarlas
                dedicated_mrz_lines = []
                if "mrz_ocr_result" in locals():
                    dedicated_mrz_lines = mrz_ocr_result.get("mrz_lines", []) or []
                if dedicated_mrz_lines:
                    mrz_lines = dedicated_mrz_lines + mrz_lines

                has_passport_pattern = any(
                    re.search(r'[A-Z]\d{7,10}', str(line).upper().replace(' ', ''))
                    for line in mrz_lines
                )

                passport_fallback = {
                    "passport_number": None,
                    "confidence": 0.0,
                }

                # Fallback para pasaporte mexicano: extraer número desde documento high-res
                if (not mrz_lines or not has_passport_pattern) and highres_document_image is not None:
                    passport_fallback = ocr_engine.extract_passport_number_fallback(highres_document_image)
                    if passport_fallback.get("passport_number"):
                        mrz_lines = [passport_fallback["passport_number"]] + mrz_lines
                        self.logger.info(
                            f"Fallback número pasaporte detectado: {passport_fallback['passport_number']} "
                            f"(conf {passport_fallback.get('confidence', 0):.1%})"
                        )

                mrz_ocr_confidence = max(
                    mrz_ocr_confidence,
                    float(passport_fallback.get("confidence", 0.0)),
                )
                
                self.logger.debug(f"Extracción de pasaporte: {len(mrz_lines)} elementos (OCR conf: {mrz_ocr_confidence:.1%})")
                
                # Validar datos extraídos
                mrz_result = mrz_validator.validate_mrz(mrz_lines)
                
                result["processing_stages"]["mrz_validation"] = {
                    "status": "success",
                    "mrz_detected": mrz_result.get("mrz_detected", False),
                    "mrz_valid": mrz_result.get("mrz_valid", False),
                    "mrz_format": mrz_result.get("format", "UNKNOWN"),
                    "mrz_lines_detected": len(mrz_lines),
                    "mrz_ocr_confidence": float(mrz_ocr_confidence),
                    "mrz_confidence": mrz_result.get("mrz_confidence_score", 0),
                    "checksum_errors": len(mrz_result.get("checksum_errors", [])),
                    "dedicated_mrz_lines": len(dedicated_mrz_lines),
                    "passport_fallback_detected": bool(passport_fallback.get("passport_number")),
                    "passport_fallback_confidence": float(passport_fallback.get("confidence", 0.0)),
                }
                
                self.logger.info(f"✓ Datos validados: format={mrz_result.get('format', 'UNKNOWN')}, "
                                f"confianza {mrz_result.get('mrz_confidence_score', 0):.1%}")
                
                
                
            except Exception as e:
                self.logger.error(f"✗ Error en validación MRZ: {str(e)}")
                result["processing_stages"]["mrz_validation"] = {
                    "status": "error",
                    "error": str(e),
                }
                # No interrumpir, continuar con scoring parcial
                mrz_result = {
                    "mrz_valid": False,
                    "checksum_errors": [str(e)],
                    "coherence_errors": [],
                    "mrz_confidence_score": 0.0,
                    "details": {},
                }
            
            # ====== ETAPA 4: VALIDACIÓN DE AUTENTICIDAD Y SCORING ======
            self.logger.info("Etapa 4/4: Análisis de autenticidad y cálculo de score")
            self._update_analysis_status("Calculando score de autenticidad...", 4, 4)
            
            try:
                validator = AuthenticityValidator()
                
                # ---- DOCUMENT QUALITY SCORE ----
                # En lugar de usar la confianza cruda de TrOCR (que es baja ~0.3
                # por naturaleza del modelo), construimos un score compuesto
                # basado en QUÉ se detectó exitosamente en el documento.
                
                raw_ocr_conf = float(ocr_result.get("ocr_avg_confidence", 0))
                mrz_score = float(mrz_result.get("mrz_confidence_score", 0))
                
                # Base: imagen procesada y OCR ejecutado correctamente
                doc_quality = 0.65
                
                # Factor 1: OCR extrajo texto con confianza razonable
                if raw_ocr_conf >= 0.50:
                    doc_quality += 0.08
                elif raw_ocr_conf >= 0.30:
                    doc_quality += 0.05
                
                # Factor 2: Se encontró número de pasaporte válido (señal más fuerte)
                has_passport_number = bool(mrz_result.get("mrz_valid") and mrz_result.get("details", {}).get("passport_number"))
                if has_passport_number:
                    doc_quality += 0.10
                
                # Factor 3: Se detectó ROI de zona MRZ en la imagen
                mrz_roi_found = "mrz_ocr_result" in locals() and mrz_ocr_result.get("roi") is not None
                if mrz_roi_found:
                    doc_quality += 0.05
                
                # Factor 4: Se detectaron múltiples líneas MRZ
                num_dedicated_lines = len(dedicated_mrz_lines) if "dedicated_mrz_lines" in locals() else 0
                if num_dedicated_lines >= 2:
                    doc_quality += 0.05
                elif len(mrz_lines) >= 2:
                    doc_quality += 0.03
                
                # Factor 5: Texto contiene patrones MRZ (P<MEX, chevrons, etc.)
                full_text_upper = (ocr_result.get("full_text", "") + " " + (mrz_ocr_result.get("full_text", "") if "mrz_ocr_result" in locals() else "")).upper()
                full_text_nospace = full_text_upper.replace(' ', '')
                
                if 'P<MEX' in full_text_nospace or 'PMEX' in full_text_nospace:
                    doc_quality += 0.05
                
                # Factor 6: MRZ ROI OCR también tuvo buena confianza
                mrz_ocr_conf = float(mrz_ocr_result.get("ocr_avg_confidence", 0)) if "mrz_ocr_result" in locals() else 0.0
                if mrz_ocr_conf >= 0.40:
                    doc_quality += 0.05
                
                # Factor 7: Imagen de alta resolución disponible (mejor procesamiento)
                if highres_document_image is not None:
                    doc_quality += 0.03
                
                ocr_score = min(doc_quality, 0.95)
                
                self.logger.info(f"  Doc quality score: {ocr_score:.2f} (raw_ocr={raw_ocr_conf:.2f}, passport_num={has_passport_number}, mrz_roi={mrz_roi_found})")
                
                # Validación completa con detección de IA
                # IMPORTANTE: Usar imagen ORIGINAL para detección de IA/edición,
                # ya que el preprocesamiento altera histograma y ruido.
                ai_detection_image = original_full_image if original_full_image is not None else processed_image
                final_result = validator.validate(
                    image_array=ai_detection_image,
                    ocr_score=ocr_score,
                    mrz_score=mrz_score,
                    ocr_confidence=ocr_score,
                    mrz_result=mrz_result,
                    original_confidence=float(ocr_score * 40 + mrz_score * 60)
                )
                
                result["processing_stages"]["scoring"] = {
                    "status": "success",
                    "score": final_result.get("autenticidad_score"),
                    "decision": final_result.get("estado"),
                    "ai_detected": final_result.get("analisis", {}).get("ai_analysis", {}).get("is_ai_generated", False),
                    "edited_detected": final_result.get("analisis", {}).get("ai_analysis", {}).get("is_edited", False),
                }
                
                result["final_result"] = final_result
                
                self.logger.info(f"✓ Score calculado: {final_result.get('autenticidad_score'):.1f}% "
                                f"({final_result.get('estado')})")
                self.logger.info(f"  Razón: {final_result.get('razon')}")
                
            except Exception as e:
                self.logger.error(f"✗ Error en validación de autenticidad: {str(e)}")
                result["processing_stages"]["scoring"] = {
                    "status": "error",
                    "error": str(e),
                }
                raise
            
            # ====== TIEMPO TOTAL ======
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds() * 1000
            result["processing_time_ms"] = round(processing_time, 1)
            
            self.logger.info("=" * 60)
            self.logger.info(f"✓ VERIFICACIÓN COMPLETADA EN {processing_time:.1f}ms")
            self.logger.info("=" * 60)
            
            return result
        
        except Exception as e:
            self.logger.error(f"✗ ERROR CRÍTICO: {str(e)}")
            result["error"] = str(e)
            result["final_result"] = {
                "autenticidad_score": 0.0,
                "estado": "REJECT",
                "anomalias": [f"Error crítico: {str(e)}"],
                "recomendacion": "Error en procesamiento - rechazar documento",
            }
            return result


# ============================================================================
# FUNCIONES DE NIVEL SUPERIOR
# ============================================================================

def verify_passport(image_input: Union[str, Path, bytes],
                   verbose: bool = False) -> Dict:
    """
    Función principal para verificar un pasaporte.
    
    Punto de entrada simplificado que crea un verificador
    y realiza la verificación completa.
    
    Args:
        image_input: Imagen del pasaporte
        verbose: Si mostrar logs detallados
        
    Returns:
        dict: Resultado completo de verificación
        
        Estructura:
        {
            "id": "uuid",
            "timestamp": "ISO8601",
            "autenticidad_score": 87.5,
            "estado": "REVIEW",
            "confianza": {
                "ocr": 0.92,
                "mrz": 0.82,
                "final": 0.875
            },
            "anomalias": [...],
            "recomendacion": "Requiere verificación manual",
            "processing_time_ms": 1234.5
        }
    """
    # Configurar logging si es verbose
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    # Crear verificador y ejecutar
    verifier = PassportVerifier()
    result = verifier.verify(image_input, verbose=verbose)
    
    # Simplificar resultado para retorno
    if result.get("final_result"):
        final = result["final_result"]
        # anomalias viene del validador; fallback a metadata_warnings
        anomalias = final.get("anomalias") or final.get("analisis", {}).get("metadata_warnings", [])
        # recomendacion: si no la trae el validador, generar desde estado/razon
        recomendacion = final.get("recomendacion") or final.get("razon", "Análisis completado")
        return {
            "id": result["id"],
            "timestamp": result["timestamp"],
            "autenticidad_score": final.get("autenticidad_score"),
            "estado": final.get("estado"),
            "confianza": final.get("confianza"),
            "anomalias": anomalias,
            "recomendacion": recomendacion,
            "processing_time_ms": result["processing_time_ms"],
            "detalles": final.get("detalles"),
        }
    else:
        return {
            "id": result["id"],
            "timestamp": result["timestamp"],
            "autenticidad_score": 0.0,
            "estado": "REJECT",
            "error": result.get("error"),
            "processing_time_ms": result["processing_time_ms"],
        }


def verify_passport_batch(image_list: list,
                         verbose: bool = False) -> list:
    """
    Verificar múltiples pasaportes en lote.
    
    Args:
        image_list: Lista de imágenes
        verbose: Si mostrar logs
        
    Returns:
        Lista de resultados de verificación
    """
    logger.info(f"Procesando lote de {len(image_list)} pasaportes")
    
    results = []
    for i, image in enumerate(image_list, 1):
        logger.info(f"Procesando {i}/{len(image_list)}")
        result = verify_passport(image, verbose=False)  # No verbose para batch
        results.append(result)
    
    logger.info(f"✓ Lote completado: {len(results)} pasaportes procesados")
    
    return results


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def print_verification_summary(result: Dict) -> None:
    """
    Imprimir resumen de verificación en consola.
    
    Args:
        result: Resultado de verify_passport()
    """
    print("\n" + "=" * 70)
    print("RESULTADO DE VERIFICACIÓN DE PASAPORTE")
    print("=" * 70)
    
    print(f"\nID de Solicitud:   {result.get('id')}")
    print(f"Timestamp:         {result.get('timestamp')}")
    print(f"Tiempo Procesamiento: {result.get('processing_time_ms'):.1f}ms")
    
    print(f"\nScore de Autenticidad: {result.get('autenticidad_score', 0):.1f}%")
    print(f"Estado:                {result.get('estado')}")
    print(f"Recomendación:         {result.get('recomendacion')}")
    
    confianza = result.get('confianza', {})
    if confianza:
        print(f"\nDesglose de Confianza:")
        print(f"  OCR:   {confianza.get('ocr', 0):.1%}")
        print(f"  MRZ:   {confianza.get('mrz', 0):.1%}")
        print(f"  Final: {confianza.get('final', 0):.1%}")
    
    anomalias = result.get('anomalias', [])
    if anomalias:
        print(f"\nAnomalías Detectadas ({len(anomalias)}):")
        for i, anom in enumerate(anomalias[:5], 1):
            print(f"  {i}. {anom}")
        if len(anomalias) > 5:
            print(f"  ... y {len(anomalias) - 5} más")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Módulo pipeline.py cargado correctamente")
    print("Funciones principales:")
    print("  - verify_passport() ← Función de verificación")
    print("  - verify_passport_batch() ← Verificación masiva")
    print("  - PassportVerifier() ← Clase para control avanzado")
