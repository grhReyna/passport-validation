"""
confidence_scorer.py - Motor de Scoring de Confianza

Combina resultados de OCR y validación MRZ para generar
un score de confianza (0-100%) sobre la autenticidad del pasaporte.

Fórmula:
score = (ocr_conf × 0.40) + (mrz_conf × 0.60)

Decisión:
- Score ≥ 90% → PASS (aceptar)
- 70% ≤ Score < 90% → REVIEW (verificación manual)
- Score < 70% → REJECT (rechazar)

Versión: 0.1.0
"""

from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

import config

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTES
# ============================================================================

DECISION_PASS = "PASS"
DECISION_REVIEW = "REVIEW"
DECISION_REJECT = "REJECT"

DECISION_DESCRIPTIONS = {
    DECISION_PASS: "✓ Documento auténtico con alta confianza",
    DECISION_REVIEW: "⚠ Documento requiere verificación manual",
    DECISION_REJECT: "✗ Documento probablemente falsificado",
}


# ============================================================================
# FUNCIONES DE CÁLCULO DE SCORES
# ============================================================================

def calculate_ocr_score(ocr_result: Dict) -> Tuple[float, List[str]]:
    """
    Calcular score de confianza del OCR.
    
    Basado en:
    - Confianza promedio de tokens
    - Número de tokens (más tokens = más confianza en extracción completa)
    - Detección de campos críticos
    
    Args:
        ocr_result: Resultado de ocr_engine.extract_text_with_confidence()
        
    Returns:
        Tupla (score 0-1, lista_anomalías)
    """
    anomalies = []
    
    try:
        # Score base: confianza promedio
        base_score = ocr_result.get("ocr_avg_confidence", 0.5)
        
        if base_score < 0.5:
            anomalies.append(f"OCR muy poco confiable: {base_score:.1%}")
            base_score = 0.3  # Penalizar severamente
        
        elif base_score < 0.65:
            anomalies.append(f"OCR con baja confianza: {base_score:.1%}")
            base_score = max(base_score, 0.5)  # Asegurar mínimo
        
        # Bonus si MRZ fue detectada (indica buen procesamiento)
        if ocr_result.get("mrz_detected", False):
            bonus = 0.1
            base_score = min(1.0, base_score + bonus)
            logger.debug(f"Bonus por MRZ detectada: +{bonus}")
        else:
            anomalies.append("MRZ no detectada en OCR")
        
        # Validar número de tokens (debe haber contenido significativo)
        num_tokens = len(ocr_result.get("tokens", []))
        if num_tokens < 3:
            anomalies.append(f"Muy pocos tokens extraídos: {num_tokens}")
            base_score = base_score * 0.7
        
        # Asegurar rango 0-1
        ocr_score = max(0.0, min(1.0, base_score))
        
        logger.debug(f"OCR Score: {ocr_score:.2%} (base: {base_score:.2%}, "
                    f"tokens: {num_tokens})")
        
        return ocr_score, anomalies
    
    except Exception as e:
        logger.error(f"Error calculando OCR score: {str(e)}")
        return 0.5, [f"Error en OCR score: {str(e)}"]


def calculate_mrz_score(mrz_result: Dict) -> Tuple[float, List[str]]:
    """
    Calcular score de confianza del MRZ.
    
    Basado en:
    - Validez general de MRZ
    - Checksums correctos
    - Coherencia de datos
    - Confianza de validación MRZ
    
    Args:
        mrz_result: Resultado de mrz_validator.validate_mrz()
        
    Returns:
        Tupla (score 0-1, lista_anomalías)
    """
    anomalies = []
    
    try:
        # Score base
        if mrz_result.get("mrz_valid", False):
            base_score = 1.0
        else:
            base_score = 0.0
        
        # Penalizar por errores de checksum específicos
        checksum_errors = mrz_result.get("checksum_errors", [])
        if checksum_errors:
            base_score = 0.5  # Media si hay errores de checksum
            for error in checksum_errors:
                anomalies.append(f"Checksum: {error}")
        
        # Penalizar por errores de coherencia
        coherence_errors = mrz_result.get("coherence_errors", [])
        if coherence_errors:
            base_score = base_score * 0.7  # Reducir score
            for error in coherence_errors:
                anomalies.append(f"Coherencia: {error}")
        
        # Usar confidence_score de validación (0-1)
        mrz_confidence = mrz_result.get("mrz_confidence_score", 0.5)
        
        # Combinar
        mrz_score = base_score * mrz_confidence
        
        # Asegurar rango 0-1
        mrz_score = max(0.0, min(1.0, mrz_score))
        
        logger.debug(f"MRZ Score: {mrz_score:.2%} (base: {base_score:.2%}, "
                    f"confidence: {mrz_confidence:.2%}, valid: {mrz_result.get('mrz_valid')})")
        
        return mrz_score, anomalies
    
    except Exception as e:
        logger.error(f"Error calculando MRZ score: {str(e)}")
        return 0.5, [f"Error en MRZ score: {str(e)}"]


def combine_scores(ocr_score: float, 
                  mrz_score: float,
                  ocr_weight: float = config.OCR_WEIGHT,
                  mrz_weight: float = config.MRZ_WEIGHT) -> float:
    """
    Combinar scores OCR y MRZ en score final.
    
    Fórmula:
    score = (ocr × ocr_weight) + (mrz × mrz_weight)
    
    Args:
        ocr_score: Score OCR 0-1
        mrz_score: Score MRZ 0-1
        ocr_weight: Peso para OCR (default 0.40)
        mrz_weight: Peso para MRZ (default 0.60)
        
    Returns:
        float: Score combinado 0-1
    """
    try:
        # Normalizar pesos (opcional, en caso no sumen 1)
        total_weight = ocr_weight + mrz_weight
        ocr_weight = ocr_weight / total_weight
        mrz_weight = mrz_weight / total_weight
        
        combined = (ocr_score * ocr_weight) + (mrz_score * mrz_weight)
        
        logger.debug(f"Scores combinados: "
                    f"OCR {ocr_score:.2%} × {ocr_weight:.1%} + "
                    f"MRZ {mrz_score:.2%} × {mrz_weight:.1%} = {combined:.2%}")
        
        return combined
    
    except Exception as e:
        logger.error(f"Error combinando scores: {str(e)}")
        return 0.5


def generate_decision(confidence_score: float) -> Tuple[str, str]:
    """
    Generar decisión basada en confidence score.
    
    Args:
        confidence_score: Score 0-1 (o 0-100)
        
    Returns:
        Tupla (decisión, descripción)
    """
    try:
        # Normalizar a rango 0-1 si está en 0-100
        if confidence_score > 1.0:
            score = confidence_score / 100.0
        else:
            score = confidence_score
        
        if score >= config.PASS_THRESHOLD:
            decision = DECISION_PASS
        elif score >= config.REVIEW_THRESHOLD:
            decision = DECISION_REVIEW
        else:
            decision = DECISION_REJECT
        
        description = DECISION_DESCRIPTIONS[decision]
        
        logger.info(f"Decisión: {decision} (score: {score:.2%})")
        
        return decision, description
    
    except Exception as e:
        logger.error(f"Error generando decisión: {str(e)}")
        return DECISION_REVIEW, "Error generando decisión"


# ============================================================================
# ANÁLISIS DE ANOMALÍAS
# ============================================================================

def identify_anomalies(ocr_result: Dict, 
                      mrz_result: Dict,
                      ocr_anomalies: List[str],
                      mrz_anomalies: List[str]) -> List[str]:
    """
    Identificar y compilar todas las anomalías detectadas.
    
    Args:
        ocr_result: Resultado de OCR
        mrz_result: Resultado de validación MRZ
        ocr_anomalies: Anomalías del score OCR
        mrz_anomalies: Anomalías del score MRZ
        
    Returns:
        Lista de anomalías compiladas y ordenadas
    """
    anomalies = []
    
    try:
        # Anomalías de OCR
        anomalies.extend(ocr_anomalies)
        
        # Anomalías de MRZ
        anomalies.extend(mrz_anomalies)
        
        # Anomalías adicionales basadas en datos
        
        # Si MRZ y OCR no coinciden en país
        mrz_country = mrz_result.get("details", {}).get("country_code", "")
        if mrz_country and mrz_country != config.PASSPORT_COUNTRY_CODE:
            anomalies.append(f"País no es MEX: {mrz_country}")
        
        # Si pasaporte está vencido
        exp_date_str = mrz_result.get("details", {}).get("expiration_date", "")
        if exp_date_str and exp_date_str != "000000":
            try:
                yy = int(exp_date_str[0:2])
                mm = int(exp_date_str[2:4])
                dd = int(exp_date_str[4:6])
                
                century = 1900 if yy >= config.DATE_MIN_YEAR else 2000
                year = century + yy
                
                # Crear fecha de expiración
                from datetime import datetime
                exp_date = datetime(year, mm, dd)
                
                if exp_date < datetime.now():
                    anomalies.append(f"Pasaporte vencido: {exp_date_str}")
            except:
                pass  # Ignorar errores en fecha
        
        # Remover duplicados manteniendo orden
        seen = set()
        unique_anomalies = []
        for anom in anomalies:
            if anom not in seen:
                seen.add(anom)
                unique_anomalies.append(anom)
        
        return unique_anomalies
    
    except Exception as e:
        logger.error(f"Error identificando anomalías: {str(e)}")
        return anomalies


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def compute_final_score(ocr_result: Dict, 
                        mrz_result: Dict) -> Dict:
    """
    Calcular score final de confianza y decisión de autenticidad.
    
    MEJORADO: Detecta si es pasaporte mexicano (sin MRZ ICAO) y
    ajusta los pesos automáticamente para justa evaluación.
    
    Args:
        ocr_result: Resultado de ocr_engine.extract_text_with_confidence()
        mrz_result: Resultado de mrz_validator.validate_mrz()
        
    Returns:
        dict: Resultado completo de scoring
    """
    logger.info("Calculando score final de confianza")
    
    try:
        # 1. Calcular score OCR
        ocr_score, ocr_anomalies = calculate_ocr_score(ocr_result)
        
        # 2. Calcular score MRZ
        mrz_score, mrz_anomalies = calculate_mrz_score(mrz_result)
        
        # 3. Ajustar pesos según tipo de pasaporte
        mrz_format = mrz_result.get("format", "UNKNOWN")
        
        if mrz_format == "MEXICAN":
            # Pasaporte mexicano: MRZ tiene menos peso porque no es ICAO
            ocr_weight = 0.80
            mrz_weight = 0.20
            logger.info(f"Formato MEXICANO detectado - ajustando pesos")
        elif mrz_format == "ICAO":
            # Pasaporte ICAO: MRZ tiene más peso (es fundamental)
            ocr_weight = 0.40
            mrz_weight = 0.60
            logger.info(f"Formato ICAO detectado")
        elif mrz_format == "PARTIAL":
            # Datos parciales: Peso equilibrado
            ocr_weight = 0.70
            mrz_weight = 0.30
            logger.info(f"Datos PARCIALES detectados")
        else:
            # Desconocido: Dar más peso a OCR (es lo que sí vemos)
            ocr_weight = 0.85
            mrz_weight = 0.15
            logger.info(f"Formato DESCONOCIDO - usando pesos conservadores")
        
        # 4. Combinar scores con pesos ajustados
        final_score_0_1 = combine_scores(ocr_score, mrz_score, 
                                         ocr_weight, mrz_weight)
        final_score_0_100 = final_score_0_1 * 100
        
        # 5. Generar decisión
        decision, description = generate_decision(final_score_0_1)
        
        # 6. Compilar anomalías
        all_anomalies = identify_anomalies(
            ocr_result, mrz_result,
            ocr_anomalies, mrz_anomalies
        )
        
        # 7. Generar recomendación
        if decision == DECISION_PASS:
            recommendation = "✓ Documento aceptado automáticamente"
        elif decision == DECISION_REVIEW:
            if mrz_format == "MEXICAN":
                recommendation = "⚠ Pasaporte mexicano - Requiere verificación manual"
            else:
                recommendation = "⚠ Requiere verificación manual por analista"
        else:
            recommendation = "✗ Documento rechazado - posible falsificación"
        
        # 8. Compilar resultado final
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "autenticidad_score": round(final_score_0_100, 1),
            "estado": decision,
            "confianza": {
                "ocr": round(ocr_score, 3),
                "mrz": round(mrz_score, 3),
                "final": round(final_score_0_1, 3),
            },
            "anomalias": all_anomalies,
            "detalles": {
                "mrz_valido": mrz_result.get("mrz_valid", False),
                "mrz_detectado": mrz_result.get("mrz_detected", False),
                "mrz_formato": mrz_result.get("format", "UNKNOWN"),
                "checksum_status": "PASS" if not mrz_result.get("checksum_errors") else "FAIL",
                "coherencia_status": "PASS" if not mrz_result.get("coherence_errors") else "FAIL",
            },
            "recomendacion": recommendation,
            "descripcion_estado": description,
        }
        
        # Agregar información de pasaporte si está disponible
        passport_details = mrz_result.get("details", {})
        if passport_details:
            result["detalles"]["passport_info"] = {
                "pais": passport_details.get("country_code"),
                "numero": passport_details.get("passport_number"),
                "nacionalidad": passport_details.get("nationality"),
                "genero": passport_details.get("sex"),
            }
        
        logger.info(f"✓ Score final: {final_score_0_100:.1f}% ({decision}) [Formato: {mrz_format}]")
        
        return result
    
    except Exception as e:
        logger.error(f"Error calculando score final: {str(e)}")
        
        # Retornar resultado error
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "autenticidad_score": 0.0,
            "estado": DECISION_REJECT,
            "confianza": {
                "ocr": 0.0,
                "mrz": 0.0,
                "final": 0.0,
            },
            "anomalias": [f"Error crítico: {str(e)}"],
            "detalles": {
                "error": str(e),
                "mrz_valido": False,
            },
            "recomendacion": "Error en procesamiento - rechazar documento",
            "descripcion_estado": "Error crítico en validación",
        }


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def explain_score(result: Dict) -> str:
    """
    Generar explicación en lenguaje natural del score.
    
    Args:
        result: Resultado de compute_final_score()
        
    Returns:
        str: Explicación detallada
    """
    try:
        score = result["autenticidad_score"]
        estado = result["estado"]
        
        explanation = f"""
╔════════════════════════════════════════════════════════════════╗
║               REPORTE DE VERIFICACIÓN DE PASAPORTE            ║
╚════════════════════════════════════════════════════════════════╝

📊 SCORE FINAL: {score:.1f}%

Estado: {estado} - {result.get('descripcion_estado', '')}

Componentes:
  • OCR: {result['confianza']['ocr']:.1%}
  • MRZ: {result['confianza']['mrz']:.1%}

Validations:
  ✓ MRZ Válida:        {'SÍ' if result['detalles'].get('mrz_valido') else 'NO'}
  ✓ Checksums:         {result['detalles'].get('checksum_status', 'UNKNOWN')}
  ✓ Coherencia:        {result['detalles'].get('coherencia_status', 'UNKNOWN')}

Recomendación:
  {result['recomendacion']}

Anomalías Detectadas: {len(result['anomalias'])}
"""
        
        if result['anomalias']:
            explanation += "\n  Detalles:\n"
            for i, anom in enumerate(result['anomalias'][:5], 1):
                explanation += f"    {i}. {anom}\n"
            if len(result['anomalias']) > 5:
                explanation += f"    ... y {len(result['anomalias']) - 5} más\n"
        
        return explanation
    
    except Exception as e:
        return f"Error generando explicación: {str(e)}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("Módulo confidence_scorer.py cargado correctamente")
    print("Funciones principales:")
    print("  - compute_final_score() ← Función principal")
    print("  - calculate_ocr_score()")
    print("  - calculate_mrz_score()")
    print("  - combine_scores()")
    print("  - explain_score()")
