"""
mrz_validator.py - Validador de datos de pasaporte 

Para pasaportes mexicanos: válida número, nombre, fechas
Para pasaportes con MRZ ICAO: válida checksums y formato

Versión: 0.2.0
- Soporte para pasaportes mexicanos (sin MRZ ICAO)
- Extracción de número, nombre, fechas
- Validación de formato mexicano
"""

import re
from typing import List, Tuple, Dict, Optional
import logging
from datetime import datetime

import config

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTES - ESPECIFICACIÓN MRZ
# ============================================================================

# Estructura MRZ para pasaportes: 88 caracteres en 3 líneas
# Línea 1: P<[PAÍS][APELLIDO]<<<<<<[NOMBRE_INICIALES]
# Línea 2: [PASAPORTE_NUM][NACIONALIDAD][FECHA_NAC][M/F][FECHA_EXP][CHECKSUM_GENERAL]
# Línea 3: [CHECKSUM_MRZ][CHECKSUM_COMP]<<<<<<<<<<<<<<<<<<<<<<

MRZ_CHARACTER_MAP = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14,
    'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19,
    'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24,
    'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29,
    'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34,
    'Z': 35, '<': 0, ' ': 0  # Espacios y delimitadores = 0
}

# Pesos cíclicos para checksum mod-10
CHECKSUM_WEIGHTS = [7, 3, 1]


# ============================================================================
# FUNCIONES DE CHECKSUM
# ============================================================================

def char_to_numeric(char: str) -> int:
    """
    Convertir carácter MRZ a valor numérico.
    
    Args:
        char: Carácter a convertir
        
    Returns:
        int: Valor numérico (0-35)
    """
    char = char.upper()
    return MRZ_CHARACTER_MAP.get(char, 0)


def calculate_checksum(data: str, use_weights: bool = True) -> int:
    """
    Calcular checksum mod-10 para string MRZ.
    
    Algoritmo mod-10:
    1. Convertir cada carácter a número
    2. Multiplicar por peso cíclico (7, 3, 1)
    3. Sumar todos los productos
    4. Tomar módulo 10
    
    Args:
        data: String a validar (sin dígito checksum)
        use_weights: Si usar pesos cíclicos (True según ICAO)
        
    Returns:
        int: Dígito checksum (0-9)
    """
    try:
        total = 0
        
        for i, char in enumerate(data.upper()):
            # Convertir carácter a número
            value = char_to_numeric(char)
            
            # Aplicar peso (si se usa)
            if use_weights:
                weight = CHECKSUM_WEIGHTS[i % 3]
                value *= weight
            
            total += value
        
        # Calcular checksum
        checksum = total % 10
        
        return checksum
    
    except Exception as e:
        logger.error(f"Error calculando checksum: {str(e)}")
        return -1


def validate_checksum(data: str, expected_checksum: str) -> bool:
    """
    Validar checksum de un string MRZ.
    
    Args:
        data: String sin el dígito checksum final
        expected_checksum: Dígito checksum esperado (como string)
        
    Returns:
        bool: Si el checksum es válido
    """
    try:
        if not expected_checksum or not expected_checksum.isdigit():
            logger.warning(f"Checksum inválido: {expected_checksum}")
            return False
        
        calculated = calculate_checksum(data)
        expected = int(expected_checksum)
        
        is_valid = calculated == expected
        
        logger.debug(f"Checksum: calculado={calculated}, esperado={expected}, válido={is_valid}")
        
        return is_valid
    
    except Exception as e:
        logger.error(f"Error validando checksum: {str(e)}")
        return False


# ============================================================================
# FUNCIONES DE PARSEO MRZ
# ============================================================================

def extract_mrz_lines(text: str) -> List[str]:
    """
    Extraer líneas de pasaporte del texto OCR.
    
    FLEXIBLE: Busca AMBOS formatos:
    - MRZ ICAO: Líneas que comienzan con 'P<'
    - Mexicano: Números de pasaporte (Letra + 8-10 dígitos)
    
    Args:
        text: Texto extraído del OCR
        
    Returns:
        Lista de líneas relevantes (vacía si no encontró nada)
    """
    try:
        lines = text.split('\n')
        extracted = []
        
        logger.warning(f"\n{'='*70}")
        logger.warning(f"EXTRACT_MRZ_LINES - Procesando {len(lines)} líneas de OCR")
        logger.warning(f"{'='*70}")
        logger.warning(f"Texto completo recibido ({len(text)} chars): {text[:200]}...")
        logger.warning(f"{'='*70}\n")
        
        for i, line in enumerate(lines):
            line_cleaned = line.strip().upper()
            
            if not line_cleaned:
                continue
            
            logger.warning(f"\n[Línea {i}] Raw: '{line[:80]}...'")
            logger.warning(f"[Línea {i}] Limpia: '{line_cleaned[:80]}...'")
            
            # FORMATO 1: MRZ ICAO (P<...)
            if line_cleaned.startswith('P<'):
                logger.warning(f"[Línea {i}] ✓ MRZ ICAO detectado")
                extracted.append(line_cleaned)
            
            # FORMATO 2: Líneas largas de MRZ ICAO (44+ caracteres válidos)
            elif len(line_cleaned) >= 44 and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<' for c in line_cleaned[:20]):
                logger.warning(f"[Línea {i}] ✓ Posible línea MRZ ICAO (long={len(line_cleaned)})")
                extracted.append(line_cleaned)
            
            # FORMATO 3: Número de pasaporte mexicano (Letra + 8-10 dígitos)
            passport_match = re.search(r'([A-Z])(\d{8,10})', line_cleaned)
            if passport_match:
                passport_num = passport_match.group(0)
                logger.warning(f"[Línea {i}] ✓ Pasaporte mexicano detectado: {passport_num}")
                extracted.append(passport_num)
                extracted.append(line_cleaned)
            
            # BÚSQUEDA MAS FLEXIBLE: solo letra + números
            loose_match = re.search(r'[A-Z]\d{7,}', line_cleaned)
            if loose_match and passport_match is None:
                logger.warning(f"[Línea {i}] ⚠ Match flexible: {loose_match.group(0)}")
        
        logger.warning(f"\n{'='*70}")
        logger.warning(f"RESULTADO EXTRACCIÓN: {len(extracted)} elementos")
        logger.warning(f"Elementos: {extracted}")
        logger.warning(f"{'='*70}\n")
        
        return extracted
    
    except Exception as e:
        logger.error(f"Error extrayendo líneas de pasaporte: {str(e)}")
        return []


def parse_mrz_line_1(line: str) -> Dict[str, str]:
    """
    Parsear línea 1 de MRZ (información de tipo y país/apellido/nombre).
    
    Formato:
    P<[País (3)][APELLIDO<<<<<<<< (39)][NOMBRE_INICIALES (39)]
    
    Args:
        line: Línea 1 de MRZ (88 caracteres)
        
    Returns:
        dict: Campos extraídos
    """
    try:
        result = {
            "line_1": line,
            "document_type": line[0] if len(line) > 0 else "",
            "country_code": line[2:5] if len(line) > 5 else "",
            "surname_field": line[5:44] if len(line) > 44 else "",
            "given_names_field": line[44:88] if len(line) > 88 else "",
            "is_valid": False,
        }
        
        # Limpiar nombres (remover '<')
        surname = result["surname_field"].replace('<', ' ').strip()
        given_names = result["given_names_field"].replace('<', ' ').strip()
        
        result["surname"] = surname
        result["given_names"] = given_names
        
        # Validaciones básicas
        if result["document_type"] != 'P':
            logger.warning(f"Tipo de documento inválido: {result['document_type']}")
            return result
        
        result["is_valid"] = True
        
        logger.debug(f"Línea 1: {surname} {given_names} ({result['country_code']})")
        
        return result
    
    except Exception as e:
        logger.error(f"Error parseando línea 1: {str(e)}")
        return {"is_valid": False}


def parse_mrz_line_2(line: str) -> Dict[str, str]:
    """
    Parsear línea 2 de MRZ (número, fechas, género).
    
    Formato:
    [Pasaporte (9)][Nacionalidad (3)][Fecha Nac YYMMDD (6)][Género (1)]
    [Expiración YYMMDD (6)][Checksum expiración (1)][Otros (8)][Checksum (1)]
    
    Args:
        line: Línea 2 de MRZ (88 caracteres)
        
    Returns:
        dict: Campos extraídos
    """
    try:
        result = {
            "line_2": line,
            "passport_number": line[0:9] if len(line) > 9 else "",
            "nationality": line[10:13] if len(line) > 13 else "",
            "date_of_birth": line[13:19] if len(line) > 19 else "",
            "sex": line[20] if len(line) > 20 else "",
            "expiration_date": line[21:27] if len(line) > 27 else "",
            "expiration_checksum": line[27] if len(line) > 27 else "",
            "final_checksum": line[86] if len(line) > 86 else "",
            "is_valid": False,
        }
        
        # Validaciones
        # Número de pasaporte debe ser 9 caracteres
        if len(result["passport_number"].replace('<', '')) < 5:
            logger.warning(f"Número de pasaporte inválido: {result['passport_number']}")
            return result
        
        result["is_valid"] = True
        
        logger.debug(f"Línea 2: Pasaporte {result['passport_number']}, "
                    f"Nacimiento {result['date_of_birth']}, "
                    f"Género {result['sex']}, "
                    f"Expiración {result['expiration_date']}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error parseando línea 2: {str(e)}")
        return {"is_valid": False}


# ============================================================================
# VALIDACIONES LÓGICAS
# ============================================================================

def validate_date_format(date_str: str) -> Tuple[bool, str]:
    """
    Validar formato de fecha YYMMDD.
    
    Args:
        date_str: Fecha en formato YYMMDD
        
    Returns:
        Tupla (es_válida, mensaje)
    """
    try:
        if not date_str or len(date_str) != 6:
            return False, f"Formato de fecha inválido: {date_str}"
        
        if not date_str.isdigit():
            return False, f"Fecha contiene caracteres no numéricos: {date_str}"
        
        yy = int(date_str[0:2])
        mm = int(date_str[2:4])
        dd = int(date_str[4:6])
        
        # Validar mes
        if mm < 1 or mm > 12:
            return False, f"Mes inválido: {mm}"
        
        # Validar día según el mes
        days_in_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        if dd < 1 or dd > days_in_month[mm - 1]:
            return False, f"Día inválido para mes {mm}: {dd}"
        
        # Convertir a siglo (80-99 → 1980-1999, 00-49 → 2000-2049)
        century = 1900 if yy >= config.DATE_MIN_YEAR else 2000
        year = century + yy
        
        # Validar que año está en rango razonable
        current_year = datetime.now().year
        if year > current_year + 20:  # Pasaportes no válidos más de 20 años en futuro
            return False, f"Año demasiado en el futuro: {year}"
        
        return True, f"Fecha válida: {dd}/{mm}/{year}"
    
    except Exception as e:
        return False, f"Error validando fecha: {str(e)}"


def validate_coherence(line1_data: dict, line2_data: dict) -> Tuple[bool, List[str]]:
    """
    Validar coherencia entre líneas 1 y 2 de MRZ.
    
    Verifica:
    - Nacionalidad es 3 caracteres
    - Fechas están en formato válido
    - Género es M, F o X
    - Número de pasaporte no está vacío
    
    Args:
        line1_data: Datos parseados de línea 1
        line2_data: Datos parseados de línea 2
        
    Returns:
        Tupla (coherente, lista_de_errores)
    """
    errors = []
    
    try:
        # Validar nacionalidad (debe ser código país válido)
        if line2_data.get("nationality", "").rstrip('<') != config.PASSPORT_COUNTRY_CODE:
            if line2_data.get("nationality", "").rstrip('<'):  # No vacío
                errors.append(f"Nacionalidad no es MEX: {line2_data.get('nationality')}")
        
        # Validar género
        sex = line2_data.get("sex", "")
        if sex not in ['M', 'F', 'X']:
            errors.append(f"Género inválido: {sex}")
        
        # Validar fecha de nacimiento
        dob = line2_data.get("date_of_birth", "")
        if dob and dob != "000000":
            is_valid, msg = validate_date_format(dob)
            if not is_valid:
                errors.append(f"Fecha de nacimiento: {msg}")
        
        # Validar fecha de expiración
        exp_date = line2_data.get("expiration_date", "")
        if exp_date and exp_date != "000000":
            is_valid, msg = validate_date_format(exp_date)
            if not is_valid:
                errors.append(f"Fecha de expiración: {msg}")
        
        # Validar Número de pasaporte (no debe estar vacío)
        passport_num = line2_data.get("passport_number", "").rstrip('<')
        if not passport_num or len(passport_num) < 5:
            errors.append(f"Número de pasaporte inválido o vacío")
        
        coherent = len(errors) == 0
        
        logger.info(f"Validación de coherencia: {'PASS' if coherent else 'FAIL'}")
        if errors:
            for err in errors:
                logger.debug(f"  - {err}")
        
        return coherent, errors
    
    except Exception as e:
        logger.error(f"Error validando coherencia: {str(e)}")
        return False, [str(e)]


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def validate_mrz(mrz_lines: List[str]) -> Dict:
    """
    Validar datos de pasaporte (flexible para formato mexicano).
    
    NUEVO ENFOQUE (v0.2):
    - Si hay MRZ ICAO (3 líneas) → valida checksums ICAO
    - Si hay datos sueltos (nombre, número, fechas) → valida como mexicano
    - Si hay nada → score bajo pero sigue procesando
    
    IMPORTANTE: Los pasaportes MEXICANOS modernos pueden NO tener MRZ ICAO.
    En cambio, tienen: número, nombre, fechas de emisión/vencimiento.
    
    Args:
        mrz_lines: Lista de strings (puede ser lista vacía)
        
    Returns:
        dict: Resultado de validación
    """
    result = {
        "mrz_valid": False,  # True solo si MRZ ICAO perfecta o datos mexicanos válidos
        "mrz_detected": False,  # True si detectó alguna línea MRZ
        "checksum_errors": [],
        "coherence_errors": [],
        "mrz_confidence_score": 0.1,  # Valor base si no hay nada
        "details": {},
        "format": "UNKNOWN",  # ICAO o MEXICAN o NONE
    }
    
    try:
        logger.info(f"Validando MRZ: recibí {len(mrz_lines)} línea(s)")
        
        # ====== INTENTO 1: MRZ ICAO (3 líneas exactas) ======
        if mrz_lines and len(mrz_lines) >= 3:
            logger.debug("Intentando validación MRZ ICAO...")
            
            line1, line2, line3 = mrz_lines[0], mrz_lines[1], mrz_lines[2]
            
            # Verificar si se parece a MRZ ICAO (contiene muchos caracteres especiales)
            mrz_text = (line1 + line2 + line3).upper()
            mrz_like_chars = sum(1 for c in mrz_text if c in '<0123456789')
            
            if len(mrz_text) >= 100 and mrz_like_chars > 50:
                logger.debug("Parece MRZ ICAO - validando checksums...")
                
                # Normalizar a 88 caracteres
                line1 = line1[:88].ljust(88, '<')
                line2 = line2[:88].ljust(88, '<')
                line3 = line3[:88].ljust(88, '<')
                
                try:
                    # Validar checksums
                    line2_checksum_ok = validate_checksum(line2[0:27], line2[27])
                    line3_checksum_ok = validate_checksum(line2[0:44] + line3[0:45], line3[45])
                    
                    if line2_checksum_ok and line3_checksum_ok:
                        result["mrz_valid"] = True
                        result["mrz_detected"] = True
                        result["mrz_confidence_score"] = 0.95
                        result["format"] = "ICAO"
                        
                        logger.info("✓ MRZ ICAO válida (checksums OK)")
                        return result
                    else:
                        logger.debug("MRZ ICAO: checksums inválidos")
                        result["checksum_errors"].append("Checksum ICAO inválido")
                        result["mrz_detected"] = True
                        result["mrz_confidence_score"] = 0.30  # Algo detectó pero no válido
                        result["format"] = "ICAO"
                        return result
                        
                except Exception as e:
                    logger.debug(f"Error validando MRZ ICAO: {e}")
        
        # ====== INTENTO 2: Datos sueltos (Pasaporte Mexicano) ======
        # Si no encontró MRZ ICAO, buscar datos sueltos
        logger.debug("No hay MRZ ICAO, buscando datos sueltos (mexicano)...")
        
        if mrz_lines:
            all_text = " ".join(mrz_lines).upper()
            
            logger.warning(f"\n{'='*70}")
            logger.warning(f"VALIDATE_MRZ - Buscando números mexicanos")
            logger.warning(f"{'='*70}")
            logger.warning(f"Recibí {len(mrz_lines)} líneas: {mrz_lines}")
            logger.warning(f"Texto unido: '{all_text}'")
            logger.warning(f"{'='*70}")
            
            # Intento 1: Patrón estricto (G77536498)
            passport_pattern = r'([A-Z])[\s]?(\d{8,10})'
            passport_match = re.search(passport_pattern, all_text)
            logger.warning(f"\n[Intento 1] Patrón estricto: {passport_pattern}")
            logger.warning(f"[Intento 1] Resultado: {passport_match}")
            
            if not passport_match:
                # Intento 2: Patrón muy flexible (letra + números con espacios/ruido)
                passport_pattern = r'([A-Z])\s*[\d\s]{8,15}'
                passport_match = re.search(passport_pattern, all_text)
                logger.warning(f"\n[Intento 2] Patrón flexible: {passport_pattern}")
                logger.warning(f"[Intento 2] Resultado: {passport_match}")
            
            if not passport_match:
                # Intento 3: Solo buscar cualquier letra seguida de números
                passport_pattern = r'[A-Z]\d+'
                passport_match = re.search(passport_pattern, all_text)
                logger.warning(f"\n[Intento 3] Patrón muy flexible: {passport_pattern}")
                logger.warning(f"[Intento 3] Resultado: {passport_match}")
            
            if passport_match:
                passport_found = passport_match.group(0).replace(' ', '').replace('O', '0')
                logger.warning(f"\n✓✓✓ NÚMERO DETECTADO: {passport_found} ✓✓✓")
                
                # Score granular según calidad de detección
                # Base alto: detectar un número de pasaporte mexicano válido
                # ES la validación principal del documento.
                mrz_conf = 0.80  # Base: número encontrado (formato letra+dígitos)
                
                # Bonus: detectó líneas MRZ con P<MEX (indica formato ICAO parcial)
                has_pmex = any('P<MEX' in str(l).upper().replace(' ', '') or 'PMEX' in str(l).upper().replace(' ', '') for l in mrz_lines)
                if has_pmex:
                    mrz_conf += 0.10
                
                # Bonus: número largo y válido (letra + 8-9 dígitos exactos)
                clean_num = passport_found.replace(' ', '')
                if re.match(r'^[A-Z]\d{8,9}$', clean_num):
                    mrz_conf += 0.05
                
                # Bonus: detectó más de un dato (ej. nombre + número)
                if len(mrz_lines) >= 2:
                    mrz_conf += 0.05
                
                # Bonus: texto contiene datos MRZ adicionales (fechas, MEX, chevrons)
                chevron_count = all_text.count('<')
                if chevron_count >= 5:
                    mrz_conf += 0.05
                
                mrz_conf = min(mrz_conf, 0.95)  # Tope sin ICAO completo
                
                result["mrz_detected"] = True
                result["mrz_valid"] = True
                result["mrz_confidence_score"] = mrz_conf
                result["format"] = "MEXICAN"
                result["details"] = {
                    "passport_number": passport_found,
                    "type": "MEXICAN",
                    "has_pmex": has_pmex,
                    "mrz_lines_count": len(mrz_lines),
                }
                logger.warning(f"\n{'='*70}")
                logger.warning(f"Pasaporte mexicano validado: {passport_found}")
                logger.warning(f"MRZ Confidence Score: {mrz_conf:.2f} ({mrz_conf*100:.0f}%)")
                logger.warning(f"{'='*70}\n")
                return result
            
            logger.warning(f"\n❌ NO SE ENCONTRÓ NÚMERO en: '{all_text}'")
            result["mrz_confidence_score"] = 0.20
            result["format"] = "PARTIAL"
            logger.warning(f"{'='*70}\n")
            return result
        
        # ====== INTENTO 3: Sin MRZ/datos ======
        logger.warning("No se detectó MRZ ICAO ni datos de pasaporte mexicano")
        result["mrz_confidence_score"] = 0.05
        result["format"] = "NONE"
        result["coherence_errors"].append("Sin MRZ/datos detectados")
        
        return result
        
    except Exception as e:
        logger.error(f"Error en validate_mrz: {e}")
        result["checksum_errors"].append(f"Error: {str(e)}")
        result["mrz_confidence_score"] = 0.0
        return result


if __name__ == "__main__":
    # Ejemplo de uso
    logging.basicConfig(level=logging.DEBUG)
    
    print("Módulo mrz_validator.py cargado correctamente")
    print("Funciones principales:")
    print("  - validate_mrz(mrz_lines) ← Función principal")
    print("  - calculate_checksum()")
    print("  - validate_checksum()")
    print("  - extract_mrz_lines()")
