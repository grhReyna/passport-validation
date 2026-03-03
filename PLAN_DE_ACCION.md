# PLAN DE ACCIÓN MRZ (REVISADO)

Fecha: 2026-03-02
Estado actual: el resultado MRZ no es consistente y hoy limita el score final.

## 1) Hallazgos clave del contexto del proyecto

1. El pipeline principal usa OCR sobre imagen completa y después intenta extraer MRZ desde texto libre.
2. La detección de ROI MRZ existe, pero no está integrada como etapa dedicada antes del OCR de MRZ.
3. El validador MRZ mezcla lógica ICAO y formato mexicano en una sola extracción, lo cual aumenta falsos positivos.
4. El archivo de detector MRZ contiene código duplicado/no alcanzable después de un return en manejo de excepciones.
5. Los mensajes de estado del proyecto están optimistas respecto a MRZ; no reflejan la brecha real de precisión.

## 2) Objetivo técnico inmediato (7 días)

Subir la tasa de lectura útil MRZ y estabilizar score final:

- Detección MRZ ROI (IoU aproximado visual): >= 0.70 en set de validación.
- OCR MRZ por línea (accuracy carácter): >= 85%.
- Validación MRZ completa (formato + checksum cuando aplique): >= 70% de documentos válidos.
- Reducir casos de REJECT por falla de MRZ en documentos reales legibles.

## 3) Plan por fases

### Fase A - Diagnóstico reproducible (Día 1)

1. Congelar un mini set de evaluación (20-30 imágenes):
   - Reales legibles
   - Reales con blur/luz difícil
   - Casos sintéticos
2. Guardar por imagen:
   - ROI detectado de documento
   - ROI detectado de MRZ
   - Texto OCR completo
   - Texto OCR sólo MRZ
   - Resultado de validación (checksum/formato)
3. Definir tabla base de métricas para comparar antes/después.

### Fase B - Corrección de arquitectura MRZ (Días 2-3)

1. Separar flujo en dos OCR:
   - OCR general de documento (contexto)
   - OCR dedicado a MRZ usando ROI MRZ
2. Integrar `find_mrz_region` de forma explícita en el pipeline principal.
3. Si no hay ROI MRZ confiable:
   - fallback a banda inferior del documento
   - luego fallback a OCR completo (último recurso)
4. Limpiar el detector MRZ:
   - eliminar ramas duplicadas/no alcanzables
   - unificar criterios de altura y densidad

### Fase C - Robustez de validación (Días 4-5)

1. Separar parser ICAO vs parser mexicano en rutas distintas.
2. Aplicar normalización OCR antes de validar:
   - O↔0, I↔1, B↔8, S↔5 en campos numéricos
3. Ejecutar checksum por campo y compuesto con reportes detallados.
4. Ajustar scoring para no penalizar de más cuando MRZ no es recuperable por calidad de imagen.

### Fase D - Cierre y criterios de salida (Días 6-7)

1. Correr pruebas con set congelado y comparar contra baseline.
2. Publicar métricas en tabla simple (antes/después).
3. Actualizar README y notas de operación con limitaciones reales.
4. Dejar checklist para regresión en cada cambio de OCR/preprocessing.

## 4) Riesgos y mitigaciones

- Riesgo: TrOCR general no está optimizado para líneas MRZ compactas.
  - Mitigación: OCR dedicado a ROI MRZ y preprocesamiento específico de banda.
- Riesgo: dataset pequeño para generalización.
  - Mitigación: benchmark fijo + augmentations orientadas a MRZ (blur, compresión, inclinación).
- Riesgo: mezcla de formatos genera validaciones ambiguas.
  - Mitigación: enrutamiento por tipo de documento antes de validar.

## 5) Prioridades de implementación en código

1. `src/pipeline.py`: introducir etapa OCR MRZ dedicada y fallback ordenado.
2. `src/mrz_roi_detector.py`: limpieza de flujo y criterios de selección.
3. `src/mrz_validator.py`: desacoplar extracción/validación por formato.
4. `test_mrz_detection.py` y scripts debug: usar métricas comparables y salida estructurada.

## 6) Definición de “listo”

Se considera resuelto cuando:

1. El pipeline reporta MRZ con trazabilidad (ROI + texto + validación) por imagen.
2. Las métricas mínimas de la sección 2 se cumplen en el set congelado.
3. El score final deja de depender de supuestos optimistas y refleja evidencia real.

---

Nota: este plan reemplaza el enfoque de “sistema 100% funcional” por uno de estabilización técnica orientada a MRZ, que hoy es el principal cuello de botella.
