# 📚 ÍNDICE DE DOCUMENTACIÓN

## 🚀 COMENZAR AQUI

1. **[RESUMEN_FINAL.md](RESUMEN_FINAL.md)** - ¿Qué se logró? ← **LEER PRIMERO**
2. **[README_LISTO.md](README_LISTO.md)** - Guía completa de uso
3. **[DEMO.md](DEMO.md)** - Paso a paso visual

---

## ⚡ INICIO RÁPIDO

### Para usar AHORA (sin esperar):
```bash
cd d:\ProyectosGRIT\ProyectoMaestría
./venv/Scripts/python.exe -m uvicorn app:app --host 127.0.0.1 --port 9000
```
Luego abre: http://127.0.0.1:9000

**Tiempo:** 10 segundos

---

## 📖 DOCUMENTACIÓN COMPLETA

### Conceptual (Explicaciones)
- [RESUMEN_FINAL.md](RESUMEN_FINAL.md) - Qué se hizo y por qué
- [PLAN_DE_ACCION.md](PLAN_DE_ACCION.md) - Análisis de opciones disponibles

### Práctico (Hacer cosas)
- [DEMO.md](DEMO.md) - Paso a paso clickeable
- [README_LISTO.md](README_LISTO.md) - Instrucciones detalladas
- [QUICKSTART.md](QUICKSTART.md) - Comandos principales

---

## 🔧 COMANDOS PRINCIPALES

| Comando | Propósito | Tiempo |
|---------|-----------|--------|
| `make server` | Iniciar API | 10s |
| `make train` | Fine-tuning | 2-5 min |
| `make train-cpu` | Fine-tuning en CPU | 5-10 min |
| `python run_training.py` | Entrenar personalizado | Variable |

---

## 📂 ARCHIVOS IMPORTANTES

### Código Principal
- `app.py` - API REST FastAPI
- `src/pipeline.py` - Orquestador principal
- `src/authenticity_validator.py` - **NUEVO:** Validación + IA
- `src/ai_detection.py` - **NUEVO:** Detección de IA/edición

### Web Interface
- `static/index.html` - UI responsive

### Scripts
- `run_training.py` - Ejecutar entrenamiento
- `prepare_dataset.py` - Preparar datos
- `verify_setup.py` - Verificar requisitos

### Datos
- `dataset_training/images/` - 44 imágenes para entrenar
- `models/trocr-finetuned/` - Modelo tras entrenar

---

## 🤔 PREGUNTAS FRECUENTES

### "¿Por dónde empiezo?"
1. Lee [RESUMEN_FINAL.md](RESUMEN_FINAL.md)
2. Ejecuta: `make server`
3. Abre: http://127.0.0.1:9000

### "¿Cómo entreno el modelo?"
```bash
python run_training.py --epochs 2
```
O usa: `make train`

### "¿Cuánto tarda?"
| Tarea | Tiempo |
|-------|--------|
| Servidor startup | 10 seg |
| Verificación imagen | 3-5 seg |
| Fine-tuning (2 épocas) | 2-5 min |

### "¿Qué detecta?"
✓ Pasaportes auténticos
✓ Pasaportes generados por IA (DALL-E, Midjourney, etc)
✓ Pasaportes editados manualmente (Photoshop, GIMP, etc)
✓ Razón específica de cada resultado

### "¿Necesito GPU?"
No requerido, pero hace más rápido
- GPU: < 1 segundo por imagen
- CPU: 3-5 segundos por imagen

---

## 🔍 EXPLORAR EL CÓDIGO

### Flujo Principal
```
user image → API (app.py)
        ↓
    preprocessing.py
        ↓
    ocr_engine.py (TrOCR)
        ↓
    mrz_validator.py (ICAO Doc 9303)
        ↓
    authenticity_validator.py ← ai_detection.py
        ↓
    decision (PASS/REVIEW/REJECT)
```

### Detección de IA
```
src/ai_detection.py
├── _analyze_histogram() → gaps = edición
├── _analyze_noise() → artificial vs natural
├── _analyze_lighting() → inconsistencia
├── _analyze_compression() → JPEG artifacts
└── _extract_exif() → software de edición
```

---

## 📊 EJEMPLOS DE RESULTADOS

### Auténtico
```json
{
  "estado": "PASS",
  "autenticidad_score": 92.5,
  "razon": "✓ AUTÉNTICO - Score: 92%"
}
```

### Falsificado por IA
```json
{
  "estado": "REJECT",
  "autenticidad_score": 8.5,
  "razon": "❌ FALSIFICADO POR IA - Confianza: 94%"
}
```

### Editado
```json
{
  "estado": "REJECT",
  "autenticidad_score": 35.2,
  "razon": "❌ FALSIFICADO - Imagen editada"
}
```

---

## 🛠️ SOLUCIÓN DE PROBLEMAS

### Puerto en uso
```bash
python -m uvicorn app:app --port 8080
```

### Out of Memory
```bash
python run_training.py --batch_size 2
```

### Módulos faltantes
```bash
./venv/Scripts/python.exe verify_setup.py
```

---

## 🌐 API REST

### Endpoint Principal
```
POST /verify-passport
Content-Type: multipart/form-data

Response:
{
  "id": "uuid",
  "timestamp": "ISO8601",
  "autenticidad_score": 92.5,
  "estado": "PASS",
  "razon": "...",
  "confianza": {...},
  "analisis": {...}
}
```

### Documentación Interactiva
```
http://127.0.0.1:9000/docs
```

---

## 📚 ARCHIVOS POR CATEGORÍA

### Documentación
- `RESUMEN_FINAL.md` - Overview completo
- `README_LISTO.md` - Guía de uso listo
- `DEMO.md` - Demo paso a paso
- `PLAN_DE_ACCION.md` - Decisiones y opciones
- `QUICKSTART.md` - Inicio rápido
- `INDEX.md` ← Estás aquí

### Código
- `src/preprocessing.py` - Normalización de imagen
- `src/ocr_engine.py` - TrOCR wrapper
- `src/mrz_validator.py` - Validación de MRZ
- `src/pipeline.py` - Orquestador
- `src/authenticity_validator.py` - Validación + IA
- `src/ai_detection.py` - Detección IA/edición
- `src/finetune_trocr_simple.py` - Fine-tuning
- `app.py` - API REST
- `config.py` - Configuración

### Scripts Ejecutables
- `run_training.py` - Entrenar modelo
- `prepare_dataset.py` - Preparar datos
- `verify_setup.py` - Verificar setup

### Datos
- `dataset_training/images/` - Dataset (44 imágenes)
- `models/` - Modelos guardados
- `archive/` - Datos originales

### Config
- `Makefile` - Comandos principales
- `requirements.txt` - Dependencias
- `config.py` - Parámetros

---

## ✅ CHECKLIST DE USO

- [ ] Leo [RESUMEN_FINAL.md](RESUMEN_FINAL.md)
- [ ] Ejecuto: `make server`
- [ ] Abro: http://127.0.0.1:9000
- [ ] Subo una imagen
- [ ] Veo los resultados
- [ ] (Opcional) Entreno: `make train`

---

## 🎯 SIGUIENTE PASO

**Elige una opción:**

### 1️⃣ Quiero ver cómo funciona AHORA
→ Lee [DEMO.md](DEMO.md)

### 2️⃣ Quiero instrucciones completas
→ Lee [README_LISTO.md](README_LISTO.md)

### 3️⃣ Quiero entender qué se hizo
→ Lee [RESUMEN_FINAL.md](RESUMEN_FINAL.md)

### 4️⃣ Quiero más detalles técnicos
→ Lee [PLAN_DE_ACCION.md](PLAN_DE_ACCION.md)

---

## 🚀 TL;DR

**Para empezar:**
```bash
cd d:\ProyectosGRIT\ProyectoMaestría
python -m uvicorn app:app --host 127.0.0.1 --port 9000
```

**Luego:** http://127.0.0.1:9000

**¡Listo!** 🎉
