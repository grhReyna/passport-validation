# 🎯 Sistema de Detección de Pasaportes Falsos

**Proyecto de Maestría - Seminario de Innovación**  
**Fecha:** Febrero 2026  
**Status:** ✅ 100% Funcional 🚀

---

## 📋 Descripción

Sistema de IA que verifica autenticidad de pasaportes mexicanos:

✅ **Detecta pasaportes auténticos** - OCR + MRZ validation
✅ **Detecta falsificaciones por IA** - DALL-E, Midjourney, Stable Diffusion  
✅ **Detecta ediciones manuales** - Photoshop, GIMP, Lightroom
✅ **Genera score (0-100%)** - PASS / REVIEW / REJECT
✅ **Análisis detallado** - Ruido, iluminación, metadata EXIF

---

## 🎯 Características

- 🔍 OCR avanzado con TrOCR (Transformers)
- 📋 Validación MRZ según ICAO Doc 9303
- 🤖 Detección de IA generativa
- ✏️ Detección de edición manual
- 📊 Score ponderado (OCR 40% + MRZ 60%)
- 💻 Web UI responsive
- 🔌 API REST FastAPI
- ⚡ Fine-tuning disponible 

---

## 🏗️ Arquitectura

```
Imagen → Preprocessing → OCR (TrOCR) → MRZ Validator
                                  ↓
         AI Detector ← Authenticity Validator
                                  ↓
                        Score + Decision
                       (PASS/REVIEW/REJECT)
```

---

## 📁 Estructura

```
ProyectoMaestría/
├── src/
│   ├── preprocessing.py
│   ├── ocr_engine.py
│   ├── mrz_validator.py
│   ├── pipeline.py
│   ├── authenticity_validator.py    ← AI Detection
│   └── ai_detection.py              ← New
├── app.py                 ← API FastAPI
├── config.py
├── run_training.py        ← Fine-tuning
├── static/index.html      ← Web UI
├── dataset_training/images/   ← 44 imágenes
├── models/
├── INDEX.md               ← Documentación
└── README_LISTO.md
```

---

## 🚀 Uso

### ⚡ Empezar (10 segundos)
```bash
cd d:\ProyectosGRIT\ProyectoMaestría
make server
```
Luego: **http://127.0.0.1:9000**

### (Opcional) Fine-tuning
```bash
make train-cpu
```

---

## � API

```
POST /verify-passport
Content-Type: multipart/form-data

Response:
{
  "autenticidad_score": 92.5,
  "estado": "PASS",
  "razon": "✓ AUTÉNTICO - Score: 92%",
  "analisis": {
    "ai_analysis": {
      "is_ai_generated": false,
      "is_edited": false,
      "confidence": 0.98
    }
  }
}
```

---

## 📊 Decisiones

| Score | Estado | Acción |
|-------|--------|--------|
| ≥ 90% | PASS | ✓ Aceptar |
| 70-89% | REVIEW | ⚠ Manual |
| < 70% | REJECT | ✗ Rechazar |

**IA Detectada → REJECT INMEDIATO**

---

## ⚙️ Stack

- **OCR:** TrOCR (microsoft/trocr-base-printed)
- **ML:** PyTorch 2.0 + Transformers 4.35
- **API:** FastAPI + Uvicorn
- **Preprocessing:** OpenCV 4.8
- **UI:** Bootstrap 5

### 5. Pipeline (`src/pipeline.py`)
- Orquestación de componentes
- Manejo de errores
- Logging y trazabilidad

---

## � Archivos principales

- `app.py` - API REST
- `config.py` - Configuración
- `run_training.py` - Fine-tuning
- `src/pipeline.py` - Core
- `INDEX.md` - Documentación central

---

## 📚 Referencias

- ICAO Doc 9303 - Machine-Readable Travel Documents
- TrOCR - Transformer-based OCR (Microsoft)
- Kaggle Dataset - Synthetic Mexican Passports

---

**Proyecto de Maestría - 2026** 🎓
