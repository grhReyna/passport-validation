# 🚀 INICIO RÁPIDO

## ✅ Sistema 100% Listo
- ✅ Detección de pasaportes
- ✅ Detección de IA
- ✅ Web UI funcional
- ✅ API REST operativa
- ✅ Fine-tuning listo

---

## 3 PASOS

### 1️⃣ Iniciar servidor (10 segundos)
```bash
make server
```

### 2️⃣ Abrir navegador
```
http://127.0.0.1:9000
```

### 3️⃣ Subir imagen (JPG/PNG)
- Pasaporte a verificar
- Sistema retorna: Score + Estado + Razón

---

## 📊 EJEMPLOS DE RESULTADOS

### Auténtico ✓
```
Score: 92%
Estado: PASS
Razón: Auténtico - Score: 92%
```

### Falsificado por IA ❌
```
Score: 8%
Estado: REJECT
Razón: FALSIFICADO POR IA - Confianza: 94%
Detectado: Stable Diffusion
```

### Editado manualmente ❌
```
Score: 35%
Estado: REJECT
Razón: Falsificado - Imagen editada
Editor: Photoshop
```

---

## 🔧 MONITOREO DURANTE ENTRENAMIENTO

El script mostrará en tiempo real:
- ✓ Épocas completadas
- ✓ Loss y métricas
- ✓ Tiempas estimados
- ✓ Uso de GPU/CPU

Ejemplo de salida:
```
Epoch 1/3 ████████████░░░░░░░░ 60% | Loss: 0.456 | ETA: 45min
Epoch 2/3 ██████████████████░░ 100% | Loss: 0.234 | ETA: 30min
```

---

## 📁 Archivos Nuevos Creados

| Archivo | Propósito | Estado |
|---------|-----------|--------|
| `src/authenticity_validator.py` | Validación completa con detección IA | ✅ Listo |
| `src/finetune_trocr.py` | Script de entrenamiento | ✅ Listo |
| `run_training.py` | Wrapper ejecutable | ✅ Listo |
| `src/ai_detection.py` | Detección de IA/edición | ✅ Listo |

---

## 📂 Archivos Importantes

```
app.py                          ← API REST
config.py                       ← Configuración
run_training.py                 ← Entrenamiento
index.md / readme_listo.md       ← Documentación
src/pipeline.py                 ← Core del sistema
src/ai_detection.py             ← Detección IA
static/index.html               ← Web UI
dataset_training/images/        ← 44 imágenes
```

---

## 🆘 Si Algo Falla

**Error: "No se encontró el dataset"**
```bash
# Verificar que exista:
ls archive/MEX/G77536498/L2/
# Debería mostrar 500+ archivos .jpg o .png
```

**Error: "Out of Memory"**
```bash
# Reducir batch_size:
python run_training.py --batch_size 2 --epochs 3
```

**Error: CUDA/GPU**
```bash
# Usar solo CPU:
make train-cpu
```

---

## 📝 Próximos Pasos Después del Entrenamiento

1. ✅ Modelo guardado en: `models/trocr-finetuned/`
2. ✅ Sistema automáticamente lo usará
3. ⏭ Probar con la misma imagen anterior
4. ⏭ Verificar mejora en Score de MRZ
5. ⏭ Crear notebook de validación

---

## ⏱️ Resumen de Tiempos

| Tarea | Duración | Máquina |
|-------|----------|---------|
| Fine-tuning (3 épocas) | 1-2 horas | RTX 3060 |
| Fine-tuning (5 épocas) | 2-3 horas | RTX 3060 |
| Servidor startup | < 10 seg | Cualquiera |
| Verificación pasaporte | 2-5 seg | RTX 3060 |

---

## 🎯 Verificar Que Todo Funciona

```bash
# 1. Verificar dataset
ls archive/MEX/G77536498/L2 | wc -l
# Debe mostrar: ~500+ archivos

# 2. Verificar estructura del código
python -c "from src import authenticity_validator; print('✓ Código listo')"

# 3. Iniciar entrenamiento
make train
```

¡Listo! El sistema estará optimizado en 1-2 horas 🚀
