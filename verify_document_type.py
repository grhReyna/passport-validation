#!/usr/bin/env python3
"""
Verifica qué tipo de documento es la imagen
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path

img_path = Path(r"D:\papeleria\passport.jpg")

if not img_path.exists():
    print(f"ERROR: Archivo no existe: {img_path}")
    exit(1)

print(f"\n{'='*80}")
print(f"ANALISIS DE IMAGEN: {img_path}")
print(f"{'='*80}\n")

# Cargar imagen
img = cv2.imread(str(img_path))
pil_img = Image.open(img_path)

print(f"[*] Formato PIL: {pil_img.format}")
print(f"[*] Modo: {pil_img.mode}")
print(f"[*] Tamaño: {pil_img.size[0]}x{pil_img.size[1]}")
print(f"[*] Tamaño archivo: {img_path.stat().st_size / 1024:.1f} KB")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(f"\n[*] CARACTERÍSTICAS:")
print(f"    Brillo promedio: {gray.mean():.1f}")
print(f"    Contraste (std): {gray.std():.1f}")

# Detectar orientación
h, w = img.shape[:2]
aspect = h/w if w > 0 else 1
print(f"    Aspecto: {aspect:.2f} ({'VERTICAL' if aspect > 1 else 'HORIZONTAL'})")

# Detectar si parece ser pasaporte (documento con estructura específica)
# Pasaportes típicamente tienen:
# - Color azul/verde oscuro
# - Texto en MAYUSCULA
# - Fotografía en región específica
# - Números en región específica

# Analizar colores
b, g, r = cv2.split(img)
print(f"\n[*] COLORES (promedio):")
print(f"    Rojo: {r.mean():.0f}")
print(f"    Verde: {g.mean():.0f}")
print(f"    Azul: {b.mean():.0f}")

# Si R <80 y G < 100 y B > 120 → Probablemente AZUL (típico de pasaportes)
# Si R+G > 150 → Probablemente CLARO (documento administrativo)

es_oscuro = (r.mean() < 100 and g.mean() < 100 and b.mean() > 120)
es_claro = (r.mean() > 150 or g.mean() > 150)

print(f"\n[*] DETECCION DE TIPO:")
print(f"    Es documento oscuro (típico pasaporte): {es_oscuro}")
print(f"    Es documento claro (típico factura/recibo): {es_claro}")

# Aplicar OCR rápido para ver palabras comunes
from transformers import ViTImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
import torch

processor = ViTImageProcessor.from_pretrained("microsoft/trocr-base-printed")
tokenizer = AutoTokenizer.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

# Redimensionar a tamaño del modelo
resized = cv2.resize(img, (512, 320))
resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
pil_resized = Image.fromarray(resized_rgb)

pixel_values = processor(pil_resized, return_tensors="pt").pixel_values

with torch.no_grad():
    generated_ids = model.generate(pixel_values, max_length=256)

text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(f"\n[*] OCR DETECTADO (primeros 100 chars):")
print(f"    '{text[:100]}'")

# Detectar palabras clave
keywords_pasaporte = ['pasaporte', 'mexico', 'numero', 'feha', 'expedicion', 'nacimiento']
keywords_factura = ['tax', 'item', 'amount', 'price', 'total', 'invoice', 'receipt']

matches_passport = sum(1 for kw in keywords_pasaporte if kw in text.lower())
matches_factura = sum(1 for kw in keywords_factura if kw in text.lower())

print(f"\n[*] PALABRAS CLAVE:")
print(f"    Coincidencias Pasaporte: {matches_pasaporte}")
print(f"    Coincidencias Factura: {matches_factura}")

if matches_factura > 0:
    print(f"\n[!!!] ADVERTENCIA: Parece ser una FACTURA o RECIBO, no un pasaporte")
elif es_oscuro:
    print(f"\n[OK] Parece ser un pasaporte (documento oscuro)")
else:
    print(f"\n[?] Documento unclear - verificar manualmente")

print(f"\n{'='*80}\n")
