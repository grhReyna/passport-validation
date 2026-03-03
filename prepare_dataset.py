#!/usr/bin/env python
"""
prepare_dataset.py - Preparar dataset plano para entrenamiento
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

def flatten_dataset():
    """Crear dataset plano a partir de la estructura profunda"""
    
    root_path = Path("archive/MEX/G77536498/L2")
    output_path = Path("dataset_training/images")
    
    print("\n📁 PREPARANDO DATASET PARA ENTRENAMIENTO")
    print("="*60)
    
    # Crear directorio de salida
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Contar imágenes recursivamente
    print(f"Buscando imágenes en: {root_path}")
    print("Esto puede tomar un poco de tiempo...")
    
    all_images = list(root_path.rglob("*.jpg")) + list(root_path.rglob("*.png")) + \
                 list(root_path.rglob("*.JPG")) + list(root_path.rglob("*.PNG"))
    
    print(f"\n✓ Total de imágenes encontradas: {len(all_images)}")
    
    if len(all_images) == 0:
        print("❌ No se encontraron imágenes")
        return False
    
    # Copiar imágenes
    print(f"\n📋 Copiando imágenes a {output_path}...")
    copied = 0
    skip = 0
    
    for i, img_path in enumerate(tqdm(all_images, desc="Copiando")):
        try:
            # Crear nombre único para evitar sobrescrituras
            dst_name = f"{i:04d}_{img_path.stem}.jpg"
            dst_path = output_path / dst_name
            
            # Copiar archivo
            shutil.copy2(img_path, dst_path)
            copied += 1
            
        except Exception as e:
            print(f"  ⚠ Error copiando {img_path}: {e}")
            skip += 1
    
    print(f"\n✓ Copiadas: {copied}")
    if skip > 0:
        print(f"⚠ Saltadas: {skip}")
    
    # Verificar
    final_images = list(output_path.glob("*.jpg")) + list(output_path.glob("*.png"))
    
    print("\n" + "="*60)
    print(f"✅ Dataset preparado: {len(final_images)} imágenes")
    print(f"📁 Ubicación: {output_path}")
    print("="*60)
    
    return len(final_images) > 100

if __name__ == "__main__":
    import sys
    success = flatten_dataset()
    sys.exit(0 if success else 1)
