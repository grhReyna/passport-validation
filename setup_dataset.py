#!/usr/bin/env python
"""
setup_dataset.py - Script para descargar y preparar el dataset

Este script:
1. Descarga el dataset de Kaggle (Synthetic Printed Mexican Passports)
2. Organiza las imágenes en directorios
3. Crea splits de train/validation/test
4. Genera un resumen estadístico

Usage:
    python setup_dataset.py
"""

import os
import sys
from pathlib import Path
import shutil
from tqdm import tqdm
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent))

import config

def check_kaggle_setup():
    """Verificar que Kaggle esté configurado"""
    logger.info("Verificando configuración de Kaggle...")
    
    try:
        import kagglehub
        logger.info("✓ kagglehub instalado correctamente")
        return True
    except ImportError:
        logger.error("✗ kagglehub no está instalado")
        logger.error("  Instala con: pip install kagglehub")
        return False

def download_dataset():
    """Descargar dataset de Kaggle"""
    logger.info("\n" + "=" * 60)
    logger.info("DESCARGANDO DATASET")
    logger.info("=" * 60)
    
    try:
        import kagglehub
        
        logger.info("Descargando: Synthetic Printed Mexican Passports")
        logger.info("Dataset: unidpro/synthetic-printed-mexican-passports")
        
        path = kagglehub.dataset_download("unidpro/synthetic-printed-mexican-passports")
        
        logger.info(f"✓ Dataset descargado en: {path}")
        return path
    
    except Exception as e:
        logger.error(f"✗ Error descargando dataset: {str(e)}")
        logger.error("  Asegúrate de tener credenciales de Kaggle configuradas")
        logger.error("  Ver: https://www.kaggle.com/settings/account")
        return None

def organize_dataset(source_path):
    """Organizar dataset en directorios estructura"""
    logger.info("\n" + "=" * 60)
    logger.info("ORGANIZANDO DATASET")
    logger.info("=" * 60)
    
    try:
        source = Path(source_path)
        
        if not source.exists():
            logger.error(f"✗ Ruta origen no existe: {source}")
            return False
        
        # Crear directorios destino
        raw_path = config.DATA_RAW_PATH
        raw_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Copiando archivos a {raw_path}...")
        
        # Copiar todos los archivos
        image_count = 0
        for file_path in source.glob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                try:
                    shutil.copy2(file_path, raw_path / file_path.name)
                    image_count += 1
                except Exception as e:
                    logger.warning(f"No se pudo copiar {file_path.name}: {str(e)}")
        
        logger.info(f"✓ {image_count} imágenes copiadas")
        
        # Crear subdirectorios para splits
        for split in ['train', 'validation', 'test']:
            split_dir = raw_path / split
            split_dir.mkdir(exist_ok=True)
            logger.info(f"  Creado: {split_dir}")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Error organizando dataset: {str(e)}")
        return False

def create_splits(raw_path):
    """Crear splits de train/validation/test"""
    logger.info("\n" + "=" * 60)
    logger.info("CREANDO SPLITS")
    logger.info("=" * 60)
    
    try:
        import random
        
        # Obtener lista de imágenes
        images = [f for f in (raw_path).iterdir() 
                 if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        if not images:
            logger.warning("No se encontraron imágenes en el dataset")
            return False
        
        logger.info(f"Total de imágenes: {len(images)}")
        
        # Shuffle
        random.seed(config.RANDOM_SEED)
        random.shuffle(images)
        
        # Calcular índices
        train_size = int(len(images) * config.TRAIN_SPLIT)
        val_size = int(len(images) * config.VAL_SPLIT)
        
        train_images = images[:train_size]
        val_images = images[train_size:train_size+val_size]
        test_images = images[train_size+val_size:]
        
        # Mover archivos a subdirectorios
        splits = {
            'train': train_images,
            'validation': val_images,
            'test': test_images
        }
        
        for split_name, split_images in splits.items():
            split_dir = raw_path / split_name
            logger.info(f"\nMoviendo {len(split_images)} imágenes a {split_name}/...")
            
            for img in tqdm(split_images):
                try:
                    shutil.move(str(img), str(split_dir / img.name))
                except Exception as e:
                    logger.warning(f"No se pudo mover {img.name}: {str(e)}")
        
        logger.info("\n✓ Splits creados correctamente:")
        logger.info(f"  Train:      {len(train_images)} imágenes")
        logger.info(f"  Validation: {len(val_images)} imágenes")
        logger.info(f"  Test:       {len(test_images)} imágenes")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Error creando splits: {str(e)}")
        return False

def print_summary():
    """Imprimir resumen del dataset"""
    logger.info("\n" + "=" * 60)
    logger.info("RESUMEN DEL DATASET")
    logger.info("=" * 60)
    
    for split in ['train', 'validation', 'test']:
        split_dir = config.DATA_RAW_PATH / split
        if split_dir.exists():
            count = len(list(split_dir.glob("*.*")))
            logger.info(f"{split:12} | {count:5} imágenes")
    
    logger.info("\n" + "=" * 60)
    logger.info("PRÓXIMOS PASOS")
    logger.info("=" * 60)
    logger.info("1. Revisar dataset en data/raw/")
    logger.info("2. Ejecutar: jupyter notebook notebooks/01_exploracion_dataset.ipynb")
    logger.info("3. Implementar módulos en src/")
    logger.info("4. Ejecutar tests: pytest tests/")
    logger.info("\nVer PLAN_DETECCION_PASAPORTES.md para más detalles")

def main():
    """Función principal"""
    logger.info("\n" + "=" * 60)
    logger.info("SETUP: Sistema de Detección de Pasaportes Falsos")
    logger.info("=" * 60)
    
    # 1. Verificar Kaggle
    if not check_kaggle_setup():
        sys.exit(1)
    
    # 2. Descargar dataset
    source_path = download_dataset()
    if source_path is None:
        sys.exit(1)
    
    # 3. Organizar dataset
    if not organize_dataset(source_path):
        sys.exit(1)
    
    # 4. Crear splits
    if not create_splits(config.DATA_RAW_PATH):
        logger.warning("No se pudieron crear splits automáticamente")
    
    # 5. Imprimir resumen
    print_summary()
    
    logger.info("\n✓ Setup completado exitosamente!\n")

if __name__ == "__main__":
    main()
