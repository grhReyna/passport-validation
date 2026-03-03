#!/usr/bin/env python
"""
run_training.py - Script para ejecutar el entrenamiento del modelo TrOCR

Uso:
    python run_training.py
    python run_training.py --dataset_path dataset_training/images --epochs 5 --batch_size 8
"""

import sys
import os
import argparse
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from finetune_trocr_simple import train_trocr


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Entrenar modelo TrOCR para detección de MRZ en pasaportes mexicanos"
    )
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="dataset_training/images",
        help="Ruta al dataset de imágenes (default: dataset_training/images)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Número de épocas (default: 3)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Tamaño del batch (default: 4)"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/trocr-finetuned",
        help="Directorio de salida (default: models/trocr-finetuned)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 ENTRENAMIENTO DE MODELO TROCR")
    print("="*70)
    print(f"Dataset:       {args.dataset_path}")
    print(f"Épocas:        {args.epochs}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Output dir:    {args.output_dir}")
    print("="*70 + "\n")
    
    try:
        train_trocr(
            dataset_path=args.dataset_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir
        )
        print("\n" + "="*70)
        print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("="*70)
        print(f"Modelo guardado en: {args.output_dir}")
        print("\nPróximo paso: Ejecutar app.py para usar el modelo entrenado")
        print("             El sistema automáticamente lo usará.\n")
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ ERROR EN ENTRENAMIENTO")
        print("="*70)
        print(f"Error: {str(e)}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
