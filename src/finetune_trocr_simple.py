#!/usr/bin/env python
"""
finetune_trocr_simple.py - Fine-tuning de TrOCR SIN dependencias problemáticas

Usa OpenCV + PyTorch directamente sin albumentations para evitar conflictos de numpy
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, get_linear_schedule_with_warmup
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimplePassportDataset(Dataset):
    """Dataset simple para imágenes de pasaportes"""
    
    def __init__(self, images_dir: str, processor: TrOCRProcessor, augment: bool = False):
        self.image_dir = Path(images_dir)
        self.processor = processor
        self.augment = augment
        
        # Buscar imágenes recursivamente
        self.images = list(self.image_dir.rglob("*.jpg")) + list(self.image_dir.rglob("*.png"))
        self.images += list(self.image_dir.rglob("*.JPG")) + list(self.image_dir.rglob("*.PNG"))
        self.images = list(set(self.images))  # Eliminar duplicados
        
        if len(self.images) == 0:
            raise ValueError(f"No se encontraron imágenes en {images_dir}")
        
        logger.info(f"✓ Dataset cargado: {len(self.images)} imágenes")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> dict:
        image_path = self.images[idx]
        
        try:
            # Cargar imagen con OpenCV
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"No se pudo leer: {image_path}")
                return self[(idx + 1) % len(self)]
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Augmentación simple
            if self.augment:
                image = self._augment_image(image)
            
            # Procesar con TrOCR
            pixel_values = self.processor(
                images=image,
                return_tensors="pt"
            ).pixel_values.squeeze(0)
            
            # Usar filename como label (en production sería anotaciones reales)
            label = str(image_path.stem)
            
            # Tokenizar
            labels = self.processor.tokenizer(
                label,
                padding="max_length",
                max_length=128,
                return_tensors="pt"
            )
            
            return {
                "pixel_values": pixel_values,
                "decoder_input_ids": labels.input_ids.squeeze(0),
                "labels": labels.input_ids.squeeze(0),
            }
        
        except Exception as e:
            logger.error(f"Error procesando {image_path}: {e}")
            return self[(idx + 1) % len(self)]
    
    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        """Augmentación simple de datos"""
        # Rotación pequeña
        if random.random() < 0.2:
            angle = random.randint(-5, 5)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
        
        # Flip horizontal
        if random.random() < 0.1:
            image = cv2.flip(image, 1)
        
        # Ruido Gaussiano
        if random.random() < 0.1:
            noise = np.random.normal(0, 10, image.shape)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        # Brightness/Contrast
        if random.random() < 0.2:
            alpha = random.uniform(0.9, 1.1)  # Contrast
            beta = random.randint(-10, 10)    # Brightness
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        return image


def train_trocr(
    dataset_path: str = "dataset_training/images",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    output_dir: str = "models/trocr-finetuned"
) -> None:
    """
    Entrenar TrOCR
    
    Args:
        dataset_path: Ruta al dataset
        epochs: Número de épocas
        batch_size: Tamaño del batch
        learning_rate: Learning rate
        output_dir: Directorio de salida
    """
    
    print("\n" + "="*70)
    print("🚀 ENTRENAMIENTO DE TROCR PARA PASAPORTES MEXICANOS")
    print("="*70)
    
    # Verificar CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n📱 Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Crear directorio de salida
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Cargar modelo y processor
    print("\n📥 Cargando modelo TrOCR base (microsoft/trocr-base-printed)...")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    model.to(device)
    
    # Crear dataset
    print(f"\n📂 Cargando dataset desde: {dataset_path}")
    dataset = SimplePassportDataset(dataset_path, processor, augment=True)
    
    # Split train/val
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )
    print(f"   Training: {n_train} | Validation: {n_val}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Scheduler
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    print("\n" + "="*70)
    print("🔥 INICIO DE ENTRENAMIENTO")
    print("="*70)
    
    best_val_loss = float('inf')
    patience = max(2, epochs // 2)
    patience_counter = 0
    
    for epoch in range(epochs):
        print(f"\n📋 Época {epoch+1}/{epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            decoder_input_ids = batch["decoder_input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                pixel_values=pixel_values,
                decoder_input_ids=decoder_input_ids,
                labels=labels
            )
            
            loss = outputs.loss
            train_loss += loss.item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Log
            if (step + 1) % max(1, len(train_loader) // 5) == 0:
                avg_loss = train_loss / (step + 1)
                print(f"  Batch {step+1}/{len(train_loader)} | Loss: {avg_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                decoder_input_ids = batch["decoder_input_ids"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(
                    pixel_values=pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\n  ✓ Training Loss:   {avg_train_loss:.4f}")
        print(f"  ✓ Validation Loss: {avg_val_loss:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print(f"  ✅ Nuevo mejor modelo guardado")
        else:
            patience_counter += 1
            print(f"  ⚠ Sin mejora ({patience_counter}/{patience})")
        
        if patience_counter >= patience:
            print(f"\n⏹ Early stopping en época {epoch+1}")
            break
    
    # Guardar modelo
    print("\n" + "="*70)
    print("💾 GUARDANDO MODELO ENTRENADO")
    print("="*70)
    
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"✓ Modelo guardado en: {output_dir}")
    
    print("\n" + "="*70)
    print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("="*70)
    print(f"\npróximo paso:")
    print(f"  1. Reinicia el servidor: make server")
    print(f"  2. El sistema usará automáticamente el modelo entrenado")
    print(f"  3. Verifica si OCR_BASE_MODEL en ocr_engine.py apunta a: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="dataset_training/images")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--output_dir", default="models/trocr-finetuned")
    
    args = parser.parse_args()
    
    train_trocr(
        dataset_path=args.dataset_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir
    )
