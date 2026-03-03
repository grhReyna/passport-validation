"""
finetune_trocr.py - Fine-tunear TrOCR con imágenes del dataset

Entrena TrOCR preentrenado con imágenes reales de pasaportes mexicanos
para mejorar precisión en reconocimiento de MRZ.

Uso:
    python finetune_trocr.py --dataset_path <ruta> --epochs 3 --batch_size 4
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Tuple, List
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import default_data_collator, AdamW, get_linear_schedule_with_warmup
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PassportMRZDataset(Dataset):
    """Dataset de imágenes de pasaportes para entrenamiento de MRZ"""
    
    def __init__(self, image_dir: Path, processor: TrOCRProcessor, augment: bool = False):
        """
        Inicializar dataset
        
        Args:
            image_dir: Directorio con imágenes
            processor: TrOCRProcessor de Hugging Face
            augment: Aplicar augmentación de datos
        """
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.augment = augment
        
        # Buscar todas las imágenes
        self.images = list(self.image_dir.glob('**/*.jpg')) + list(self.image_dir.glob('**/*.png'))
        
        logger.info(f"Dataset cargado: {len(self.images)} imágenes")
        
        # Augmentación de datos
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.1),
            A.Rotate(limit=5, p=0.2),
            A.GaussNoise(p=0.1),
            A.OneOf([
                A.MotionBlur(p=0.1),
                A.GaussianBlur(p=0.1),
            ], p=0.1),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(),
            ToTensorV2()
        ]) if augment else None
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Obtener item del dataset
        
        Returns:
            dict: pixel_values, labels, decoder_input_ids
        """
        image_path = self.images[idx]
        
        try:
            # Cargar imagen
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            
            # Augmentación
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            
            # Procesar con TrOCR
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            
            # Extraer nombre del archivo como texto de referencia
            # En producción, esto vendría de anotaciones
            labels = str(image_path.stem)
            
            # Procesar texto
            encoding = self.processor.tokenizer(
                labels,
                padding="max_length",
                max_length=128,
                return_tensors="pt"
            )
            
            return {
                "pixel_values": pixel_values.squeeze(),
                "decoder_input_ids": encoding.input_ids.squeeze(),
                "labels": encoding.input_ids.squeeze(),
            }
        
        except Exception as e:
            logger.error(f"Error procesando {image_path}: {str(e)}")
            # Retornar item siguiente en caso de error
            return self[idx + 1] if idx + 1 < len(self) else self[0]


def train_trocr(
    dataset_path: str,
    output_dir: str = "models/trocr-finetuned",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    device: str = "auto"
) -> None:
    """
    Fine-tunear TrOCR
    
    Args:
        dataset_path: Ruta a dataset
        output_dir: Directorio de salida para modelo
        epochs: Número de épocas
        batch_size: Tamaño del lote
        learning_rate: Tasa de aprendizaje
        device: 'cuda', 'cpu' o 'auto'
    """
    
    # Verificar GPU
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("=" * 70)
    logger.info("FINE-TUNING TrOCR PARA PASAPORTES MEXICANOS")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    
    # Cargar modelo preentrenado
    logger.info("Cargando modelo TrOCR preentrenado...")
    model_name = "microsoft/trocr-base-printed"
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.to(device)
    
    # Crear dataset
    logger.info(f"Cargando dataset desde {dataset_path}...")
    dataset = PassportMRZDataset(dataset_path, processor, augment=True)
    
    if len(dataset) == 0:
        logger.error("¡Dataset vacío! No hay imágenes para entrenar.")
        return
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=default_data_collator
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=default_data_collator
    )
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    # Entrenar
    logger.info("\nIniciando entrenamiento...")
    logger.info("=" * 70)
    
    for epoch in range(epochs):
        logger.info(f"\nEpoca {epoch + 1}/{epochs}")
        
        # Entrenamiento
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
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
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 5 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                logger.info(f"  Batch {batch_idx + 1}/{len(train_loader)} - Loss: {avg_loss:.4f}")
        
        # Validación
        model.eval()
        val_loss = 0
        
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
        
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Validation Loss: {avg_val_loss:.4f}")
    
    # Guardar modelo
    logger.info("\n" + "=" * 70)
    logger.info("Entrenamiento completado")
    logger.info("=" * 70)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(output_path / "model")
    processor.save_pretrained(output_path / "processor")
    
    logger.info(f"Modelo guardado en: {output_path}")
    logger.info("\n✓ Fine-tuning completado exitosamente!")


def main():
    parser = argparse.ArgumentParser(description="Fine-tunear TrOCR para pasaportes")
    parser.add_argument("--dataset_path", default="archive/MEX/G77536498/L2", help="Ruta al dataset")
    parser.add_argument("--output_dir", default="models/trocr-finetuned", help="Directorio de salida")
    parser.add_argument("--epochs", type=int, default=3, help="Número de épocas")
    parser.add_argument("--batch_size", type=int, default=4, help="Tamaño del lote")
    parser.add_argument("--lr", type=float, default=5e-5, help="Tasa de aprendizaje")
    parser.add_argument("--device", default="auto", help="Device (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    # Verificar dataset
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        logger.error(f"Dataset no encontrado: {dataset_path}")
        sys.exit(1)
    
    train_trocr(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device
    )


if __name__ == "__main__":
    main()
