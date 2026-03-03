"""
synthetic_passport_generator.py - Generar pasaportes sintéticos para testing

Genera imágenes de pasaportes mexicanos con datos válidos:
- Incluye MRZ válidos con checksums correctos
- Datos coherentes (nombres, fechas, números de pasaporte)
- Imágenes realistas con ruido y variaciones
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime, timedelta
import random
import string
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MexicanPassportGenerator:
    """Generador de pasaportes mexicanos sintéticos"""
    
    # Caracteres válidos según ICAO
    MRZ_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # Pesos para checksum mod-10
    CHECKSUM_WEIGHTS = [7, 3, 1]
    
    # Caracteres a números para checksum
    CHAR_TO_NUM = {
        'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17,
        'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25,
        'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33,
        'Y': 34, 'Z': 35
    }
    
    def __init__(self):
        """Inicializar generador"""
        self.names_db = {
            'surnames': [
                'GARCÍA', 'RODRÍGUEZ', 'MARTÍNEZ', 'HERNÁNDEZ', 'LOPEZ',
                'GONZALEZ', 'PEREZ', 'SANCHEZ', 'RAMIREZ', 'TORRES',
                'CRUZ', 'MORALES', 'GUTIERREZ', 'REYES', 'DIAZ'
            ],
            'names': [
                'JUAN', 'JOSÉ', 'CARLOS', 'MANUEL', 'LUIS',
                'FRANCISCO', 'ALEJANDRO', 'MARIO', 'SERGIO', 'JORGE',
                'MARÍA', 'ANA', 'ROSA', 'CARMEN', 'TERESA'
            ]
        }
    
    @staticmethod
    def calculate_checksum(data: str) -> str:
        """
        Calcular checksum ICAO mod-10
        
        Args:
            data: Cadena de caracteres
            
        Returns:
            Dígito checksum (0-9)
        """
        total = 0
        for i, char in enumerate(data):
            # Convertir a número
            if char.isdigit():
                value = int(char)
            else:
                value = MexicanPassportGenerator.CHAR_TO_NUM.get(char, 0)
            
            # Aplicar peso cíclico
            weight = MexicanPassportGenerator.CHECKSUM_WEIGHTS[i % 3]
            total += (value * weight) % 10
        
        return str(total % 10)
    
    def generate_valid_mrz(self) -> tuple[str, dict]:
        """
        Generar MRZ válido con datos coherentes
        
        Returns:
            (mrz_text, metadata_dict): Texto MRZ de 2 líneas y metadatos
        """
        # Generar datos
        surname = random.choice(self.names_db['surnames'])
        given_name = random.choice(self.names_db['names'])
        
        # Número de pasaporte (6 dígitos + letra)
        passport_num = f"{random.randint(100000, 999999)}{''.join(random.choices(string.ascii_uppercase, k=1))}"
        
        # Fecha de emisión (hace 1-10 años)
        issue_days_ago = random.randint(365, 3650)
        issue_date = datetime.now() - timedelta(days=issue_days_ago)
        issue_date_str = issue_date.strftime("%y%m%d")
        
        # Fecha de expiración (10 años después de emisión)
        expiry_date = issue_date + timedelta(days=3650)
        expiry_date_str = expiry_date.strftime("%y%m%d")
        
        # Fecha de nacimiento (18-80 años atrás)
        birth_days_ago = random.randint(365*18, 365*80)
        birth_date = datetime.now() - timedelta(days=birth_days_ago)
        birth_date_str = birth_date.strftime("%y%m%d")
        
        # Género
        gender = random.choice(['M', 'F'])
        
        # Nacionalidad (México = MEX)
        nationality = 'MEX'
        
        # Línea 1: P<MEXICOSURNAME<<GIVENNAME
        line1_base = f"P<{nationality}{surname}<<{given_name}"
        # Completar con < hasta 44 caracteres
        line1 = line1_base + '<' * (44 - len(line1_base))
        
        # Línea 2: PASSPORTNUMCHECK<YEARMONTHDAYCHECK_YEARMONTHDAYCHECK_GENDERCHECK
        check_passport = self.calculate_checksum(passport_num)
        check_date1 = self.calculate_checksum(birth_date_str)
        check_date2 = self.calculate_checksum(expiry_date_str)
        
        # Composite check (todos los checksums anteriores)
        composite_data = passport_num + check_passport + birth_date_str + check_date1 + expiry_date_str + check_date2 + gender
        check_composite = self.calculate_checksum(composite_data)
        
        # Formato correcto: PASSPORTCHECK_BIRTHDATECHECK_EXPIRYDATECHECK_GENDERCHECK<.........CHECK_
        line2 = f"{passport_num}{check_passport}{birth_date_str}{check_date1}{expiry_date_str}{check_date2}{gender}<<<<<<<<<<{check_composite}"
        
        mrz_text = f"{line1}\n{line2}"
        
        metadata = {
            'surname': surname,
            'given_name': given_name,
            'passport_number': passport_num,
            'birth_date': birth_date_str,
            'issue_date': issue_date_str,
            'expiry_date': expiry_date_str,
            'gender': gender,
            'nationality': nationality
        }
        
        return mrz_text, metadata
    
    def generate_passport_image(self, width: int = 1200, height: int = 750) -> Image.Image:
        """
        Generar imagen de pasaporte sintética
        
        Args:
            width: Ancho de imagen en píxeles
            height: Alto de imagen en píxeles
            
        Returns:
            Imagen PIL del pasaporte
        """
        # Crear imagen base (color azul como pasaportes mexicanos)
        img = Image.new('RGB', (width, height), color=(25, 55, 109))
        draw = ImageDraw.Draw(img)
        
        # Generar MRZ válido
        mrz_text, metadata = self.generate_valid_mrz()
        
        # Dibujar elementos en la imagen
        # Título
        draw.text((50, 30), "PASSAPORTE", fill=(255, 255, 255))
        draw.text((50, 60), "MÉXICO", fill=(255, 215, 0))
        
        # Datos personales
        y_pos = 120
        draw.text((50, y_pos), f"APELLIDO: {metadata['surname']}", fill=(255, 255, 255))
        draw.text((50, y_pos + 40), f"NOMBRE: {metadata['given_name']}", fill=(255, 255, 255))
        draw.text((50, y_pos + 80), f"PASAPORTE: {metadata['passport_number']}", fill=(255, 255, 255))
        draw.text((50, y_pos + 120), f"GÉNERO: {metadata['gender']}", fill=(255, 255, 255))
        
        # Fechas
        y_pos = 350
        draw.text((50, y_pos), f"NACIMIENTO: {metadata['birth_date']}", fill=(255, 255, 255))
        draw.text((50, y_pos + 40), f"EXPEDICIÓN: {metadata['issue_date']}", fill=(255, 255, 255))
        draw.text((50, y_pos + 80), f"VENCIMIENTO: {metadata['expiry_date']}", fill=(255, 255, 255))
        
        # MRZ (zona de lectura mecánica)
        y_pos = 550
        mrz_lines = mrz_text.split('\n')
        draw.text((50, y_pos), mrz_lines[0], fill=(0, 0, 0), font=None)
        draw.text((50, y_pos + 35), mrz_lines[1], fill=(0, 0, 0), font=None)
        
        # Añadir ruido realista
        img_array = np.array(img)
        
        # Ruido gaussiano suave
        noise = np.random.normal(0, 5, img_array.shape).astype(np.uint8)
        img_array = np.clip(img_array.astype(int) + noise.astype(int), 0, 255).astype(np.uint8)
        
        # Variación de brillo
        brightness_factor = random.uniform(0.95, 1.05)
        img_array = np.clip(img_array.astype(float) * brightness_factor, 0, 255).astype(np.uint8)
        
        img = Image.fromarray(img_array)
        
        return img, metadata, mrz_text


def generate_passport_dataset(output_dir: str, num_images: int = 10, split: dict = None):
    """
    Generar dataset de pasaportes sintéticos
    
    Args:
        output_dir: Directorio de salida
        num_images: Número total de imágenes a generar
        split: Dict con splits {'train': 0.7, 'validation': 0.15, 'test': 0.15}
    """
    if split is None:
        split = {'train': 0.7, 'validation': 0.15, 'test': 0.15}
    
    generator = MexicanPassportGenerator()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Crear subdirectorios
    for split_name in split.keys():
        (output_path / split_name).mkdir(exist_ok=True)
    
    logger.info(f"Generando {num_images} pasaportes sintéticos...")
    
    # Calcular número de imágenes por split
    split_counts = {
        name: int(num_images * ratio)
        for name, ratio in split.items()
    }
    
    # Ajustar para que sume exactamente num_images
    split_counts['train'] += num_images - sum(split_counts.values())
    
    generated_count = 0
    metadata_list = []
    
    for split_name, count in split_counts.items():
        for i in range(count):
            try:
                # Generar imagen
                img, metadata, mrz = generator.generate_passport_image()
                
                # Número de archivo único
                file_num = generated_count + 1
                filename = f"passport_{file_num:04d}.png"
                filepath = output_path / split_name / filename
                
                # Guardar imagen
                img.save(filepath)
                
                # Guardar metadatos
                metadata['mrz'] = mrz
                metadata['filename'] = filename
                metadata['split'] = split_name
                metadata_list.append(metadata)
                
                generated_count += 1
                
                if generated_count % 10 == 0:
                    logger.info(f"  Generados {generated_count}/{num_images} pasaportes")
            
            except Exception as e:
                logger.error(f"Error generando pasaporte {generated_count}: {str(e)}")
    
    logger.info(f"✓ Generados {generated_count} pasaportes en {output_path}")
    logger.info(f"  - train: {split_counts['train']}")
    logger.info(f"  - validation: {split_counts['validation']}")
    logger.info(f"  - test: {split_counts['test']}")
    
    return metadata_list


if __name__ == "__main__":
    # Generar 30 pasaportes de prueba
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    output_dir = Path(__file__).parent.parent / "data" / "raw"
    metadata = generate_passport_dataset(str(output_dir), num_images=30)
    
    print(f"\n✓ Dataset generado exitosamente!")
    print(f"  Archivos en: {output_dir}")
