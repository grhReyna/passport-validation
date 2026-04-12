#!/usr/bin/env python3
"""
Detectar GPU disponible e instalar PyTorch con soporte CUDA correspondiente.
"""
import subprocess
import sys
import json

def detect_cuda():
    """Detectar CUDA usando nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=compute_cap,driver_version,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines:
                # Obtener versión de CUDA del driver
                cuda_info = subprocess.run(
                    ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
                    capture_output=True, text=True, timeout=5
                )
                if cuda_info.returncode == 0:
                    return True, cuda_info.stdout.strip().split('\n')[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False, None

def detect_cuda_version():
    """Detectar versión de CUDA installer."""
    try:
        result = subprocess.run(
            ['nvcc', '--version'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and 'release' in result.stdout:
            # Extraer versión como "12.1" o "11.8"
            for word in result.stdout.split():
                if 'release' in result.stdout[result.stdout.index(word):result.stdout.index(word)+20]:
                    parts = word.split('.')
                    if len(parts) >= 2:
                        return f"{parts[0]}.{parts[1]}"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None

def get_pytorch_command(has_gpu, cuda_version):
    """Retornar comando pip para instalar PyTorch correcto."""
    
    if not has_gpu:
        print("GPU no detectada - Usando PyTorch CPU")
        return "torch==2.0.0 torchvision==0.15.0"
    
    # Si hay GPU, usar PyTorch 2.2+ que tiene mejor soporte CUDA
    print(f"GPU NVIDIA detectada - Instalando PyTorch 2.2 con soporte CUDA")
    # PyTorch 2.2 con CUDA 12.1 (compatible con RTX 30-series y superiores)
    return "torch torchvision --index-url https://download.pytorch.org/whl/cu121"

if __name__ == "__main__":
    has_gpu, compute_cap = detect_cuda()
    cuda_ver = detect_cuda_version()
    
    # Crear JSON con resultado
    result = {
        "has_gpu": has_gpu,
        "compute_cap": compute_cap,
        "cuda_version": cuda_ver,
        "pytorch_install_cmd": get_pytorch_command(has_gpu, cuda_ver)
    }
    
    # Imprimir JSON para que PowerShell/Bash lo parsee
    print(json.dumps(result))
