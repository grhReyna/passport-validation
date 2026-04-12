"""
API REST - Sistema de Detección de Pasaportes Falsos
Framework: FastAPI
Versión: 0.1.0

Endpoints:
  - POST /verify-passport: Verificar autenticidad de pasaporte
  - GET /health: Estado de la API
  - GET /docs: Documentación Swagger
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import logging
from datetime import datetime
from pathlib import Path
import uuid
import config

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Crear aplicación FastAPI
app = FastAPI(
    title="Sistema de Detección de Pasaportes Falsos",
    description="Verifica autenticidad de pasaportes mexicanos usando IA (OCR + MRZ)",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration (permitir requests desde localhost en desarrollo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5000",
        "http://localhost:5000",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Logger
logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# ============================================================================
# IMPORTAR PIPELINE REAL
# ============================================================================

from src.pipeline import verify_passport

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["UI"])
async def root():
    """
    Servir interfaz web principal
    """
    static_dir = Path(__file__).parent / "static"
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Sistema de Detección de Pasaportes - API REST"}

@app.get("/favicon.ico", tags=["UI"])
async def favicon():
    """Servir favicon (prevenir error 404)"""
    return {"status": "ok"}

@app.get("/health", tags=["Sistema"])
async def health_check():
    """
    Verificar estado de la API
    
    Returns:
        dict: Estado de salud de la API
    """
    return {
        "status": "OK",
        "service": "Sistema de Detección de Pasaportes",
        "version": "0.1.0",
        "timestamp": datetime.utcnow().isoformat()
    }

# Estado global del modelo
_model_status = {"status": "not_loaded", "step": "", "progress": 0, "total": 0}

@app.get("/model-status", tags=["Sistema"])
async def model_status():
    """Estado de carga/descarga del modelo"""
    return _model_status

@app.post("/verify-passport", tags=["Verificación"])
async def verify_passport_endpoint(file: UploadFile = File(...)):
    """
    Verificar autenticidad de un pasaporte
    
    Args:
        file (UploadFile): Archivo de imagen del pasaporte (PNG/JPG)
        
    Returns:
        dict: Resultado de verificación con score de confianza
        
    Raises:
        HTTPException: Si el archivo no es válido o supera tamaño máximo
        
    Example response:
        {
            "id": "uuid-123",
            "timestamp": "2026-02-24T10:30:00Z",
            "autenticidad_score": 87.5,
            "estado": "REVIEW",
            "confianza": {
                "ocr": 0.92,
                "mrz": 0.95,
                "coherencia": 0.80
            },
            "anomalias": [...],
            "detalles": {...},
            "recomendacion": "REVIEW: Verificación manual requerida"
        }
    """
    try:
        # Validar tipo de archivo
        allowed_types = ["image/jpeg", "image/png", "image/jpg"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Tipo de archivo no permitido. Usa PNG o JPG. Recibido: {file.content_type}"
            )
        
        # Leer contenido del archivo
        contents = await file.read()
        
        # Validar tamaño
        if len(contents) > config.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Archivo demasiado grande. Máximo: {config.MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
            )
        
        # Procesar imagen con pipeline real
        logger.info(f"Procesando imagen: {file.filename} ({len(contents) / 1024:.1f}KB)")
        result = verify_passport(contents, verbose=False)
        
        # Validar que hubo resultado
        if result.get("error"):
            logger.error(f"Error procesando {file.filename}: {result['error']}")
            raise HTTPException(status_code=500, detail=f"Error procesando imagen: {result['error']}")
        
        if result.get("final_result"):
            final = result["final_result"]
            logger.info(f"Resultado para {file.filename}: {final.get('estado')} ({final.get('autenticidad_score'):.1f}%)")
        
        return JSONResponse(content=result)
    
    except HTTPException as e:
        logger.exception(f"HTTPException al procesar {file.filename}")
        raise e
    
    except Exception as e:
        logger.exception(f"Error procesando {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando imagen: {str(e)}"
        )

@app.post("/verify-passport-debug", tags=["Debug"])
async def verify_passport_debug(file: UploadFile = File(...)):
    """
    Endpoint de DEBUG - Procesa imagen y retorna info detallada del preprocesamiento
    """
    try:
        import cv2
        import numpy as np
        from src import mrz_roi_detector, preprocessing

        def _json_safe(value):
            if isinstance(value, dict):
                return {k: _json_safe(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_json_safe(v) for v in value]
            if isinstance(value, tuple):
                return [_json_safe(v) for v in value]
            if isinstance(value, np.integer):
                return int(value)
            if isinstance(value, np.floating):
                return float(value)
            if isinstance(value, np.bool_):
                return bool(value)
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            return value
        
        contents = await file.read()
        
        # Cargar imagen ORIGINAL
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"error": "No se pudo decodificar imagen"}
        
        h_original, w_original = image.shape[:2]
        aspect_ratio = h_original / w_original if w_original > 0 else 1
        
        # Análisis de imagen ORIGINAL
        debug_info = {
            "imagen_original": {
                "ancho": w_original,
                "alto": h_original,
                "aspecto": round(aspect_ratio, 3),
                "tipo_detectado": "2 PÁGINAS LADO A LADO" if aspect_ratio < 0.5 else ("2 PÁGINAS VERTICALES" if aspect_ratio > 1.2 else "NORMAL")
            }
        }
        
        # Detectar ROI con función preprocesamiento
        roi = preprocessing.detect_passport_roi(image)
        x, y, w_roi, h_roi = roi
        debug_info["roi_detectado"] = {
            "x": x,
            "y": y,
            "ancho": w_roi,
            "alto": h_roi,
            "descripcion": "Región que será recortada"
        }
        
        # Recortar
        cropped = image[y:y+h_roi, x:x+w_roi]
        h_crop, w_crop = cropped.shape[:2]
        
        # Redimensionar
        resized = preprocessing.resize_image(cropped)
        h_resize, w_resize = resized.shape[:2]
        
        debug_info["despues_preprocesamiento"] = {
            "tras_recorte": {"ancho": w_crop, "alto": h_crop},
            "tras_resize": {"ancho": w_resize, "alto": h_resize}
        }
        
        # Test MRZ en imagen preprocesada
        mrz_roi = mrz_roi_detector.find_mrz_region(resized)
        debug_info["mrz"] = {
            "detectado": mrz_roi is not None,
            "roi": {"x": mrz_roi[0], "y": mrz_roi[1], "w": mrz_roi[2], "h": mrz_roi[3]} if mrz_roi else None
        }
        
        # Procesar con pipeline completo
        result = verify_passport(contents, verbose=False)
        debug_info["resultado_final"] = {
            "score": result.get("final_result", {}).get("autenticidad_score"),
            "estado": result.get("final_result", {}).get("estado"),
            "confianzas": result.get("final_result", {}).get("confianza", {})
        }
        
        return JSONResponse(content=_json_safe(debug_info))
        
    except Exception as e:
        logger.exception(f"Error en debug: {str(e)}")
        return {"error": str(e)}

@app.get("/stats", tags=["Sistema"])
async def get_stats():
    """
    Obtener estadísticas del sistema
    
    Returns:
        dict: Estadísticas de uso y rendimiento
    """
    return {
        "service": "Sistema de Detección de Pasaportes",
        "version": "0.1.0",
        "status": "desarrollo",
        "componentes": {
            "preprocessing": "Pendiente",
            "ocr": "Pendiente",
            "mrz_validator": "Pendiente",
            "confidence_scorer": "Pendiente",
            "pipeline": "Pendiente"
        },
        "dataset": {
            "nombre": "Synthetic Printed Mexican Passports",
            "fuente": "Kaggle (unidpro)",
            "estado": "Pendiente descarga"
        }
    }

# ============================================================================
# MANEJO DE ERRORES
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Manejador para rutas no encontradas"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Ruta no encontrada",
            "path": request.url.path,
            "message": "Usa /docs para ver documentación de API"
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Manejador general de excepciones"""
    logger.exception("Error no manejado")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Error interno del servidor",
            "message": "Contactar administrador"
        }
    )

# ============================================================================
# EVENTOS DEL CICLO DE VIDA
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Ejecutar al iniciar la aplicación"""
    logger.info("═" * 60)
    logger.info("Sistema de Detección de Pasaportes - INICIANDO")
    logger.info(f"Versión: 0.1.0")
    logger.info(f"Host: {config.API_HOST}:{config.API_PORT}")
    logger.info(f"Documentación: http://{config.API_HOST}:{config.API_PORT}/docs")
    logger.info("═" * 60)
    
    # TODO: Cargar modelos (TrOCR, etc.)
    # TODO: Validar que dataset está descargado
    # TODO: Compilar/cachear componentes

@app.on_event("shutdown")
async def shutdown_event():
    """Ejecutar al apagar la aplicación"""
    logger.info("═" * 60)
    logger.info("Sistema de Detección de Pasaportes - APAGANDO")
    logger.info("═" * 60)
    
    # TODO: Limpiar recursos
    # TODO: Guardar estadísticas

# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.API_RELOAD,
        log_level=config.LOG_LEVEL.lower()
    )
