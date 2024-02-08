from pathlib import Path
import sys

root_directory = Path(__file__).resolve().parents[3]
sys.path.append(str(root_directory))

from app.services.image_detection import perform_object_detection
from app.schemas.detection_schema import ImageDetectionRequest, ImageDetectionResponse
from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from typing import Optional
import logging

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)

router = APIRouter()

MAX_FILE_SIZE_MB = 15

@router.post("/detect/image", 
             response_model=ImageDetectionResponse, 
             summary="Object Detection in Uploaded Image", 
             tags=["Image Detection"],
             description="Performs object detection on an uploaded image and returns annotated image in base64 & list of detected objects.",
)
@limiter.limit("25/second")
async def detect_objects_in_image(
    request: Request,
    image: UploadFile = File(...),
    task_type: str = "detection",
    confidence_threshold: int = 25,
) -> Optional[JSONResponse]:
    
    try:
        logger.info("Received request to detect objects in an image.")

        content_type = image.content_type
        if content_type not in ["image/jpg", "image/jpeg", "image/png"]:
            raise HTTPException(status_code=415, detail="Invalid image format. Only JPG, JPEG and PNG are supported.")

        if image.file.seek(0, 2) > (MAX_FILE_SIZE_MB * 1024 * 1024):
            raise HTTPException(status_code=413, detail=f"File size exceeds the maximum allowed ({MAX_FILE_SIZE_MB} MB).")
        
        # Reset the file position back to the beginning
        image.file.seek(0)

        contents = await image.read()
        request_data = ImageDetectionRequest(image=contents, task_type=task_type, confidence_threshold=confidence_threshold)
        result = await perform_object_detection(request_data, content_type=content_type)

        return result

    except HTTPException as e:
        if e.status_code == 422:
            logger.error(f"Validation error: {e.detail}")
            return JSONResponse(content={"error": "Validation error", "detail": e.detail}, status_code=422)
        if e.status_code == 413:
            logger.warning(f"Validation error: {e.detail}")
            return JSONResponse(content={"error": "Validation error", "detail": e.detail}, status_code=413)
        if e.status_code == 415:
            logger.warning(f"Validation error: {e.detail}")
            return JSONResponse(content={"error": "Validation error", "detail": e.detail}, status_code=415)
    except Exception as e:
        logger.exception(f"Internal server error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")