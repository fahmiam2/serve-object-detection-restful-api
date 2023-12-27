from pathlib import Path
import sys

root_directory = Path(__file__).resolve().parents[3]
sys.path.append(str(root_directory))

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from app.services.image_detection import perform_object_detection
from app.schemas.detection_schema import ImageDetectionRequest, ImageDetectionResponse

router = APIRouter()

MAX_FILE_SIZE_MB = 10

@router.post("/detect/image", 
             response_model=ImageDetectionResponse, 
             summary="Object Detection in Uploaded Images", 
             tags=["Image Detection"],
             description="Performs object detection on an uploaded image and returns annotated image in base64 & list of detected objects.")
async def detect_objects_in_image(
    image: UploadFile = File(...),
    task_type: str = "detection",
    confidence_threshold: int = 25
):
    if image.content_type not in ["image/jpg", "image/jpeg", "image/png"]:
        raise HTTPException(status_code=422, detail="Invalid image format. Only JPG, JPEG and PNG are supported.")

    if image.file.seek(0, 2) > (MAX_FILE_SIZE_MB * 1024 * 1024):
        raise HTTPException(status_code=413, detail=f"File size exceeds the maximum allowed ({MAX_FILE_SIZE_MB} MB).")
    
    # Reset the file position back to the beginning
    image.file.seek(0)
    
    try:
        contents = await image.read()
        request_data = ImageDetectionRequest(image=contents, task_type=task_type, confidence_threshold=confidence_threshold)
        result = await perform_object_detection(request_data)
        return result
    except FileNotFoundError as file_not_found_error:
        raise HTTPException(status_code=400, detail=f"File not found: {file_not_found_error}")
    except HTTPException as e:
        if e.status_code == 422:
            return JSONResponse(content={"detail": "Validation error", "errors": e.detail}, status_code=422)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
