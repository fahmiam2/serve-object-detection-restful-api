from pathlib import Path
import sys

root_directory = Path(__file__).resolve().parents[3]
sys.path.append(str(root_directory))

from fastapi import HTTPException
from app.schemas.detection_schema import ImageDetectionRequest

def validate_object_detection_request(request: ImageDetectionRequest):
    if not request.image:
        raise HTTPException(status_code=422, detail="Invalid input. Must provide an image file.")
    
    task_type = request.task_type
    if task_type.lower() not in ["detection", "segmentation"]:
        raise HTTPException(status_code=422, detail="Invalid input. Task type must be detection or segmentation")
    
    confidence_threshold = request.confidence_threshold
    if not (25 <= confidence_threshold <= 100):
        raise HTTPException(status_code=422, detail="Invalid input. Confidence must be between 25 and 100.")
