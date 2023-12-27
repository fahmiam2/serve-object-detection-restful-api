from pathlib import Path
import sys

root_directory = Path(__file__).resolve().parents[3]
sys.path.append(str(root_directory))

from fastapi import HTTPException
from app.schemas.detection_schema import VideoDetectionRequest

def validate_video_detection_request(request: VideoDetectionRequest):
    
    task_type = request.task_type
    if task_type.lower() not in ["detection", "segmentation"]:
        raise HTTPException(status_code=422, detail="Invalid input. Task type must be detection or segmentation")
    
    confidence_threshold = request.confidence_threshold
    if not (25 <= confidence_threshold <= 100):
        raise HTTPException(status_code=422, detail="Invalid input. Confidence must be between 25 and 100.")

    annotator = request.annotator
    if annotator not in ["bounding_box", "box_corner", "color", "circle", "dot", "triangle", "ellipse", "halo", "mask", "polygon"]:
        raise HTTPException(status_code=422, detail="Invalid input. Annotator type is not supported.")
    
    use_tracer = request.use_tracer
    tracer = request.tracer
    if use_tracer and tracer not in ["tracer", "heatmap"]:
        raise HTTPException(status_code=422, detail="Invalid input. Tracer type is not supported.")
