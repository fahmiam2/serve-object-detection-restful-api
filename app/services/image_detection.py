from pathlib import Path
import sys

root_directory = Path(__file__).resolve().parents[3]
sys.path.append(str(root_directory))

from app.schemas.detection_schema import ImageDetectionRequest, ImageDetectionResponse
from app.validation.image_detection import validate_object_detection_request
from model.yolov8s.image_detection_model import YoloV8ImageObjectDetection
import cv2
import numpy as np
from fastapi.responses import JSONResponse
import base64

async def perform_object_detection(request_data: ImageDetectionRequest) -> ImageDetectionResponse:
    validate_object_detection_request(request_data)
    
    yolo_model = YoloV8ImageObjectDetection(chunked=request_data.image, task_type=request_data.task_type)
    annotated_image, object_counts = await yolo_model(conf_threshold=request_data.confidence_threshold)

    # Convert annotated image to base64
    _, img_encoded = cv2.imencode('.jpg', annotated_image)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    # Prepare response data
    response_data = {
        "frame": img_base64,
        "object_counts": object_counts
    }

    return JSONResponse(content=response_data)
