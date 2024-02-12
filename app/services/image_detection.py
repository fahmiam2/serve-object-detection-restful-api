from pathlib import Path
import sys

root_directory = Path(__file__).resolve().parents[3]
sys.path.append(str(root_directory))

from app.schemas.detection_schema import ImageDetectionRequest, ImageDetectionResponse
from app.validation.image_detection import validate_object_detection_request
from model.image_detection_model import YoloV8ImageObjectDetection
import cv2
from fastapi.responses import JSONResponse
import base64

async def perform_object_detection(request_data: ImageDetectionRequest, content_type: str) -> JSONResponse:
    validate_object_detection_request(request_data)
    
    yolo_model = YoloV8ImageObjectDetection(chunked=request_data.image, model_type=request_data.model_type, task_type=request_data.task_type)
    annotated_image, object_counts = await yolo_model(conf_threshold=request_data.confidence_threshold)

    # Convert annotated image to base64
    if content_type.lower() in ["image/jpg", "image/jpeg"]:
        _, img_encoded = cv2.imencode('.jpg', annotated_image)
    elif content_type.lower() == "image/png":
        _, img_encoded = cv2.imencode('.png', annotated_image)
    else:
        return JSONResponse(content={"error": "Unsupported content type"}, status_code=400)

    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    # Prepare response data
    response_data = {
        "object_counts": object_counts,
        "frame": img_base64
    }

    return JSONResponse(content=response_data, status_code=200)
