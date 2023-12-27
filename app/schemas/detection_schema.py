from typing import List, Union
from pydantic import BaseModel
from io import BytesIO

class ImageDetectionRequest(BaseModel):
    image: bytes
    task_type: str
    confidence_threshold: int

class ImageDetectionResponse(BaseModel):
    frame: str
    object_counts: dict

class VideoDetectionRequest(BaseModel):
    task_type: str
    confidence_threshold: int
    annotator: str 
    use_tracer: bool
    tracer: str 

class VideoDetectionResponse(BaseModel):
    url_video: str

class WebcamDetectionRequest(BaseModel):
    model: str
    confidence: int

class DetectionResponse(BaseModel):
    labels: List[str]

class WebSocketConnectionResponse(BaseModel):
    detail: str

class WebSocketDetectionResponse(BaseModel):
    detail: str 
