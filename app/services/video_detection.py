from pathlib import Path
import sys

root_directory = Path(__file__).resolve().parents[3]
sys.path.append(str(root_directory))

from typing import Any
from app.schemas.detection_schema import VideoDetectionRequest
from app.validation.video_detection import validate_video_detection_request
from model.yolov8s.video_detection_model import YoloV8VideoObjectDetection
from fastapi.responses import StreamingResponse
import supervision as sv
from io import BytesIO

async def perform_video_detection(request_data: VideoDetectionRequest, source_path: str, target_path: str):
    # Validate the request
    validate_video_detection_request(request_data)
    
    # Create YoloV8VideoObjectDetection instance
    yolo_model = YoloV8VideoObjectDetection(task_type=request_data.task_type)
    
    source_video_info = sv.VideoInfo.from_video_path(video_path=source_path) 
    with sv.VideoSink(target_path=target_path, video_info=source_video_info) as sink:
        for index, frame in enumerate(
            sv.get_video_frames_generator(source_path=source_path)
        ):
            result_frame = yolo_model.callback(
                frame, 
                index,
                conf_threshold=request_data.confidence_threshold,
                annotator=request_data.annotator,
                use_tracer=request_data.use_tracer,
                tracer=request_data.tracer
            )
            sink.write_frame(frame=result_frame)
    
