from pathlib import Path
import sys

root_directory = Path(__file__).resolve().parents[3]
sys.path.append(str(root_directory))

from typing import Any
from app.schemas.detection_schema import VideoDetectionRequest
from app.validation.video_detection import validate_video_detection_request
from model.yolov8s.video_detection_model import YoloV8VideoObjectDetection
from fastapi.responses import JSONResponse
from google.cloud import storage
from config.settings import GCS_KEY_FILE, GCS_BUCKET_NAME
import logging
import supervision as sv

logger = logging.getLogger(__name__)

def upload_to_gcs(bucket_name, source_file_path, object_name, credential_path) -> None:
    client = storage.Client.from_service_account_json(credential_path)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_filename(source_file_path)

def generate_signed_url(bucket_name, object_name, credential_path, expiration=3600) -> str:
    client = storage.Client.from_service_account_json(credential_path)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    url = blob.generate_signed_url(
        version="v4",
        expiration=expiration,
        method="GET",
    )
    return url

async def perform_video_detection(request_data: VideoDetectionRequest, filename, source_path, target_path) -> JSONResponse:
    # Validate the request
    validate_video_detection_request(request_data)
    
    # Create YoloV8VideoObjectDetection instance
    yolo_model = YoloV8VideoObjectDetection(task_type=request_data.task_type)
    
    logger.info("Performing object detection")
    source_video_info = sv.VideoInfo.from_video_path(video_path=source_path) 
    with sv.VideoSink(target_path=target_path, video_info=source_video_info, codec="h264") as sink:
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

    logger.info("Object detection successfully completed")

    logger.info(f"Uploading {filename} to GCS")
    # Additional logic to upload annotated frames to GCS
    gcs_object_name = f"{filename}"  # Adjust the naming as needed
    upload_to_gcs(bucket_name=GCS_BUCKET_NAME, source_file_path=target_path, object_name=gcs_object_name, credential_path=GCS_KEY_FILE)
    logger.info(f"Successfully uploading {filename} to gcs")

    logger.info(f"Generate signed URL for accessing file: {filename}")
    # Additional logic to generate signed URL for the video
    signed_url = generate_signed_url(bucket_name=GCS_BUCKET_NAME, object_name=gcs_object_name, credential_path=GCS_KEY_FILE)
    logger.info("Signed url has successfully created")
    
    return JSONResponse(content={"url_video": signed_url}, status_code=200)
    
