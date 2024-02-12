from pathlib import Path
import sys

root_directory = Path(__file__).resolve().parents[3]
sys.path.append(str(root_directory))

from typing import Any
from app.schemas.detection_schema import VideoDetectionRequest
from app.validation.video_detection import validate_video_detection_request
from app.utils import video as vd
from model.video_detection_model import YoloV8VideoObjectDetection
from fastapi.responses import JSONResponse
from google.auth.transport import requests
from google.cloud import storage
from config.settings import GCS_KEY_FILE, GCS_BUCKET_NAME
import asyncio
import google.auth
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Global variable to store the GCS client
gcs_client = None

def get_gcs_client(credential_path: str = None) -> storage.Client:
    global gcs_client
    if gcs_client is None:
        try:
            if credential_path:
                gcs_client = storage.Client.from_service_account_json(credential_path)
            else:
                gcs_client = storage.Client()
        except Exception as e:
            logger.error(f"Error creating GCS client: {e}")
            raise
    return gcs_client

def upload_to_gcs(bucket_name: str, source_file_path: str, object_name: str, credential_path: str = None) -> None:
    client = get_gcs_client(credential_path)

    if client is None:
        raise ValueError("GCS client must be provided.")

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_filename(source_file_path)

    return client

def generate_signed_url(bucket_name: str, object_name: str, credential_path: str = None, expiration: int = 3600) -> str:
    
    if credential_path is None:
        credentials, _ = google.auth.default()
        r = requests.Request()
        credentials.refresh(r)

        client = storage.Client()
    else:
        client = storage.Client.from_service_account_json(credential_path)

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    service_account_email = credentials.service_account_email

    url = blob.generate_signed_url(
        service_account_email=service_account_email,
        access_token=credentials.token,
        version="v4",
        method="GET",
        expiration=expiration,
    )
    return url

async def process_frame_async(frame: np.ndarray, index: int, 
                              yolo_model: YoloV8VideoObjectDetection, 
                              request_data: VideoDetectionRequest, sink: vd.VideoSink) -> None:
    try:
        logger.info(f"Processing frame {index}")
        result_frame = yolo_model.callback(
            frame,
            index,
            conf_threshold=request_data.confidence_threshold,
            annotator=request_data.annotator,
            use_tracer=request_data.use_tracer,
            tracer=request_data.tracer
        )
        sink.write_frame(frame=result_frame)
        logger.info(f"Frame {index} processed successfully")
    except Exception as e:
        logger.error(f"Error processing frame {index}: {e}")
    finally:
        logger.info(f"Frame {index} processing complete")

async def perform_video_detection(request_data: VideoDetectionRequest, filename: str, 
                                  source_path: str, target_path: str) -> JSONResponse:
    # Validate the request
    validate_video_detection_request(request_data)

    # Create YoloV8VideoObjectDetection instance
    yolo_model = YoloV8VideoObjectDetection(task_type=request_data.task_type)
    
    logger.info("Performing object detection")
    source_video_info = vd.VideoInfo.from_video_path(video_path=source_path) 

    async with vd.VideoSink(target_path=target_path, video_info=source_video_info, codec="vp09") as sink:
        tasks = [process_frame_async(frame, index, yolo_model, request_data, sink)
                 for index, frame in enumerate(vd.get_video_frames_generator(source_path=source_path))]
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error during frame processing: {e}")

    logger.info("Object detection successfully completed")

    logger.info(f"Uploading {filename} to GCS")
    
    # Additional logic to upload annotated frames to GCS
    gcs_object_name = f"{filename}" 
    client = upload_to_gcs(bucket_name=GCS_BUCKET_NAME, source_file_path=target_path, object_name=gcs_object_name, credential_path=GCS_KEY_FILE)
    logger.info(f"Successfully uploading {filename} to gcs")
    logger.info(f"here is the client: {client}")

    logger.info(f"Generate signed URL for accessing file: {filename}")
    # Additional logic to generate signed URL for the video
    signed_url = generate_signed_url(bucket_name=GCS_BUCKET_NAME, object_name=gcs_object_name)
    logger.info("Signed url has successfully created")
    
    return JSONResponse(content={"url_video": signed_url}, status_code=200)
    
