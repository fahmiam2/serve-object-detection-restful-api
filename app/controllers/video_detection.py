from pathlib import Path
import sys

root_directory = Path(__file__).resolve().parents[3]
sys.path.append(str(root_directory))

from app.services.video_detection import perform_video_detection
from app.schemas.detection_schema import VideoDetectionRequest, VideoDetectionResponse
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
import logging
import os
import uuid

logger = logging.getLogger(__name__)

limiter = Limiter(get_remote_address)

BASE_INPUT_FOLDER_PATH = "./temp/input_videos/"
BASE_OUTPUT_FOLDER_PATH = "./temp/output_videos/"
Path(BASE_INPUT_FOLDER_PATH).mkdir(parents=True, exist_ok=True)
Path(BASE_OUTPUT_FOLDER_PATH).mkdir(parents=True, exist_ok=True)

router = APIRouter()

def stream_file(video_path):
    with open(video_path, mode="rb") as file_bytes:
        for stream_chunk in file_bytes:
            yield stream_chunk

async def save_video_locally(video: UploadFile) -> tuple[str, str]:
    try:
        logger.info("Save video in local server")

        allowed_content_types = ["video/mp4", "video/mpeg", "video/quicktime"]
        if video.content_type not in allowed_content_types:
            raise HTTPException(status_code=422, detail="Invalid video file type. Supported types: " + ", ".join(allowed_content_types))

        MAX_VIDEO_SIZE_MB = 100
        size = await video.read()
        if len(size) > MAX_VIDEO_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"Video file size exceeds the maximum allowed ({MAX_VIDEO_SIZE_MB} MB).")
        
        await video.seek(0)

        file_extension = video.filename.split(".")[-1]
        file_name = str(uuid.uuid4()) + "." + file_extension
        file_input_path = BASE_INPUT_FOLDER_PATH + file_name

        with open(file_input_path, "wb") as video_file:
            video_file.write(video.file.read())

        logger.info("Successfully save video in local server.")

        return file_name, file_input_path
    except HTTPException as e:
        raise e
    
def delete_local_files(input_path: str, output_path: str) -> None:
    try:
        logger.info(f"Deleting local files: {input_path} and {output_path}")
        os.remove(input_path)
        os.remove(output_path)
        logger.info("Local files deleted successfully")
    except Exception as e:
        logger.warning(f"Error deleting local files: {e}")

@router.post("/detect/video", 
             response_model=VideoDetectionResponse,
             summary="Serves Object Detection in Video", 
             tags=["Video serve"],
             description="Performs object detection on an uploaded video and returns url of annotated video ")
@limiter.limit("5/second")
async def detect_objects_in_video(
    request: Request,
    video: UploadFile = File(...),
    task_type: str = "detection",
    confidence_threshold: int = 25,
    annotator: str = "bounding_box",
    use_tracer: bool = False,
    tracer: str = "tracer"
):
    try:
        logger.info("Received request to detect objects in an video.")
        filename, local_input_video_path = await save_video_locally(video)

        local_annotated_video_path = BASE_OUTPUT_FOLDER_PATH + filename

        request_data = VideoDetectionRequest(
            task_type=task_type,
            confidence_threshold=confidence_threshold,
            annotator=annotator,
            use_tracer=use_tracer,
            tracer=tracer
        )
        result = await perform_video_detection(request_data, filename, source_path=local_input_video_path, target_path=local_annotated_video_path)
        return result
    
    except FileNotFoundError as file_not_found_error:
        logger.error(f"File not found error: {file_not_found_error}")
        raise HTTPException(status_code=400, detail=f"File not found: {file_not_found_error}")
    except HTTPException as e:
        if e.status_code == 422:
            logger.warning(f"Validation error: {e.detail}")
            return JSONResponse(content={"detail": "Validation error", "errors": e.detail}, status_code=422)
        if e.status_code == 413:
            logger.warning(f"Validation error: {e.detail}")
            return JSONResponse(content={"detail": "Validation error", "errors": e.detail}, status_code=413)
    except Exception as e:
        logger.exception(f"Internal server error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    finally:
        if local_input_video_path and local_annotated_video_path:
            delete_local_files(input_path=local_input_video_path, output_path=local_annotated_video_path)