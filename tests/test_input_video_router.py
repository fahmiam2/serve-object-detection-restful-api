from pathlib import Path
import sys

root_directory = Path(__file__).resolve().parents[1]
sys.path.append(str(root_directory))

import io
import os
import pytest
from fastapi.testclient import TestClient
from main import app
from unittest.mock import patch
from app.schemas.detection_schema import VideoDetectionRequest, VideoDetectionResponse

client = TestClient(app)

base_route = "/detect/video"

@pytest.fixture
def _get_video_path():
    def get_video_path(file_name):
        return os.path.join(str(root_directory), "static/video", file_name)
    return get_video_path

def _get_image_path(file_name):
    return os.path.join(str(root_directory), "static/images", file_name)

def test_invalid_video_format(_get_video_path):
    
    video_path = _get_image_path("horse.jpeg")

    with open(video_path, "rb") as video_file:

        files = {"video": (video_path, video_file, "image/jpeg")}

        response = client.post(
            url=f"{base_route}?task_type=detection&confidence_threshold=25&annotator=bounding_box&use_tracer=False",
            files=files
        )

        # Check that the response status code is 422 (Unprocessable Entity)
        assert response.status_code == 415

        # Check the response data
        response_data = response.json()
        assert "Invalid video file type. Supported types" in response_data["detail"]

def test_missing_video():
    # Make a request to the endpoint without providing an image
    response = client.post(
        url=f"{base_route}?task_type=detection&confidence_threshold=25",
    )

    # Check that the response status code is 422 Unprocessable Entity
    assert response.status_code == 422

    # Check the response content
    response_data = response.json()
    assert response_data["detail"][0]["input"] == None

def test_invalid_task_type(_get_video_path):
    video_path = _get_video_path("people-walking.mp4")

    with open(video_path, "rb") as video_file:
        files = {"video": (video_path, video_file, "video/mp4")}

        response = client.post(
            url=f"{base_route}?task_type=kuy&confidence_threshold=25&annotator=bounding_box&use_tracer=False",
            files=files
        )

    # Check that the response status code is 422 (Unprocessable Entity)
    assert response.status_code == 422

    # Check the response data
    response_data = response.json()
    assert response_data["error"] == "Validation error"
    assert response_data["detail"] == "Invalid input. Task type must be detection or segmentation"

def test_invalid_confidence_threshold(_get_video_path):
    video_path = _get_video_path("people-walking.mp4")

    with open(video_path, "rb") as video_file:
        files = {"video": (video_path, video_file, "video/mp4")}

        response = client.post(
            url=f"{base_route}?task_type=detection&confidence_threshold=10&annotator=bounding_box&use_tracer=False",
            files=files
        )

    # Check that the response status code is 422 (Unprocessable Entity)
    assert response.status_code == 422

    # Check the response data
    response_data = response.json()
    assert response_data["error"] == "Validation error"
    assert response_data["detail"] == "Invalid input. Confidence must be between 25 and 100."

def test_invalid_annotator(_get_video_path):
    video_path = _get_video_path("people-walking.mp4")

    with open(video_path, "rb") as video_file:
        files = {"video": (video_path, video_file, "video/mp4")}

        response = client.post(
            url=f"{base_route}?task_type=detection&confidence_threshold=25&annotator=bounding&use_tracer=False",
            files=files
        )

    # Check that the response status code is 422 (Unprocessable Entity)
    assert response.status_code == 422

    # Check the response data
    response_data = response.json()
    assert response_data["error"] == "Validation error"
    assert response_data["detail"] == "Invalid input. Annotator type is not supported."

def test_invalid_tracer(_get_video_path):
    video_path = _get_video_path("people-walking.mp4")

    with open(video_path, "rb") as video_file:
        files = {"video": (video_path, video_file, "video/mp4")}

        response = client.post(
            url=f"{base_route}?task_type=detection&confidence_threshold=25&annotator=bounding_box&use_tracer=True&tracer=kuy",
            files=files
        )

    # Check that the response status code is 422 (Unprocessable Entity)
    assert response.status_code == 422

    # Check the response data
    response_data = response.json()
    assert response_data["error"] == "Validation error"
    assert response_data["detail"] == "Invalid input. Tracer type is not supported."

def test_successful_object_detection(_get_video_path):
    video_path = _get_video_path("people-walking.mp4")

    # Mock the upload_to_gcs and generate_signed_url functions
    with patch("app.services.video_detection.upload_to_gcs") as mock_upload, \
         patch("app.services.video_detection.generate_signed_url") as mock_generate_url:

        # Set the return values for the mock functions
        mock_upload.return_value = None
        mock_generate_url.return_value = "https://mocked-signed-url.com"

        with open(video_path, "rb") as video_file:
            files = {"video": (video_path, video_file, "video/mp4")}

            response = client.post(
                url=f"{base_route}?task_type=detection&confidence_threshold=25&annotator=bounding_box&use_tracer=False",
                files=files
            )

        # Check that the response status code is 200
        assert response.status_code == 200

        # Check the response data
        response_data = response.json()
        assert "url_video" in response_data
        print(response_data)
        assert response_data["url_video"] == "https://mocked-signed-url.com"