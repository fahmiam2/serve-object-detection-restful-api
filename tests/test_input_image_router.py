from pathlib import Path
import sys

root_directory = Path(__file__).resolve().parents[1]
sys.path.append(str(root_directory))

from fastapi.testclient import TestClient
from main import app
from PIL import Image
import io
import os

client = TestClient(app)

base_route = "/detect/image"

def _get_image_path(file_name):
    return os.path.join(str(root_directory), "static/images", file_name)

def _get_video_path(file_name):
    return os.path.join(str(root_directory), "static/video", file_name)

# Function to create a fake JPEG image
def _create_fake_image(size=(100, 100), color=(255, 0, 0)):
    image = Image.new("RGB", size, color)
    fake_image_io = io.BytesIO()
    image.save(fake_image_io, format="JPEG")
    fake_image_io.seek(0)
    return fake_image_io

def test_successful_object_detection():
    
    image_path = _get_image_path("horse.jpeg")
    
    # Open the image file
    with open(image_path, "rb") as image_file:
        # Create a dictionary containing the image file
        files = {"image": (image_path, image_file, "image/jpeg")}
        
        # Make a request to the endpoint
        response = client.post(
            url=f"{base_route}?task_type=detection&confidence_threshold=25",
            files=files,
        )

        # Check that the response status code is 200 OK
        assert response.status_code == 200

        # Check the response content
        response_data = response.json()
        assert "frame" in response_data
        assert "object_counts" in response_data

def test_invalid_image_format():
    # Create a fake image with an invalid format for testing
    fake_image = io.BytesIO(b"fake_image_content")
    files = {"image": ("fake_image.txt", fake_image, "text/plain")}
    
    # Make a request to the endpoint
    response = client.post(
        url=f"{base_route}?task_type=detection&confidence_threshold=25",
        files=files,
    )

    # Check that the response status code is 422 Unprocessable Entity
    assert response.status_code == 415

    # Check the response content
    response_data = response.json()
    assert "detail" in response_data

def test_missing_image():
    # Make a request to the endpoint without providing an image
    response = client.post(
        url=f"{base_route}?task_type=detection&confidence_threshold=25",
    )

    # Check that the response status code is 422 Unprocessable Entity
    assert response.status_code == 422

    # Check the response content
    response_data = response.json()
    assert response_data["detail"][0]["input"] == None

def test_invalid_task_type():
    # Create a fake image for testing
    fake_image = _create_fake_image(size=(1024, 1024))  # Adjust the size as needed
    files = {"image": ("fake_image.jpg", fake_image, "image/jpeg")}
    
    # Make a request to the endpoint with an invalid task_type
    response = client.post(
        url=f"{base_route}?task_type=invalid&confidence_threshold=25",
        files=files,
    )

    # Check that the response status code is 422 Unprocessable Entity
    assert response.status_code == 422

    # Check the response content
    response_data = response.json()
    assert response_data["error"] == "Validation error"
    assert response_data["detail"] == "Invalid input. Task type must be detection or segmentation"

def test_invalid_confidence_threshold():
    image_path = _get_image_path("horse.jpeg")
    
    # Open the image file
    with open(image_path, "rb") as image_file:
        # Create a dictionary containing the image file
        files = {"image": (image_path, image_file, "image/jpeg")}
        
        # Make a request to the endpoint
        response = client.post(
            url=f"{base_route}?task_type=detection&confidence_threshold=10",
            files=files,
        )

        # Check that the response status code is 422 Unprocessable Entity
        assert response.status_code == 422

        # Check the response content
        response_data = response.json()
        assert response_data["error"] == "Validation error"
    assert response_data["detail"] == "Invalid input. Confidence must be between 25 and 100."

def test_internal_server_error():
    # Create a fake image for testing
    fake_image = io.BytesIO(b"fake_image_content")
    files = {"image": ("fake_image.jpg", fake_image, "image/jpeg")}
    
    # Simulate an internal server error
    app.state.testing = True
    
    # Make a request to the endpoint
    response = client.post(
        url=f"{base_route}?task_type=detection&confidence_threshold=25",
        files=files,
    )

    # Check that the response status code is 500 Internal Server Error
    assert response.status_code == 500

    # Check the response content
    response_data = response.json()
    assert "detail" in response_data

    # Reset testing mode
    app.state.testing = False

