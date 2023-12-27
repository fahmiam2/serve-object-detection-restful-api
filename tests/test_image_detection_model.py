from pathlib import Path
import sys

root_directory = Path(__file__).resolve().parents[1]
sys.path.append(str(root_directory))

from model.yolov8s.image_detection_model import YoloV8ImageObjectDetection
import asyncio
import cv2
import numpy as np
import pytest
import supervision as sv

@pytest.fixture
def yolo_detector():
    image_path = './static/images/photo1699780157.jpeg' 
    with open(image_path, 'rb') as image_file:
        sample_image_bytes = image_file.read()
    return YoloV8ImageObjectDetection(chunked=sample_image_bytes, task_type="detection")

def test_device_detection(yolo_detector):
    device = yolo_detector._get_device()
    assert device in ["mps", "cuda", "cpu"]

def test_image_loading(yolo_detector):
    image = yolo_detector._get_image_from_chunked()
    assert isinstance(image, np.ndarray)

async def run_detection_async(yolo_detector, conf_threshold):
    return await yolo_detector(conf_threshold)

def test_detection(yolo_detector):
    loop = asyncio.get_event_loop()
    frame, object_counts = loop.run_until_complete(run_detection_async(yolo_detector, conf_threshold=25))

    assert isinstance(frame, np.ndarray)
    assert isinstance(object_counts, dict)

    # print(object_counts)

    # cv2.imshow("Annotated Frame", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()