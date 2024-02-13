from pathlib import Path
import sys

root_directory = Path(__file__).resolve().parents[1]
sys.path.append(str(root_directory))

from ultralytics import YOLO
from typing import List, Tuple, Dict, Union
import cv2
import logging
import platform
import numpy as np
import supervision as sv
import torch
import os
import urllib.request

logger = logging.getLogger(__name__)

class YoloV8ImageObjectDetection:
    MODEL_PATHS = {
        "detection": {
            "yolov8n": "./model/yolov8n/yolov8n.pt",
            "yolov8s": "./model/yolov8s/yolov8s.pt",
            "yolov8m": "./model/yolov8m/yolov8m.pt",
            "yolov8l": "./model/yolov8l/yolov8l.pt",
            "yolov8x": "./model/yolov8x/yolov8x.pt"
        },
        "segmentation": {
            "yolov8n": "./model/yolov8n/yolov8n-seg.pt",
            "yolov8s": "./model/yolov8s/yolov8s-seg.pt",
            "yolov8m": "./model/yolov8m/yolov8m-seg.pt",
            "yolov8l": "./model/yolov8l/yolov8l-seg.pt",
            "yolov8x": "./model/yolov8x/yolov8x-seg.pt"
        }
    }

    def __init__(self, chunked: bytes = None, task_type: str = "detection", model_type: str = "yolov8s"):
        self._bytes = chunked
        self._task_type = task_type
        self._model_type = model_type
        self.PATH = self._get_model_path()
        self.model = self._load_model()
        self.device = self._get_device()
        self.classes = self.model.names

    def _get_model_path(self) -> str:
        if self._task_type in YoloV8ImageObjectDetection.MODEL_PATHS and \
                self._model_type in YoloV8ImageObjectDetection.MODEL_PATHS[self._task_type]:
            return YoloV8ImageObjectDetection.MODEL_PATHS[self._task_type][self._model_type]
        else:
            raise ValueError(f"Unsupported task_type: {self._task_type} or model_type: {self._model_type}")

    def _get_device(self) -> str:
        if platform.system().lower() == "darwin":
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_model(self) -> YOLO:
        try:
            logger.info(f"Loading {self._model_type} model with task type: {self._task_type}")
            if not os.path.exists(self.PATH):
                # Download the model if it doesn't exist
                self._download_model()
            model = YOLO(self.PATH)
            return model
        except Exception as e:
            logger.exception(f"Error loading YOLO model: {e}")
            raise e
    
    def _download_model(self) -> None:
        # Define download URLs for different model types and tasks
        download_urls = {
            "detection": {
                "yolov8m": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt",
                "yolov8l": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt",
                "yolov8x": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt"
            },
            "segmentation": {
                "yolov8m": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-seg.pt",
                "yolov8l": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-seg.pt",
                "yolov8x": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-seg.pt"
            }
        }

        if self._model_type in download_urls.get(self._task_type, {}):
            download_url = download_urls[self._task_type][self._model_type]
            # Extract filename from download URL
            filename = os.path.basename(download_url)
            logger.info(f"Downloading {self._model_type} model from: {download_url}")
            # Download the model file
            urllib.request.urlretrieve(download_url, self.PATH)
            logger.info(f"Model downloaded to: {self.PATH}")
        else:
            logger.warning(f"Model for task_type: {self._task_type} and model_type: {self._model_type} is not available for download.")
    
    async def __call__(self, conf_threshold: int = 25) -> Tuple[np.ndarray, Dict[str, int]]:
        try:
            logger.info("Performing object detection")
            input_image = self._get_image_from_chunked()
            results = self.model.predict(input_image, agnostic_nms=True, verbose=False)[0]
            detections, labels, object_counts = self._get_results_from_detections(results, conf_threshold / 100.0)
            annotated_image = self._annotate_object_detections(input_image, labels, detections)
            logger.info("Object detection successfully completed")
            return annotated_image, object_counts
        except Exception as e:
            logger.exception(f"Error during object detection: {e}")
            raise e
    
    def _get_image_from_chunked(self) -> np.ndarray:
        arr = np.asarray(bytearray(self._bytes), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
        return img
    
    def _get_results_from_detections(self, detections: object, conf_threshold: float) -> Tuple[object, List[str], Dict[str, int]]:
        detections_sv = sv.Detections.from_ultralytics(detections)
        filtered_detections = detections_sv[detections_sv.confidence >= conf_threshold]
        label_detections = self._get_labels_objects(filtered_detections)
        detected_classes = [self.classes[class_id] for class_id in filtered_detections.class_id]
        object_counts = self._count_objects(detected_classes)

        return filtered_detections, label_detections, object_counts
    
    def _get_labels_objects(self, detections: object) -> List[str]:
        labels = [
            f"{self.classes[class_id]} ({confidence*100:0.2f}%)"
            for confidence, class_id
            in zip(detections.confidence, detections.class_id)
        ]
        
        return labels

    def _count_objects(self, object_list: List[str]) -> Dict[str, int]:
        class_counts = {}
        for obj in object_list:
            class_counts[obj] = class_counts.get(obj, 0) + 1
        return class_counts
    
    def _annotate_object_detections(self, image, labels: List[str], detections: object) -> np.ndarray:
        try:
            logger.info("Annotating object detections.")
            if self._task_type == "detection":
                bounding_box_annotator = sv.BoundingBoxAnnotator()
                annotated_image = bounding_box_annotator.annotate(
                    scene=image, 
                    detections=detections
                )
            elif self._task_type == "segmentation":
                mask_annotator = sv.MaskAnnotator()
                annotated_image = mask_annotator.annotate(
                    scene=image, 
                    detections=detections
                )
            else:
                raise ValueError(f"Unsupported task_type: {self._task_type}")
            
            # for large image >> text_scale=5, text_padding=60, text_thickness=5
            label_annotator = sv.LabelAnnotator()

            annotated_image = label_annotator.annotate(
                scene=annotated_image, detections=detections, labels=labels
            )

            return annotated_image

        except Exception as e:
            logger.exception(f"Error during annotation: {e}")
