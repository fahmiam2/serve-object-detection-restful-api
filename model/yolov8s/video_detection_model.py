from pathlib import Path
import sys

root_directory = Path(__file__).resolve().parents[2]
sys.path.append(str(root_directory))

from typing import Any
from ultralytics import YOLO
import cv2
import platform
import numpy as np
import supervision as sv
import torch

tracker = sv.ByteTrack()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

class YoloV8VideoObjectDetection:
    CONF_THRESHOLD_DEFAULT = 25
    DETECTION_PATH = "./model/yolov8s/yolov8s.pt"
    SEGMENTATION_PATH = "./model/yolov8s/yolov8s-seg.pt"

    ANNOTATOR_TYPE_DETECTION = {
        "bounding_box": sv.BoundingBoxAnnotator(),
        "box_corner": sv.BoxCornerAnnotator(),
        "color": sv.ColorAnnotator(),
        "circle": sv.CircleAnnotator(),  # Fixed spelling
        "dot": sv.DotAnnotator(),
        "triangle": sv.TriangleAnnotator(),
        "ellipse": sv.EllipseAnnotator(),
    }

    ANNOTATOR_TYPE_SEGMENTATION = {
        "halo": sv.HaloAnnotator(),
        "mask": sv.MaskAnnotator(),
        "polygon": sv.PolygonAnnotator(),
    }

    TRACER_TYPE = {
        "tracer": sv.TraceAnnotator(),
        "heatmap": sv.HeatMapAnnotator()
    }

    def __init__(self, task_type: str = "detection") -> None:
        self._validate_task_type(task_type)
        self._task_type = task_type
        self.PATH = self._get_model_path()
        self.model = self._load_model()
        self.device = self._get_device()
        self.classes = self.model.names

    def _validate_task_type(self, task_type: str) -> None:
        """Validate the provided task type."""
        if task_type not in ["detection", "segmentation"]:
            raise ValueError(f"Invalid task_type: {task_type}")

    def _get_model_path(self):
        """Get the appropriate model path based on task_type."""
        if self._task_type == "detection":
            return YoloV8VideoObjectDetection.DETECTION_PATH
        elif self._task_type == "segmentation":
            return YoloV8VideoObjectDetection.SEGMENTATION_PATH
        else:
            raise ValueError(f"Unsupported task_type: {self._task_type}")
    
    def _get_device(self):
        """Gets best device for your system

        Returns:
            device (str): The device to use for YOLO for your system
        """
        if platform.system().lower() == "darwin":
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def _load_model(self):
        """Loads Yolo8 model from pytorch hub or a path on disk

        Returns:
            model (Model) - Trained Pytorch model
        """
        model = YOLO(self.PATH)
        return model
    
    def callback(self, frame: np.ndarray, _: int, conf_threshold: int = 25, annotator: str = "bounding_box", use_tracer: bool = False, tracer: str = None) -> Any:
        if annotator not in ["bounding_box", "box_corner", "color", "circel", "dot", "triangle", "ellipse", "halo", "mask", "polygon"]:
            raise ValueError(f"Invalid annotator type: {annotator}")

        results = self.model(frame)[0]
        detections_sv = sv.Detections.from_ultralytics(results)
        detections = detections_sv[detections_sv.confidence >= conf_threshold / 100]

        # annotating
        annotated_frame = self._get_annotator_instance(frame, detections, annotator)

        # labelling
        detections = tracker.update_with_detections(detections)

        labels = [
            f"#{tracker_id} {results.names[class_id]} ({confidence*100:0.2f}%)"
            for class_id, tracker_id, confidence
            in zip(detections.class_id, detections.tracker_id, detections.confidence)
        ]
        
        annotated_frame = self._annotate_with_label(annotated_frame, detections, labels)
        
        # tracing
        if use_tracer and tracer is not None:
            if tracer not in ["tracer", "heatmap"]:
                raise ValueError(f"Invalid tracer type: {tracer}")
            annotated_frame = self._annotate_with_tracer(annotated_frame, tracer, detections)
            
        return annotated_frame
    
    def _get_annotator_instance(self, frame, detections, annotator):
        if isinstance(annotator, str):
            if annotator in self.ANNOTATOR_TYPE_DETECTION and self._task_type == "detection":
                annotator_instance =  self.ANNOTATOR_TYPE_DETECTION[annotator]
            elif annotator in self.ANNOTATOR_TYPE_SEGMENTATION and self._task_type == "segmentation":
                annotator_instance = self.ANNOTATOR_TYPE_SEGMENTATION[annotator]
            else:
                raise ValueError(f"Unsupported annotator for task_type: {annotator} for {self._task_type}")
            
            annotate_frame = annotator_instance.annotate(
                    scene=frame.copy(),
                    detections=detections
                )
            return annotate_frame
        else:
            raise ValueError("Invalid annotator type")
        
    def _annotate_with_label(self, annotated_frame, detections, labels):
        label_annotator = sv.LabelAnnotator()
        return label_annotator.annotate(
            annotated_frame, detections=detections, labels=labels
        )
        
    def _annotate_with_tracer(self, annotated_frame, tracer, detections):
        if tracer is not None and isinstance(tracer, str) and tracer in self.TRACER_TYPE:
            tracer_instance = self.TRACER_TYPE[tracer]
            return tracer_instance.annotate(
                annotated_frame, detections=detections
            )
        return annotated_frame