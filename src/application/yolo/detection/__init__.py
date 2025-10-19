from ultralytics import YOLO
import numpy as np
from typing import Generator
from entities.yolo.detection import EntityYoloDetection as EntityYoloDetection
from ultralytics.engine.results import Results

class YoloDetection:
    def __init__(
        self,
        confidence_threshold: float = 0.9,
        iou_threshold: float = 0.3,
        device: str = "cuda",
        classes: list[int] = [0],
    ):
        self.model = YOLO("models/yolo11x.pt")
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.classes = classes


    def extract(self, frame: np.ndarray) -> Generator[EntityYoloDetection, None, None]:
        results = self.model(
            frame,
            classes=self.classes,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
        result: Results = results[0]
        if result.boxes is None:
            return []
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_id = int(box.cls[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            detections.append(EntityYoloDetection(
                x1, y1, x2, y2, class_id, confidence))
        return detections
