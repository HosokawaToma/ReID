import numpy as np
from typing import List
from ultralytics import YOLO
from data_class.yolo_detections import YoloDetections
from data_class.yolo_bounding_box import YoloBoundingBox

MODEL_PATH = "models/yolo11n-pose.pt"
CONFIDENCE_THRESHOLD = 0.5
PERSON_CLASS_ID = 0
VERBOSE = False
TRACKER = "bytetrack.yaml"

class YoloProcessor:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self.person_class_id = PERSON_CLASS_ID
        self.verbose = VERBOSE
        self.tracker = TRACKER

    def extract_person_detections(self, frame: np.ndarray) -> List[YoloDetections]:
        results = self.model.track(
            frame,
            persist=True,
            classes=[self.person_class_id],
            conf=self.confidence_threshold,
            verbose=self.verbose,
            tracker=self.tracker
        )

        if not results or len(results) == 0:
            return []

        detections = []
        result = results[0]

        if result.boxes is not None:
            for box in result.boxes:
                box_id = int(box.id[0].cpu().numpy())
                box_xyxy = box.xyxy[0].cpu().numpy()
                box_xywh = box.xywh[0].cpu().numpy()
                box_conf = box.conf[0].cpu().numpy()
                box_cls = int(box.cls[0].cpu().numpy())
                bounding_box = YoloBoundingBox(box_xyxy[0], box_xyxy[1], box_xyxy[2], box_xyxy[3])
                detections.append(YoloDetections(
                    box_id, bounding_box, box_xywh, box_conf, box_cls
                ))

        return detections
