import numpy as np
from ultralytics import YOLO
from data_class.yolo_mask import YoloMasks

MODEL_PATH = "models/yolo11x-seg.pt"
CONFIDENCE_THRESHOLD = 0.05
IOU_THRESHOLD = 0.95
IMGSZ = (640, 640)
AGNOSTIC_NMS = False
PERSON_CLASS_ID = 0
VERBOSE = False
TRACKER = "bytetrack.yaml"
DATA = "coco-seg.yaml"


class YoloSegProcessor:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self.iou_threshold = IOU_THRESHOLD
        self.imgsz = IMGSZ
        self.agnostic_nms = AGNOSTIC_NMS
        self.person_class_id = PERSON_CLASS_ID
        self.verbose = VERBOSE
        self.tracker = TRACKER

    def extract_person_masks(self, frame: np.ndarray) -> YoloMasks | None:
        results = self.model.track(
            frame,
            persist=True,
            classes=[self.person_class_id],
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=self.verbose,
            tracker=self.tracker,
            data=DATA,
            imgsz=self.imgsz,
            agnostic_nms=self.agnostic_nms
        )

        if not results or len(results) == 0:
            return None
        return YoloMasks(results[0].masks)
