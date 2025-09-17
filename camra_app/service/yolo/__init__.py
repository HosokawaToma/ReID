from typing import List

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.models.yolo import YOLO

from entities.yolo.bounding_box import EntityYoloBoundingBox
from entities.yolo.detections import EntityYoloDetections

MODEL_PATH = "resources/models/yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.3
IMGSZ = (640, 640)
AGNOSTIC_NMS = False
PERSON_CLASS_ID = 0
VERBOSE = False
TRACKER = "bytetrack.yaml"
DATA = "coco-pose.yaml"

class ServiceYolo:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self.iou_threshold = IOU_THRESHOLD
        self.imgsz = IMGSZ
        self.agnostic_nms = AGNOSTIC_NMS
        self.person_class_id = PERSON_CLASS_ID
        self.verbose = VERBOSE
        self.tracker = TRACKER

    def extract_person_detections(self, frame: np.ndarray) -> List[EntityYoloDetections]:
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
            return []

        detections = []
        result = results[0]

        if result.boxes is not None:
            for box in result.boxes:
                box: Boxes
                if box.id is not None:
                    box_id = int(box.id[0].cpu().numpy())
                else:
                    box_id = None
                box_xyxy = box.xyxy[0].cpu().numpy()
                box_x1 = int(box_xyxy[0])
                box_y1 = int(box_xyxy[1])
                box_x2 = int(box_xyxy[2])
                box_y2 = int(box_xyxy[3])
                box_xywh = box.xywh[0].cpu().numpy()
                box_width = int(box_xywh[2])
                box_height = int(box_xywh[3])
                box_conf = box.conf[0].cpu().numpy()
                box_cls = int(box.cls[0].cpu().numpy())
                bounding_box = EntityYoloBoundingBox(
                        box_x1, box_y1, box_x2, box_y2, box_width, box_height)
                detections.append(EntityYoloDetections(
                    box_id, bounding_box, box_conf, box_cls
                ))

        return detections
