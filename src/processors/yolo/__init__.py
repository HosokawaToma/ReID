import numpy as np
from typing import List
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Keypoints
from data_class.yolo_detections import YoloDetections
from data_class.yolo_bounding_box import YoloBoundingBox
from data_class.yolo_keypoints import YoloKeypoints
from data_class.yolo_keypoint import YoloKeypoint

MODEL_PATH = "models/yolo11n-pose.pt"
CONFIDENCE_THRESHOLD = 0.5
PERSON_CLASS_ID = 0
VERBOSE = False
TRACKER = "bytetrack.yaml"
DATA = "coco-pose.yaml"

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
            tracker=self.tracker,
            data=DATA
        )

        if not results or len(results) == 0:
            return []

        detections = []
        result = results[0]

        if result.boxes is not None and result.keypoints is not None:
            for box, keypoints in zip(result.boxes, result.keypoints):
                box: Boxes
                keypoints: Keypoints
                box_id = int(box.id[0].cpu().numpy())
                box_xyxy = box.xyxy[0].cpu().numpy()
                box_xywh = box.xywh[0].cpu().numpy()
                box_conf = box.conf[0].cpu().numpy()
                box_cls = int(box.cls[0].cpu().numpy())
                bounding_box = YoloBoundingBox(
                    int(box_xyxy[0]), int(box_xyxy[1]), int(box_xyxy[2]), int(box_xyxy[3]), int(box_xywh[2]), int(box_xywh[3]))
                keypoint_xys = keypoints.xy[0].cpu().numpy()
                keypoint_confs = keypoints.conf[0].cpu().numpy()
                yolo_keypoints = YoloKeypoints(
                    YoloKeypoint(keypoint_xys[0][0], keypoint_xys[0][1], keypoint_confs[0]),
                    YoloKeypoint(keypoint_xys[1][0], keypoint_xys[1][1], keypoint_confs[1]),
                    YoloKeypoint(keypoint_xys[2][0], keypoint_xys[2][1], keypoint_confs[2]),
                    YoloKeypoint(keypoint_xys[3][0], keypoint_xys[3][1], keypoint_confs[3]),
                    YoloKeypoint(keypoint_xys[4][0], keypoint_xys[4][1], keypoint_confs[4]),
                    YoloKeypoint(keypoint_xys[5][0], keypoint_xys[5][1], keypoint_confs[5]),
                    YoloKeypoint(keypoint_xys[6][0], keypoint_xys[6][1], keypoint_confs[6]),
                    YoloKeypoint(keypoint_xys[7][0], keypoint_xys[7][1], keypoint_confs[7]),
                    YoloKeypoint(keypoint_xys[8][0], keypoint_xys[8][1], keypoint_confs[8]),
                    YoloKeypoint(keypoint_xys[9][0], keypoint_xys[9][1], keypoint_confs[9]),
                    YoloKeypoint(keypoint_xys[10][0], keypoint_xys[10][1], keypoint_confs[10]),
                    YoloKeypoint(keypoint_xys[11][0], keypoint_xys[11][1], keypoint_confs[11]),
                    YoloKeypoint(keypoint_xys[12][0], keypoint_xys[12][1], keypoint_confs[12]),
                    YoloKeypoint(keypoint_xys[13][0], keypoint_xys[13][1], keypoint_confs[13]),
                    YoloKeypoint(keypoint_xys[14][0], keypoint_xys[14][1], keypoint_confs[14]),
                    YoloKeypoint(keypoint_xys[15][0], keypoint_xys[15][1], keypoint_confs[15]),
                    YoloKeypoint(keypoint_xys[16][0], keypoint_xys[16][1], keypoint_confs[16])
                )
                detections.append(YoloDetections(
                    box_id, bounding_box, yolo_keypoints, box_conf, box_cls
                ))

        return detections

    def verification_person_detections(person_detections: List[YoloDetections]) -> List[YoloDetections]:
        return_person_detections = []
        for person_detection in person_detections:
            for return_person_detection in return_person_detections:
                if person_detection.get_detection_id() == return_person_detection.get_detection_id():
                    return_person_detections.remove(return_person_detection)
            return_person_detections.append(person_detection)
        return return_person_detections
