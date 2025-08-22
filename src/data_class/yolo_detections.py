import numpy as np
from dataclasses import dataclass
from data_class.yolo_bounding_box import YoloBoundingBox
from data_class.yolo_keypoints import YoloKeypoints

@dataclass
class YoloDetections:
    _detection_id: int
    _bounding_box: YoloBoundingBox
    _keypoints: YoloKeypoints
    _confidence: float
    _class: int

    def __init__(self, detection_id: int, bounding_box: YoloBoundingBox, keypoints: YoloKeypoints, confidence: float, cls: int):
        self._detection_id = detection_id
        self._bounding_box = bounding_box
        self._keypoints = keypoints
        self._confidence = confidence
        self._class = cls

    def get_confidence(self) -> float:
        return self._confidence

    def get_class(self) -> int:
        return self._class

    def get_detection_id(self) -> int:
        return self._detection_id

    def get_bounding_box(self) -> YoloBoundingBox:
        return self._bounding_box

    def get_keypoints(self) -> YoloKeypoints:
        return self._keypoints

