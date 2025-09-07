from typing import List
from data_class.yolo_detections import YoloDetections
from data_class.yolo_bounding_box import YoloBoundingBox
from data_class.yolo_keypoints import YoloKeypoints

MARGIN = 50
CONFIDENCE_THRESHOLD = 0.2

class YoloVerificationProcessor:
    def __init__(self):
        self.margin = MARGIN

    def verification_person_detections(self,person_detections: List[YoloDetections]) -> List[YoloDetections]:
        return_person_detections = []

        for person_detection in person_detections:
            bounding_box = person_detection.get_bounding_box()
            if self._is_out_of_frame(bounding_box, 1920, 1080):
                continue
            return_person_detections.append(person_detection)

        return return_person_detections

    def _is_keypoints_confidence(self, keypoints: YoloKeypoints) -> bool:
        keypoint_list = keypoints.get_list()
        for keypoint in keypoint_list:
            if keypoint.get_confidence() < CONFIDENCE_THRESHOLD:
                return False
        return True

    def _is_out_of_frame(self, box: YoloBoundingBox, image_width: int, image_height: int) -> bool:
        """バウンディングボックスが画面から見切れているか判定する。"""
        x1, y1, x2, y2 = box.get_coordinate()
        if x1 <= self.margin or y1 <= self.margin or x2 >= (image_width - self.margin) or y2 >= (image_height - self.margin):
            return True
        return False
