from typing import List
from data_class.yolo_detections import YoloDetections
from data_class.yolo_bounding_box import YoloBoundingBox
from data_class.yolo_keypoints import YoloKeypoints

IOU_THRESHOLD = 0.5
MARGIN = 50
KEYPOINT_CONFIDENCE_THRESHOLD = 0.05

class YoloVerificationProcessor:
    def __init__(self):
        self.iou_threshold = IOU_THRESHOLD
        self.margin = MARGIN
        self.keypoint_confidence_threshold = KEYPOINT_CONFIDENCE_THRESHOLD

    def verification_person_detections(self,person_detections: List[YoloDetections]) -> List[YoloDetections]:
        return_person_detections = []

        for person_detection in person_detections:
            bounding_box = person_detection.get_bounding_box()
            if self._is_out_of_frame(bounding_box, 1920, 1080):
                continue
            return_person_detections.append(person_detection)

        return return_person_detections

    def _calculate_iou(self, box1: YoloBoundingBox, box2: YoloBoundingBox) -> float:
        """
        2つのバウンディングボックスのIoUを計算する。
        """
        x1, y1, x2, y2 = box1.get_coordinate()
        x1_2, y1_2, x2_2, y2_2 = box2.get_coordinate()
        x_left = max(x1, x1_2)
        y_top = max(y1, y1_2)
        x_right = min(x2, x2_2)
        y_bottom = min(y2, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

        union_area = box1_area + box2_area - intersection_area

        iou = intersection_area / union_area
        return iou

    def _is_overlap(self, box1: YoloBoundingBox, box2: YoloBoundingBox) -> bool:
        iou = self._calculate_iou(box1, box2)
        if iou < self.iou_threshold:
            return True
        return False

    def _is_out_of_frame(self, box: YoloBoundingBox, image_width: int, image_height: int) -> bool:
        """バウンディングボックスが画面から見切れているか判定する。"""
        x1, y1, x2, y2 = box.get_coordinate()
        if x1 <= self.margin or y1 <= self.margin or x2 >= (image_width - self.margin) or y2 >= (image_height - self.margin):
            return True
        return False

    def _is_keypoint_occluded(self, keypoints: YoloKeypoints) -> bool:
        """キーポイントが信用できるか判定する。"""
        keypoint_list = keypoints.get_list()
        for keypoint in keypoint_list:
            if keypoint.get_confidence() < self.keypoint_confidence_threshold:
                return False
        return True
