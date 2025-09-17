from typing import List

from entities.yolo.bounding_box import EntityYoloBoundingBox
from entities.yolo.detections import EntityYoloDetections


class ServiceYoloDetectionsVerification:
    MARGIN = 50

    @staticmethod
    def verify(person_detections: List[EntityYoloDetections]) -> List[EntityYoloDetections]:
        return_person_detections = []

        for person_detection in person_detections:
            bounding_box = person_detection.get_bounding_box()
            if ServiceYoloDetectionsVerification._is_out_of_frame(bounding_box, 1920, 1080):
                continue
            return_person_detections.append(person_detection)

        return return_person_detections

    @staticmethod
    def _is_out_of_frame(box: EntityYoloBoundingBox, image_width: int, image_height: int) -> bool:
        x1, y1, x2, y2 = box.get_coordinate()
        if (x1 <= ServiceYoloDetectionsVerification.MARGIN or
            y1 <= ServiceYoloDetectionsVerification.MARGIN or
            x2 >= (image_width - ServiceYoloDetectionsVerification.MARGIN) or
            y2 >= (image_height - ServiceYoloDetectionsVerification.MARGIN)):
            return True
        return False
