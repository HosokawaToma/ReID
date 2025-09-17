from typing import List

import numpy as np

from entities.yolo.detections import EntityYoloDetections


class ServiceYoloCropPersonFrame:
    @staticmethod
    def crop(frame: np.ndarray, detection: EntityYoloDetections) -> np.ndarray:
        x1, y1, x2, y2 = detection.get_bounding_box().get_coordinate()
        person_cropped_image = frame[y1:y2, x1:x2]
        return person_cropped_image

    @staticmethod
    def crops(frame: np.ndarray, detections: List[EntityYoloDetections]) -> np.ndarray:
        person_cropped_images = []
        for detection in detections:
            person_cropped_image = ServiceYoloCropPersonFrame.crop(frame, detection)
            person_cropped_images.append(person_cropped_image)
        return person_cropped_images
