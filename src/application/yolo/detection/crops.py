import numpy as np
from entities.yolo.detection import EntityYoloDetection


class ApplicationYoloDetectionCrops:
    def crop(self, frame: np.ndarray, detection: EntityYoloDetection) -> np.ndarray:
        return frame[detection.y1:detection.y2, detection.x1:detection.x2]
