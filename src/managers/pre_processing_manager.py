import cv2
import numpy as np


class PreProcessingManager:
    def __init__(self):
        pass

    def clahe(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image
