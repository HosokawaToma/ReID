import cv2
import numpy as np


class PreProcessingManager:
    def __init__(self):
        pass

    def clahe(self, image: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
