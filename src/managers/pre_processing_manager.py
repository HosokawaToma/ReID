import cv2
import numpy as np
from pre_processing.retinex import MSR, SSR


class PreProcessingManager:
    def __init__(self):
        pass

    def clahe(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image[:, :, 0] = clahe.apply(image[:, :, 0])
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    def retinex(self, image: np.ndarray) -> np.ndarray:
        variance = 300
        img_ssr = SSR(image, variance)
        return img_ssr
