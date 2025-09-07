import cv2
import numpy as np


class GanmaPreProcessor:
    def __init__(self, gamma: float = 1.1):
        self.gamma = gamma
        self.inv_gamma = 1.0 / gamma
        self.table = np.array([((i / 255.0) ** self.inv_gamma) * 255
                               for i in np.arange(0, 256)]).astype("uint8")

    def process(self, image: np.ndarray) -> np.ndarray:
        return cv2.LUT(image, self.table)
