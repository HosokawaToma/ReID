import numpy as np
import cv2


class ApplicationVideoFrameCorrectionGamma:
    def __init__(self, gamma: float = 1.1):
        self.gamma = gamma
        self.inv_gamma = 1.0 / gamma
        self.table = np.array([((i / 255.0) ** self.inv_gamma) * 255
                               for i in np.arange(0, 256)]).astype("uint8")

    def correct(self, frame):
        return cv2.LUT(frame, self.table)
