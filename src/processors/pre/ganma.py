import cv2
import numpy as np
from library.pre_processing.ganma import adjust_gamma


class GanmaPreProcessor:
    def process(self, image: np.ndarray) -> np.ndarray:
        return adjust_gamma(image, 1.5)
