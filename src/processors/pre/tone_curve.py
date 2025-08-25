import cv2
import numpy as np
from library.pre_processing.tone_curve import apply_reinhard_tonemapping

class ToneCurvePreProcessor:
    def __init__(self):
        pass

    def process(self, image: np.ndarray) -> np.ndarray:
        return apply_reinhard_tonemapping(image, 1.0, 0.5)
