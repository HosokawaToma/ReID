import numpy as np
from library.pre_processing.logarithmic_transform import logarithmic_transform
from .base import BasePreProcessor


class LogarithmicTransformProcessor(BasePreProcessor):
    def __init__(self):
        super().__init__()

    def process(self, image: np.ndarray) -> np.ndarray:
        image_tensor = self._np_image_to_tensor(image)
        image_tensor.to(self.device)
        logarithmic_image_tensor = logarithmic_transform(image_tensor)
        logarithmic_image = self._tensor_image_to_np(logarithmic_image_tensor)
        return logarithmic_image
