import numpy as np
from library.pre_processing.clahe import clahe_gpu
from .base import BasePreProcessor

class ClahePreProcessor(BasePreProcessor):
    def __init__(self, device: str):
        self._device = device

    def process(self, image: np.ndarray) -> np.ndarray:
        image_rgb_tensor = self._np_image_to_tensor(image)
        image_rgb_tensor.to(self._device)
        clahe_image_rgb_tensor = clahe_gpu(image_rgb_tensor)
        clahe_image = self._tensor_image_to_np(clahe_image_rgb_tensor)
        return clahe_image
