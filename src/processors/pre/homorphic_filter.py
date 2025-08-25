import numpy as np
from library.pre_processing.homorphic_filter import homomorphic_filter
from .base import BasePreProcessor


class HomorphicFilterProcessor(BasePreProcessor):
    def __init__(self, device: str):
        self._device = device

    def process(self, image: np.ndarray) -> np.ndarray:
        image_tensor = self._np_image_to_tensor(image)
        image_tensor.to(self._device)
        homomorphic_image_tensor = homomorphic_filter(image_tensor)
        homomorphic_image = self._tensor_image_to_np(homomorphic_image_tensor)
        return homomorphic_image
