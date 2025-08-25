import numpy as np
from library.pre_processing.msrcr import msrcr_enhance
from .base import BasePreProcessor

class MsrcrPreProcessor(BasePreProcessor):
    def __init__(self, device: str):
        self._device = device

    def process(self, image: np.ndarray) -> np.ndarray:
        image_tensor = self._np_image_to_tensor(image)
        image_tensor.to(self._device)
        msrcr_image_tensor = msrcr_enhance(image_tensor)
        msrcr_image = self._tensor_image_to_np(msrcr_image_tensor)
        return msrcr_image
