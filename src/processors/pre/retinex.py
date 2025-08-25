import numpy as np
from library.pre_processing.retinex import stable_retinex_gpu
from .base import BasePreProcessor


class RetinexPreProcessor(BasePreProcessor):
    def __init__(self, device: str):
        self._device = device

    def process(self, image: np.ndarray) -> np.ndarray:
        image_tensor = self._np_image_to_tensor(image)
        image_tensor.to(self._device)
        retinex_image_tensor = stable_retinex_gpu(image_tensor, [15, 80, 250])
        retinex_image = self._tensor_image_to_np(retinex_image_tensor)
        return retinex_image
