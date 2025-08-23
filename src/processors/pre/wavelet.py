import numpy as np
from library.pre_processing.wavelet import wavelet_enhance
from .base import BasePreProcessor

class WaveletPreProcessor(BasePreProcessor):
    def __init__(self):
        super().__init__()

    def process(self, image: np.ndarray) -> np.ndarray:
        image_tensor = self._np_image_to_tensor(image)
        image_tensor.to(self.device)
        wavelet_image_tensor = wavelet_enhance(image_tensor, device=self.device)
        wavelet_image = self._tensor_image_to_np(wavelet_image_tensor)
        return wavelet_image
