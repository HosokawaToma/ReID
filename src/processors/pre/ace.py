import numpy as np
from library.pre_processing.ace import ace_filter
from .base import BasePreProcessor

class AcePreProcessor(BasePreProcessor):
    def __init__(self):
        super().__init__()

    def process(self, image: np.ndarray) -> np.ndarray:
        image_tensor = self._np_image_to_tensor(image)
        image_tensor.to(self.device)
        ace_image_tensor = ace_filter(image_tensor)
        ace_image = self._tensor_image_to_np(ace_image_tensor)
        return ace_image
