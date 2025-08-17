import torch
import numpy as np
from pre_processing.clahe import clahe_gpu
from pre_processing.retinex import ssr_gpu


class PreProcessingManager:
    def __init__(self):
        pass

    def clahe(self, image: np.ndarray) -> np.ndarray:
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        clahe_image_tensor = clahe_gpu(image_tensor)
        clahe_image = clahe_image_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255.0
        return clahe_image.astype(np.uint8)

    def retinex(self, image: np.ndarray) -> np.ndarray:
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        retinex_image_tensor = ssr_gpu(image_tensor)
        retinex_image = retinex_image_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255.0
        return retinex_image.astype(np.uint8)
