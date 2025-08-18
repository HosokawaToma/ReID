import numpy as np
import torch
import cv2
from library.pre_processing.clahe import clahe_gpu

class ClahePreProcessor:
    def __init__(self, device: str = "cpu"):
        self.device = device

    def _np_image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb_float = image_rgb.astype(np.float32)
        image_rgb_normalized = image_rgb_float / 255.0
        image_rgb_tensor = torch.from_numpy(
            image_rgb_normalized).permute(2, 0, 1).unsqueeze(0)
        return image_rgb_tensor

    def _tensor_image_to_np(self, image: torch.Tensor) -> np.ndarray:
        image_rgb = image.squeeze(0).permute(1, 2, 0).numpy() * 255.0
        image_rgb_uint8 = image_rgb.astype(np.uint8)
        output_bgr_np = cv2.cvtColor(image_rgb_uint8, cv2.COLOR_RGB2BGR)
        return output_bgr_np

    def clahe(self, image: np.ndarray) -> np.ndarray:
        image_rgb_tensor = self._np_image_to_tensor(image)
        image_rgb_tensor.to(self.device)
        clahe_image_rgb_tensor = clahe_gpu(image_rgb_tensor)
        clahe_image = self._tensor_image_to_np(clahe_image_rgb_tensor)
        return clahe_image
