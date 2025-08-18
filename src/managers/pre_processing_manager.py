import torch
import numpy as np
from pre_processing.clahe import clahe_gpu
from pre_processing.retinex import stable_retinex_gpu
import cv2

class PreProcessingManager:
    def __init__(self, device: str = "cpu"):
        self.device = device

    def _np_image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb_float = image_rgb.astype(np.float32)
        image_rgb_normalized = image_rgb_float / 255.0
        image_rgb_tensor = torch.from_numpy(image_rgb_normalized).permute(2, 0, 1).unsqueeze(0)
        return image_rgb_tensor

    def _tensor_image_to_np(self, image: torch.Tensor) -> np.ndarray:
        image_rgb = image.squeeze(0).permute(1, 2, 0).numpy() * 255.0
        image_rgb_uint8 = image_rgb.astype(np.uint8)
        output_bgr_np = cv2.cvtColor(image_rgb_uint8, cv2.COLOR_RGB2BGR)
        return output_bgr_np

    def np_image_output(self, image: np.ndarray, output_path: str) -> None:
        cv2.imwrite(output_path, image)

    def clahe(self, image: np.ndarray) -> np.ndarray:
        image_tensor = self._np_image_to_tensor(image)
        image_tensor.to(self.device)
        clahe_image_tensor = clahe_gpu(image_tensor)
        clahe_image = self._tensor_image_to_np(clahe_image_tensor)
        return clahe_image

    def retinex(self, image: np.ndarray) -> np.ndarray:
        image_tensor = self._np_image_to_tensor(image)
        image_tensor.to(self.device)
        retinex_image_tensor = stable_retinex_gpu(image_tensor, [2, 5, 10])
        retinex_image = self._tensor_image_to_np(retinex_image_tensor)
        return retinex_image
