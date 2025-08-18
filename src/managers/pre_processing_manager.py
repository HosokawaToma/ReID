import torch
import numpy as np
from pre_processing.clahe import clahe_gpu
from pre_processing.retinex import ssr_gpu
import cv2

class PreProcessingManager:
    def __init__(self):
        pass

    def np_image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb_float = image_rgb.astype(np.float32)
        image_rgb_normalized = image_rgb_float / 255.0
        image_rgb_tensor = torch.from_numpy(image_rgb_normalized).permute(2, 0, 1).unsqueeze(0)
        return image_rgb_tensor

    def tensor_image_output(self, image: torch.Tensor, output_path: str) -> None:
        image_rgb = image.squeeze(0).permute(1, 2, 0).numpy() * 255.0
        image_rgb_uint8 = image_rgb.astype(np.uint8)
        cv2.imwrite(output_path, image_rgb_uint8)

    def clahe(self, image: torch.Tensor) -> torch.Tensor:
        clahe_image_tensor = clahe_gpu(image)
        return clahe_image_tensor

    def retinex(self, image: torch.Tensor) -> torch.Tensor:
        retinex_image_tensor = ssr_gpu(image)
        return retinex_image_tensor
