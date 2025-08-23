import numpy as np
from library.pre_processing.anisotropic_diffusion import anisotropic_diffusion
from .base import BasePreProcessor

class AnisotropicDiffusionPreProcessor(BasePreProcessor):
    def __init__(self):
        super().__init__()

    def process(self, image: np.ndarray) -> np.ndarray:
        image_tensor = self._np_image_to_tensor(image)
        image_tensor.to(self.device)
        anisotropic_diffusion_image_tensor = anisotropic_diffusion(
            image_tensor, n_iter=100, delta_t=0.05, kappa=0.5, device=self.device)
        anisotropic_diffusion_image = self._tensor_image_to_np(anisotropic_diffusion_image_tensor)
        return anisotropic_diffusion_image
