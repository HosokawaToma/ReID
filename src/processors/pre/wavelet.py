import numpy as np
from library.pre_processing.wavelet import (
    wavelet_enhance,
    wavelet_brightness_uniformity,
    wavelet_adaptive_brightness_uniformity
)
from .base import BasePreProcessor


class WaveletPreProcessor(BasePreProcessor):
    def __init__(self, device: str,
                 brightness_uniformity: bool = True,
                 edge_enhancement: bool = True,
                 brightness_factor: float = 1.2,
                 edge_factor: float = 1.5,
                 preserve_contrast: bool = True,
                 adaptive_brightness: bool = False,
                 window_size: int = 32,
                 adaptive_strength: float = 0.5):
        self._device = device
        self._brightness_uniformity = brightness_uniformity
        self._edge_enhancement = edge_enhancement
        self._brightness_factor = brightness_factor
        self._edge_factor = edge_factor
        self._preserve_contrast = preserve_contrast
        self._adaptive_brightness = adaptive_brightness
        self._window_size = window_size
        self._adaptive_strength = adaptive_strength

    def process(self, image: np.ndarray) -> np.ndarray:
        image_tensor = self._np_image_to_tensor(image)
        image_tensor.to(self._device)

        if self._adaptive_brightness:
            # 適応的明度均一化を使用
            wavelet_image_tensor = wavelet_adaptive_brightness_uniformity(
                image_tensor,
                window_size=self._window_size,
                strength=self._adaptive_strength
            )
        else:
            # 従来の明度均一化とエッジ強調
            wavelet_image_tensor = wavelet_enhance(
                image_tensor,
                brightness_uniformity=self._brightness_uniformity,
                edge_enhancement=self._edge_enhancement,
                brightness_factor=self._brightness_factor,
                edge_factor=self._edge_factor,
                preserve_contrast=self._preserve_contrast
            )

        wavelet_image = self._tensor_image_to_np(wavelet_image_tensor)
        return wavelet_image

    def process_brightness_only(self, image: np.ndarray) -> np.ndarray:
        """明度均一化のみを実行するメソッド"""
        image_tensor = self._np_image_to_tensor(image)
        image_tensor.to(self._device)

        if self._adaptive_brightness:
            # 適応的明度均一化を使用
            wavelet_image_tensor = wavelet_adaptive_brightness_uniformity(
                image_tensor,
                window_size=self._window_size,
                strength=self._adaptive_strength
            )
        else:
            # 従来の明度均一化
            wavelet_image_tensor = wavelet_brightness_uniformity(
                image_tensor,
                factor=self._brightness_factor,
                preserve_contrast=self._preserve_contrast
            )

        wavelet_image = self._tensor_image_to_np(wavelet_image_tensor)
        return wavelet_image

    def process_adaptive_brightness(self, image: np.ndarray) -> np.ndarray:
        """適応的明度均一化のみを実行するメソッド"""
        image_tensor = self._np_image_to_tensor(image)
        image_tensor.to(self._device)

        wavelet_image_tensor = wavelet_adaptive_brightness_uniformity(
            image_tensor,
            window_size=self._window_size,
            strength=self._adaptive_strength
        )

        wavelet_image = self._tensor_image_to_np(wavelet_image_tensor)
        return wavelet_image
