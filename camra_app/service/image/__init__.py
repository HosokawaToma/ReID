import io

import cv2
import numpy as np
from PIL import Image


class ServiceImage:
    def encode_image(self, image: np.ndarray) -> bytes:
        if image.dtype != np.uint8:
            image = (
                image * 255).astype(np.uint8)
        image_rgb = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        return image_bytes
