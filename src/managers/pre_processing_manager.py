import cv2
import numpy as np
from pre_processing.retinex import MSR, SSR


class PreProcessingManager:
    def __init__(self):
        pass

    def clahe(self, image: np.ndarray) -> np.ndarray:
        # 型をuint8にそろえる（floatなどが来てもOKにする）
        img = image
        if img.dtype != np.uint8:
            if np.issubdtype(img.dtype, np.floating) and img.max() <= 1.0:
                img = (img * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img = cv2.normalize(img, None, 0, 255,
                                    cv2.NORM_MINMAX).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        if img.ndim == 2:
            # グレースケール: そのままCLAHE→BGRに戻す
            eq = clahe.apply(img)
            return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

        if img.shape[2] == 4:
            # 透明あり: BGRへ
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # カラー(BGR): L*a*b*のLだけCLAHE
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def retinex(self, image: np.ndarray) -> np.ndarray:
        variance = 300
        img_ssr = SSR(image, variance)
        return img_ssr
