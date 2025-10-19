import numpy as np
from entities.yolo.mask import EntityYoloMask
import cv2


class ApplicationYoloSegmentationIsolation:
    def isolate(self, mask: EntityYoloMask) -> np.ndarray:
        image = np.copy(mask.original_frame)
        black_frame = np.zeros(image.shape[:2], np.uint8)
        contour = mask.xy.astype(np.int32).reshape(-1, 1, 2)
        _ = cv2.drawContours(
            black_frame, [contour], -1, (255, 255, 255), cv2.FILLED)
        mask3ch = cv2.cvtColor(black_frame, cv2.COLOR_GRAY2BGR)
        isolated = cv2.bitwise_and(mask3ch, image)
        return isolated
