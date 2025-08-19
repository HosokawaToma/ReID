from pathlib import Path
import numpy as np
import cv2

class Market1501DataLoadProcessor:
    @staticmethod
    def load_image(file_path: Path) -> tuple[int, int, int, np.ndarray]:
        image = cv2.imread(str(file_path))

        file_stem = file_path.stem
        parts = file_stem.split("_")

        person_id = int(parts[0])

        camera_view_part = parts[1]

        camera_view_parts = camera_view_part[1:].split('s')

        camera_id = int(camera_view_parts[0]) - 1
        view_id = int(camera_view_parts[1]) - 1

        return person_id, camera_id, view_id, image
