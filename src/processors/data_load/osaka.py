import os
from pathlib import Path
from typing import List
import cv2

class OsakaDataLoadProcessor:
    @staticmethod
    def load_data(file_path: Path) -> tuple[int, int, int, int, int, int]:
        image = cv2.imread(str(file_path))

        file_stem = file_path.stem
        parts = file_stem.split("_")

        person_id = int(parts[0])

        camera_id = int(parts[1])
        view_id = int(parts[2])

        return person_id, camera_id, view_id, image
