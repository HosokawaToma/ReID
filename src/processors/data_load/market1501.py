from pathlib import Path
from data_class.person_data_set_features import PersonDataSetFeatures
import numpy as np
import cv2
from typing import Callable
import torch

DATA_SET_DIR_STR = "resources/data_sets"

class Market1501DataLoadProcessor:
    def __init__(self, use_data_set_name: str):
        self.market1501_dir_path = Path(DATA_SET_DIR_STR) / use_data_set_name
        self.gallery_dir_path = Path(self.market1501_dir_path / "gallery")
        self.query_dir_path = Path(self.market1501_dir_path / "query")
        self.gallery_features = PersonDataSetFeatures(
            persons_id=[], cameras_id=[], views_id=[], features=torch.tensor([]))
        self.query_features = PersonDataSetFeatures(
            persons_id=[], cameras_id=[], views_id=[], features=torch.tensor([]))

    def load_image(self, file_path: Path) -> tuple[int, int, int, np.ndarray]:
        image = cv2.imread(str(file_path))

        file_stem = file_path.stem
        parts = file_stem.split("_")

        person_id = int(parts[0])

        camera_view_part = parts[1]

        camera_view_parts = camera_view_part[1:].split('s')

        camera_id = int(camera_view_parts[0]) - 1
        view_id = int(camera_view_parts[1]) - 1

        return person_id, camera_id, view_id, image

    def load_gallery(self, extract_feat: Callable[[np.ndarray, int, int], torch.Tensor]) -> None:
        for file_path in self.gallery_dir_path.glob("*"):
            if not file_path.is_file():
                continue
            person_id, camera_id, view_id, image = self.load_image(file_path)
            if person_id == -1:
                continue
            feat = extract_feat(image, camera_id, view_id)
            self.gallery_features.persons_id.append(person_id)
            self.gallery_features.cameras_id.append(camera_id)
            self.gallery_features.views_id.append(view_id)
            self.gallery_features.features = torch.cat([self.gallery_features.features, feat], dim=0)

    def load_query(self, extract_feat: Callable[[np.ndarray, int, int], torch.Tensor]) -> None:
        for file_path in self.query_dir_path.glob("*"):
            if not file_path.is_file():
                continue
            person_id, camera_id, view_id, image = self.load_image(file_path)
            if person_id == -1:
                continue
            feat = extract_feat(image, camera_id, view_id)
            self.query_features.persons_id.append(person_id)
            self.query_features.cameras_id.append(camera_id)
            self.query_features.views_id.append(view_id)
            self.query_features.features = torch.cat([self.query_features.features, feat], dim=0)

    def get_gallery_features(self) -> PersonDataSetFeatures:
        return self.gallery_features

    def get_query_features(self) -> PersonDataSetFeatures:
        return self.query_features
