from dataclasses import dataclass
import torch

@dataclass
class PersonDataSetFeatures:
    persons_id: list[int]
    cameras_id: list[int]
    views_id: list[int]
    features: torch.Tensor

    def __init__(self, persons_id: list[int] = [], cameras_id: list[int] = [], views_id: list[int] = [], features: torch.Tensor = torch.Tensor([])):
        self.persons_id = persons_id
        self.cameras_id = cameras_id
        self.views_id = views_id
        self.features = features

    def add_feature(self, feat: torch.Tensor) -> None:
        self.features = torch.cat([self.features, feat], dim=0)

    def add_person_id(self, person_id: int) -> None:
        self.persons_id.append(person_id)

    def add_camera_id(self, camera_id: int) -> None:
        self.cameras_id.append(camera_id)

    def add_view_id(self, view_id: int) -> None:
        self.views_id.append(view_id)
