from dataclasses import dataclass
import torch

@dataclass
class PersonDataSetFeatures:
    _persons_id: list[int]
    _cameras_id: list[int]
    _views_id: list[int]
    _features: torch.Tensor
    _device: str

    def __init__(
        self,
        persons_id: list[int] = [],
        cameras_id: list[int] = [],
        views_id: list[int] = [],
        features: torch.Tensor = torch.Tensor([]),
        device: str = "cpu"
    ):
        self._persons_id = persons_id
        self._cameras_id = cameras_id
        self._views_id = views_id
        self._features = features
        self._device = device
        self._features = self._features.to(self._device)

    def add_feature(self, feat: torch.Tensor) -> None:
        self._features = torch.cat([self._features, feat], dim=0)

    def add_person_id(self, person_id: int) -> None:
        self._persons_id.append(person_id)

    def add_camera_id(self, camera_id: int) -> None:
        self._cameras_id.append(camera_id)

    def add_view_id(self, view_id: int) -> None:
        self._views_id.append(view_id)

    def get_features(self) -> torch.Tensor:
        return self._features

    def get_person_ids(self) -> list[int]:
        return self._persons_id

    def get_camera_ids(self) -> list[int]:
        return self._cameras_id

    def get_view_ids(self) -> list[int]:
        return self._views_id

    def get_num_features(self) -> int:
        return self._features.size(0)
