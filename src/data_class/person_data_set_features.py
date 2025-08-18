from dataclasses import dataclass
import torch

@dataclass
class PersonDataSetFeatures:
    persons_id: list[int]
    cameras_id: list[int]
    views_id: list[int]
    features: torch.Tensor

    def __init__(self, persons_id: list[int], cameras_id: list[int], views_id: list[int], features: torch.Tensor):
        self.persons_id = persons_id
        self.cameras_id = cameras_id
        self.views_id = views_id
        self.features = features
