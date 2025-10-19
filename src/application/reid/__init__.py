import os
from PIL import Image
from torchvision import transforms
from library.reid_models.clip_reid.datasets.make_dataloader_clipreid import make_dataloader as make_clip_dataloader
from library.reid_models.clip_reid.model.make_model_clipreid import make_model as make_clip_model
from library.reid_models.clip_reid.config import cfg
import torch
import numpy as np
from application.reid.clip_reid_model import ApplicationClipReidModel
from application.reid.assign_id import ApplicationReIDAssignID

class ApplicationReID:
    def __init__(self, model_type: str = "clip_reid"):
        self.model = ApplicationClipReidModel()
        self.assigner = ApplicationReIDAssignID()

    def extract_feature(self, image: np.ndarray, camera_id: int = 0, view_id: int = 0) -> torch.Tensor:
        return self.model.extract_feature(image, camera_id, view_id)

    def assign_id(self, feature: torch.Tensor) -> int:
        return self.assigner.assign_person_id(feature)
