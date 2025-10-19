import os
from PIL import Image
from torchvision import transforms
from library.reid_models.clip_reid.datasets.make_dataloader_clipreid import make_dataloader as make_clip_dataloader
from library.reid_models.clip_reid.model.make_model_clipreid import make_model as make_clip_model
from library.reid_models.clip_reid.config import cfg
import torch
import numpy as np
from application.reid.reid_model import ApplicationReidModel


class ApplicationClipReidModel(ApplicationReidModel):
    CONFIG_FILE_PATH = "src/application/reid/clip_reid/configs/person/vit_clipreid.yml"
    OPTIONS = [
        "MODEL.SIE_CAMERA",
        "True",
        "MODEL.SIE_COE",
        "1.0",
        "MODEL.STRIDE_SIZE",
        "[12, 12]",
        "DATASETS.ROOT_DIR",
        "dataset",
        "MODEL.PRETRAIN_PATH",
        "models/jx_vit_base_p16_224-80ecf9dd.pth",
        "TEST.WEIGHT",
        "models/Market1501_clipreid_12x12sie_ViT-B-16_60.pth"
    ]

    def __init__(self):
        cfg.merge_from_file(self.CONFIG_FILE_PATH)
        cfg.merge_from_list(self.OPTIONS)
        cfg.freeze()
        self.config = cfg

        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
        _, _, _, _, num_classes, camera_num, view_num = make_clip_dataloader(
            self.config)
        model = make_clip_model(
            self.config, num_class=num_classes, camera_num=camera_num, view_num=view_num)

        model.load_param(self.config.TEST.WEIGHT)
        model.eval()
        model.to(self.config.MODEL.DEVICE)
        self.model = model

        self.transform = transforms.Compose([
            transforms.Resize(self.config.INPUT.SIZE_TEST),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.INPUT.PIXEL_MEAN,
                std=self.config.INPUT.PIXEL_STD
            )
        ])

    def extract_feature(
        self,
        image: np.ndarray,
        camera_id: int = 0,
        view_id: int = 0,
    ) -> torch.Tensor:
        image_pil = Image.fromarray(image[:, :, ::-1])
        image_tensor = self.transform(image_pil).unsqueeze(
            0).to(self.config.MODEL.DEVICE)
        camera_id_tensor = None
        view_id_tensor = None

        if self.config.MODEL.SIE_CAMERA:
            camera_id_tensor = torch.tensor(
                camera_id, dtype=torch.long).to(self.config.MODEL.DEVICE)

        if self.config.MODEL.SIE_VIEW:
            view_id_tensor = torch.tensor(
                view_id, dtype=torch.long).to(self.config.MODEL.DEVICE)

        with torch.no_grad():
            feature = self.model(
                image_tensor, cam_label=camera_id_tensor, view_label=view_id_tensor)

        return feature
