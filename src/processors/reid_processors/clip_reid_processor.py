import logging
import numpy as np
import torch
from library.reid_models.clip_reid.config import cfg
from library.reid_models.clip_reid.model.make_model_clipreid import make_model as make_clip_model
from library.reid_models.clip_reid.datasets.make_dataloader_clipreid import make_dataloader as make_clip_dataloader
from torchvision import transforms
from PIL import Image

CONFIG_FILE_PATH = "src/library/reid_models/clip_reid/configs/person/vit_clipreid.yml"
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

class ClipReIDProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = None
        self.model = None
        self.transform = None
        self._initialize_config()
        self._initialize_model()
        self._initialize_transform()

    def _initialize_config(self) -> None:
        """設定の初期化"""
        cfg.merge_from_file(CONFIG_FILE_PATH)
        cfg.merge_from_list(OPTIONS)
        cfg.freeze()
        self.config = cfg

    def _initialize_model(self) -> None:
        """CLIP-ReIDモデルの初期化"""
        _, _, _, _, num_classes, camera_num, view_num = make_clip_dataloader(self.config)
        model = make_clip_model(
            self.config, num_class=num_classes, camera_num=camera_num, view_num=view_num)

        model.load_param(self.config.TEST.WEIGHT)
        self.device = self.config.MODEL.DEVICE
        self.sie_camera = self.config.MODEL.SIE_CAMERA
        self.sie_view = self.config.MODEL.SIE_VIEW
        self.model = model

    def _initialize_transform(self) -> None:
        """トランスフォームの初期化"""
        self.transform = transforms.Compose([
            transforms.Resize(self.config.INPUT.SIZE_TEST),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.INPUT.PIXEL_MEAN,
                std=self.config.INPUT.PIXEL_STD
            )
        ])

    def extract_features(
        self,
        image: np.ndarray,
        camera_id: int = 0,
        view_id: int = 0,
    ) -> torch.Tensor:
        """
        切り抜かれた人物画像から特徴量を抽出する

        :param image_crop: 人物の切り抜き画像 (BGR format)
        :param camera_id: カメラID (デフォルト: 0)
        :param view_id: ビューID (デフォルト: 0)
        :return: L2正規化された特徴量ベクトル
        :raises Exception: 特徴抽出に失敗した場合
        """

        image_pil = Image.fromarray(image[:, :, ::-1])
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        camera_id_tensor = None
        view_id_tensor = None

        if self.config.MODEL.SIE_CAMERA:
            camera_id_tensor = torch.tensor(camera_id, dtype=torch.long)
            camera_id_tensor.to(self.device)

        if self.config.MODEL.SIE_VIEW:
            view_id_tensor = torch.tensor(view_id, dtype=torch.long)
            view_id_tensor.to(self.device)

        with torch.no_grad():
            feat = self.model(image_tensor, cam_label=camera_id_tensor, view_label=view_id_tensor)

        return feat
