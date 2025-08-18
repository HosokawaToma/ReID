"""ReID特徴抽出モジュール"""
from reid_models.clip_reid.config import cfg as clip_cfg
from reid_models.clip_reid.model.make_model_clipreid import make_model as make_clip_model
from reid_models.clip_reid.datasets.make_dataloader_clipreid import make_dataloader as make_clip_dataloader
from reid_models.trans_reid.config import cfg as trans_cfg
from reid_models.trans_reid.model.make_model import make_model as make_transreid_model
from reid_models.trans_reid.datasets.make_dataloader import make_dataloader as make_trans_dataloader
import logging
import numpy as np
from dataclasses import dataclass
from typing import List
import torch
from PIL import Image
from torchvision import transforms

@dataclass
class ClipReIDConfig:
    """Configuration for CLIP-ReID model, matching YACS defaults."""
    class Config:
        FILE_PATH: str = "src/reid_models/clip_reid/configs/person/vit_clipreid.yml"
        OPTIONS: List[str] = [
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

@dataclass
class TransReIDConfig:
    """Configuration for the Transformer‑based ReID model."""
    class Config:
        FILE_PATH: str = "src/reid_models/trans_reid/configs/Market/vit_transreid_stride.yml"
        OPTIONS: List[str] = [
            "DATASETS.ROOT_DIR",
            "dataset",
            "MODEL.PRETRAIN_PATH",
            "models/jx_vit_base_p16_224-80ecf9dd.pth",
            "TEST.WEIGHT",
            "models/vit_transreid_market.pth"
        ]

CLIP_REID_CONFIG = ClipReIDConfig()
TRANS_REID_CONFIG = TransReIDConfig()

class ReIDModelManager:
    """ReID特徴抽出専用クラス"""

    def __init__(self, backend: str = "clip"):
        """
        ReIDManagerを初期化する

        :param backend: 使用するReIDモデルのバックエンド ("clip", "trans_reid", "la_transformer")
        """
        print(f"ReIDModelManager初期化開始: {backend}")
        self.backend = backend
        self.model = None
        self.transform = None
        self.device = None
        self.sie_camera = False
        self.sie_view = False
        self.logger = logging.getLogger(__name__)

        self._initialize_model(backend)
        print(f"ReIDModelManager初期化完了: {backend}")

    def _initialize_model(self, backend: str) -> None:
        """
        指定されたバックエンドのReIDモデルを初期化する

        :param backend: 使用するReIDモデルのバックエンド
        :raises Exception: モデル初期化に失敗した場合
        """
        self.logger.info(f"{backend} ReIDモデルの初期化を開始...")

        if backend == "clip":
            self._initialize_clip_model()
        elif backend == "trans_reid":
            self._initialize_transreid_model()
        else:
            raise ValueError(f"不明なReIDバックエンドです: {backend}")

        self.model.to(self.device)
        self.model.eval()

        self.logger.info(f"{backend} ReIDモデルが正常にロードされました。")

    def _initialize_clip_model(self) -> None:
        """CLIP-ReIDモデルの初期化"""
        print("CLIP-ReIDモデル初期化開始...")
        clip_cfg.merge_from_file(CLIP_REID_CONFIG.Config.FILE_PATH)
        clip_cfg.merge_from_list(CLIP_REID_CONFIG.Config.OPTIONS)
        clip_cfg.freeze()
        _, _, _, _, num_classes, camera_num, view_num = make_clip_dataloader(
            clip_cfg)
        model = make_clip_model(
            clip_cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

        self.logger.info(f"CLIP-ReIDモデル作成完了")

        model.load_param(clip_cfg.TEST.WEIGHT)
        self.device = clip_cfg.MODEL.DEVICE
        self.sie_camera = clip_cfg.MODEL.SIE_CAMERA
        self.sie_view = clip_cfg.MODEL.SIE_VIEW
        self.model = model

        self.logger.info(
            f"CLIP-ReIDモデル重み読み込み完了: {clip_cfg.TEST.WEIGHT}")

        self.transform = transforms.Compose([
            transforms.Resize(clip_cfg.INPUT.SIZE_TEST),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=clip_cfg.INPUT.PIXEL_MEAN,
                std=clip_cfg.INPUT.PIXEL_STD
            )
        ])

        self.logger.info(
            f"CLIP-ReIDモデルのトランスフォーム設定完了: {clip_cfg.INPUT.SIZE_TEST}")

    def _initialize_transreid_model(self) -> None:
        """TransReIDモデルの初期化"""
        print("TransReIDモデル初期化開始...")
        trans_cfg.merge_from_file(TRANS_REID_CONFIG.Config.FILE_PATH)
        trans_cfg.merge_from_list(TRANS_REID_CONFIG.Config.OPTIONS)
        trans_cfg.freeze()

        _, _, _, _, num_classes, camera_num, view_num = make_trans_dataloader(
            trans_cfg)
        model = make_transreid_model(
            trans_cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

        self.logger.info(f"TransReIDモデル作成完了")

        model.load_param(trans_cfg.TEST.WEIGHT)
        self.device = trans_cfg.MODEL.DEVICE
        self.sie_camera = trans_cfg.MODEL.SIE_CAMERA
        self.sie_view = trans_cfg.MODEL.SIE_VIEW
        self.model = model

        self.logger.info(
            f"TransReIDモデル重み読み込み完了: {trans_cfg.TEST.WEIGHT}")

        self.transform = transforms.Compose([
            transforms.Resize(trans_cfg.INPUT.SIZE_TEST),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=trans_cfg.INPUT.PIXEL_MEAN,
                std=trans_cfg.INPUT.PIXEL_STD
            )
        ])

        self.logger.info(
            f"TransReIDモデルのトランスフォーム設定完了: {trans_cfg.INPUT.SIZE_TEST}")

    def extract_features(
        self,
        image: torch.Tensor,
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
        if self.model is None:
            raise Exception("ReIDモデルが初期化されていません")

        if image is None or image.size == 0:
            raise Exception("無効な画像が提供されました")
        image = self.transform(image)
        camera_id_tensor = None
        view_id_tensor = None

        if self.sie_camera:
            camera_id_tensor = torch.tensor(camera_id, dtype=torch.long)
            camera_id_tensor.to(self.device)

        if self.sie_view:
            view_id_tensor = torch.tensor(view_id, dtype=torch.long)
            view_id_tensor.to(self.device)

        with torch.no_grad():
            if self.backend == "clip":
                feat = self.model(
                    image, cam_label=camera_id_tensor, view_label=view_id_tensor)
                feat = torch.nn.functional.normalize(feat, dim=1, p=2)

            elif self.backend == "trans_reid":
                feat = self.model(
                    image_tensor, cam_label=camera_id_tensor, view_label=view_id_tensor)
                feat = torch.nn.functional.normalize(feat, dim=1, p=2)

            else:
                raise ValueError(f"不明なReIDバックエンドです: {self.backend}")

        if feat.numel() == 0:
            raise ValueError(f"{self.backend}モデルから空の特徴ベクトルが返されました")

        self.logger.debug(
            f"特徴抽出完了: 形状={feat}, バックエンド={self.backend}")
        return feat
