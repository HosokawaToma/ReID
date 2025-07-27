"""ReID特徴抽出モジュール"""
from typing import Tuple
import logging

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import timm

# 各モデルのファクトリ関数をインポート
from models.clip_reid.make_model_clipreid import make_model as make_clip_model
from models.la_transformer.model import LATransformerTest
from models.trans_reid.make_model import make_model as make_transreid_model
from config import (
    APP_CONFIG, REID_CONFIG,
    CLIP_REID_CONFIG, TRANS_REID_CONFIG, LA_TRANSFORMER_CONFIG
)
from exceptions import PersonIdentificationError

logger = logging.getLogger(__name__)


class ReIDManager:
    """ReID特徴抽出専用クラス"""

    def __init__(self, backend: str = "clip"):
        """
        ReIDManagerを初期化する

        :param backend: 使用するReIDモデルのバックエンド ("clip", "trans_reid", "la_transformer")
        """
        self.backend = backend
        self.model = None
        self.cfg = None
        self.transform = None

        # モデルを初期化
        self._initialize_model(backend)

    def _initialize_model(self, backend: str) -> None:
        """
        指定されたバックエンドのReIDモデルを初期化する

        :param backend: 使用するReIDモデルのバックエンド
        :raises PersonIdentificationError: モデル初期化に失敗した場合
        """
        logger.info(f"{backend} ReIDモデルの初期化を開始...")

        try:
            if backend == "clip":
                self.cfg = CLIP_REID_CONFIG

                # Market1501データセットの設定
                num_classes = REID_CONFIG.num_classes  # Market1501データセット用
                camera_num = REID_CONFIG.camera_num    # Market1501のカメラ数
                view_num = REID_CONFIG.view_num        # ビュー数

                self.model = make_clip_model(
                    self.cfg,
                    num_classes,
                    camera_num,
                    view_num,
                    APP_CONFIG.device
                )

                # 重みをロード
                self.model.load_param(
                    REID_CONFIG.clip_model_path,
                    APP_CONFIG.device
                )
                logger.info(
                    f"CLIP-ReIDモデル重み読み込み完了: {REID_CONFIG.clip_model_path}")

            elif backend == "trans_reid":
                self.cfg = TRANS_REID_CONFIG

                # Market1501データセットの設定
                num_classes = REID_CONFIG.num_classes

                self.model = make_transreid_model(
                    self.cfg, num_classes,
                    REID_CONFIG.camera_num,
                    REID_CONFIG.view_num
                )

                self.model.load_param(REID_CONFIG.trans_reid_model_path)

            elif backend == "la_transformer":
                self.cfg = LA_TRANSFORMER_CONFIG

                # ベースモデルをtimmで作成
                vit_base = timm.create_model(
                    self.cfg.MODEL.BACKBONE,
                    pretrained=True,
                    num_classes=751
                )
                self.model = LATransformerTest(vit_base, self.cfg.MODEL.LAMBDA)

                # 重みをロード
                state_dict = torch.load(
                    REID_CONFIG.la_transformer_model_path,
                    map_location=APP_CONFIG.device
                )
                self.model.load_state_dict(state_dict, strict=False)
                logger.info(
                    f'LA-Transformerモデルを読み込み: {len(state_dict)}/{len(state_dict)} パラメータ')

            else:
                raise ValueError(f"不明なReIDバックエンドです: {backend}")

            # モデルをデバイスに移動し、評価モードに設定
            self.model.to(APP_CONFIG.device)
            self.model.eval()

            # 画像変換を初期化
            self._build_transforms()

            logger.info(f"{backend} ReIDモデルが正常にロードされました。")

        except Exception as e:
            logger.error(f"{backend}モデル初期化エラー: {e}")
            raise PersonIdentificationError(f"ReIDモデルの初期化に失敗しました: {e}")

    def _build_transforms(self) -> None:
        """画像の前処理トランスフォームを構築する"""
        input_size = self.cfg.INPUT.SIZE_TRAIN

        transform_list = [
            transforms.Resize(
                input_size,
                interpolation=3
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]

        self.transform = transforms.Compose(transform_list)

    def extract_features(self, image_crop: np.ndarray, camera_id: int, view_id: int = 0) -> np.ndarray:
        """
        切り抜かれた人物画像から特徴量を抽出する

        :param image_crop: 人物の切り抜き画像 (BGR format)
        :param camera_id: カメラID
        :param view_id: ビューID (デフォルト: 0)
        :return: 抽出された特徴量ベクトル
        :raises PersonIdentificationError: 特徴抽出に失敗した場合
        """
        if self.model is None or self.cfg is None:
            raise PersonIdentificationError("ReIDモデルが初期化されていません")

        if image_crop is None or image_crop.size == 0:
            raise PersonIdentificationError("無効な画像が提供されました")

        try:
            # BGR to RGB変換
            image_pil = Image.fromarray(image_crop[:, :, ::-1])
            image_tensor = self.transform(
                image_pil).unsqueeze(0).to(APP_CONFIG.device)

            with torch.no_grad():
                if self.backend == "clip":
                    features = self.model(x=image_tensor, get_image=True)

                elif self.backend == "trans_reid":
                    features = self.model(image_tensor, cam_label=camera_id,
                                          view_label=view_id)

                elif self.backend == "la_transformer":
                    features = self.model(image_tensor)

                    # LA-Transformerは(1, 14, 768)の形状で返す
                    # 論文通り：各パート特徴を保持（単純結合しない）
                    if features.is_cuda:
                        features = features.cpu()

                    # 論文通り：14個のパート特徴を結合して高次元特徴ベクトルを作成
                    # Shape: (1, 14, 768) → (1, 10752)
                    # これにより各パートの局所情報を全て保持
                    batch_size = features.size(0)
                    features = features.view(batch_size, -1)
                    features = features.to(APP_CONFIG.device)

                else:
                    raise ValueError(f"不明なReIDバックエンドです: {self.backend}")

            # 特徴ベクトルの検証
            if features.numel() == 0:
                raise ValueError(f"{self.backend}モデルから空の特徴ベクトルが返されました")

            # L2正規化を適用
            normalized_features = torch.nn.functional.normalize(
                features, p=2, dim=1)

            # NumPy配列に変換
            features_numpy = normalized_features.cpu().numpy()

            # バッチ次元を除去 (1, feature_dim) -> (feature_dim,)
            if features_numpy.ndim == 2 and features_numpy.shape[0] == 1:
                features_numpy = features_numpy.squeeze(0)

            logger.debug(
                f"特徴抽出完了: 形状={features_numpy.shape}, バックエンド={self.backend}")
            return features_numpy

        except Exception as e:
            logger.error(f"特徴抽出エラー: {e}")
            raise PersonIdentificationError(f"特徴抽出に失敗しました: {e}")
