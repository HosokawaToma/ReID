"""ReID特徴抽出モジュール"""
from config import (
    APP_CONFIG,
    CLIP_REID_CONFIG,
    TRANS_REID_CONFIG,
    LA_TRANSFORMER_CONFIG,
    DATASET_CONFIG,
    REID_MODEL_PATH_CONFIG
)
from reid_models.trans_reid.make_model import make_model as make_transreid_model
from reid_models.la_transformer.model import LATransformerTest
from reid_models.clip_reid.make_model_clipreid import make_model as make_clip_model
import timm
import logging
import warnings
import numpy as np

import torch
from PIL import Image
from torchvision import transforms

# 警告を完全に抑制
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# 各モデルのファクトリ関数をインポート


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
        self.logger = logging.getLogger(__name__)

        # モデルを初期化
        self._initialize_model(backend)
        print(f"ReIDModelManager初期化完了: {backend}")

    def _initialize_model(self, backend: str) -> None:
        """
        指定されたバックエンドのReIDモデルを初期化する

        :param backend: 使用するReIDモデルのバックエンド
        :raises Exception: モデル初期化に失敗した場合
        """
        self.logger.info(f"{backend} ReIDモデルの初期化を開始...")

        try:
            if backend == "clip":
                self._initialize_clip_model()
            elif backend == "trans_reid":
                self._initialize_transreid_model()
            elif backend == "la_transformer":
                self._initialize_la_transformer_model()
            else:
                raise ValueError(f"不明なReIDバックエンドです: {backend}")

            # モデルをデバイスに移動し、評価モードに設定
            self.model.to(APP_CONFIG.device)
            self.model.eval()

            self.logger.info(f"{backend} ReIDモデルが正常にロードされました。")

        except Exception as e:
            self.logger.error(f"{backend}モデル初期化エラー: {e}")
            raise Exception(f"ReIDモデルの初期化に失敗しました: {e}")

    def _initialize_clip_model(self) -> None:
        """CLIP-ReIDモデルの初期化"""
        print("CLIP-ReIDモデル初期化開始...")
        self.model = make_clip_model(
            CLIP_REID_CONFIG,
            DATASET_CONFIG.Market1501.num_classes,
            DATASET_CONFIG.Market1501.camera_num,
            DATASET_CONFIG.Market1501.view_num,
            APP_CONFIG.device
        )

        # 重みをロード
        self.model.load_param(
            REID_MODEL_PATH_CONFIG.Path.clip_reid,
            APP_CONFIG.device
        )

        self.logger.info(
            f"CLIP-ReIDモデル重み読み込み完了: {REID_MODEL_PATH_CONFIG.Path.clip_reid}")

        self.transform = transforms.Compose([
            transforms.Resize(CLIP_REID_CONFIG.INPUT.SIZE_TRAIN),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=CLIP_REID_CONFIG.INPUT.MEAN,
                std=CLIP_REID_CONFIG.INPUT.STD
            )
        ])

        self.logger.info(
            f"CLIP-ReIDモデルの前処理設定完了: {CLIP_REID_CONFIG.INPUT.SIZE_TRAIN}")
        print("CLIP-ReIDモデル初期化完了")

    def _initialize_transreid_model(self) -> None:
        """TransReIDモデルの初期化"""
        print("TransReIDモデル初期化開始...")
        self.model = make_transreid_model(
            TRANS_REID_CONFIG,
            DATASET_CONFIG.Market1501.num_classes,
            DATASET_CONFIG.Market1501.camera_num,
            DATASET_CONFIG.Market1501.view_num
        )

        self.model.load_param(REID_MODEL_PATH_CONFIG.Path.trans_reid)

        self.logger.info(
            f"TransReIDモデル重み読み込み完了: {REID_MODEL_PATH_CONFIG.Path.trans_reid}")

        self.transform = transforms.Compose([
            transforms.Resize(TRANS_REID_CONFIG.INPUT.SIZE_TRAIN),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=TRANS_REID_CONFIG.INPUT.MEAN,
                std=TRANS_REID_CONFIG.INPUT.STD
            )
        ])

        self.logger.info(
            f"TransReIDモデルの前処理設定完了: {TRANS_REID_CONFIG.INPUT.SIZE_TRAIN}")
        print("TransReIDモデル初期化完了")

    def _initialize_la_transformer_model(self) -> None:
        """LA-Transformerモデルの初期化"""
        print("LA-Transformerモデル初期化開始...")
        # ベースモデルをtimmで作成
        vit_base = timm.create_model(
            LA_TRANSFORMER_CONFIG.MODEL.BACKBONE,
            pretrained=True,
            num_classes=DATASET_CONFIG.Market1501.num_classes
        )
        self.model = LATransformerTest(
            vit_base, LA_TRANSFORMER_CONFIG.MODEL.LAMBDA)

        # 重みをロード
        state_dict = torch.load(
            REID_MODEL_PATH_CONFIG.Path.la_transformer,
            map_location=APP_CONFIG.device
        )
        self.model.load_state_dict(state_dict, strict=False)

        self.logger.info(
            f"LA-Transformerモデル重み読み込み完了: {REID_MODEL_PATH_CONFIG.Path.la_transformer}")

        self.transform = transforms.Compose([
            transforms.Resize(LA_TRANSFORMER_CONFIG.INPUT.SIZE_TRAIN),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=LA_TRANSFORMER_CONFIG.INPUT.MEAN,
                std=LA_TRANSFORMER_CONFIG.INPUT.STD
            )
        ])

        self.logger.info(
            f"LA-Transformerモデルの前処理設定完了: {LA_TRANSFORMER_CONFIG.INPUT.SIZE_TRAIN}")
        print("LA-Transformerモデル初期化完了")

    def extract_features(self, image_crop: np.ndarray, camera_id: int = 0, view_id: int = 0) -> np.ndarray:
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

        if image_crop is None or image_crop.size == 0:
            raise Exception("無効な画像が提供されました")

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

            # NumPy配列に変換
            features_numpy = features.cpu().numpy()

            # バッチ次元を除去 (1, feature_dim) -> (feature_dim,)
            if features_numpy.ndim == 2 and features_numpy.shape[0] == 1:
                features_numpy = features_numpy.squeeze(0)

            self.logger.debug(
                f"特徴抽出完了: 形状={features_numpy.shape}, バックエンド={self.backend}")
            return features_numpy

        except Exception as e:
            self.logger.error(f"特徴抽出エラー: {e}")
            raise Exception(f"特徴抽出に失敗しました: {e}")
