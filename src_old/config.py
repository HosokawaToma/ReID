"""アプリケーション設定管理モジュール"""
from dataclasses import dataclass, field
from typing import Tuple, Dict
import torch
import logging


def _get_default_device() -> torch.device:
    """デフォルトデバイスを取得"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.getLogger(__name__).info(f"使用デバイス: {device}")
    return device


@dataclass(frozen=True)
class AppConfig:
    """アプリケーション全体の設定"""
    # デバイス設定
    device: torch.device = field(default_factory=_get_default_device)

    # UI設定
    quit_key: str = "q"
    window_name: str = "Multi-camera ReID"

    # カメラ設定
    max_cameras_to_check: int = 10
    frame_wait_time: float = 0.01
    camera_timeout: float = 5.0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30

    # 表示設定
    cv_wait_key_delay: int = 1
    display_fps: int = 30

    # ログ設定
    log_level: str = "INFO"
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


@dataclass(frozen=True)
class YOLOConfig:
    """YOLO関連の設定"""
    model_path: str = "models/yolo11n-pose.pt"
    person_class_id: int = 0
    confidence_threshold: float = 0.5  # より多くの人物を検出するため閾値を下げる


@dataclass(frozen=True)
class ReIDConfig:
    """ReID関連の設定"""
    # モデルパス
    clip_model_path: str = "models/Market1501_clipreid_12x12sie_ViT-B-16_60.pth"
    trans_reid_model_path: str = "models/vit_transreid_market.pth"
    la_transformer_model_path: str = "models/net_best.pth"

    # モデル別設定
    model_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'clip': 0.7,
        'trans_reid': 0.6,
        'trans_reid_jpm': 0.6,
        'la_transformer': 0.6,
    })
    model_ema_alpha: Dict[str, float] = field(default_factory=lambda: {
        'clip': 0.7,
        'trans_reid': 0.8,
        'trans_reid_jpm': 0.8,
        'la_transformer': 0.9,
    })

    # データセット設定
    num_classes: int = 751
    camera_num: int = 6
    view_num: int = 1

    # 識別設定
    threshold: float = 0.1

    # ReRanking設定
    use_re_ranking: bool = True
    rerank_k1: int = 20
    rerank_k2: int = 6
    rerank_lambda: float = 0.3

    # 評価設定
    evaluation_max_rank: int = 50
    evaluation_metric: str = "cosine"  # "cosine" or "euclidean"
    evaluation_normalize: bool = True
    evaluation_rerank: bool = True
    evaluation_k1: int = 20
    evaluation_k2: int = 6
    evaluation_lambda: float = 0.3


@dataclass(frozen=True)
class CLIPReIDConfig:
    """CLIP-ReID モデル設定"""
    class MODEL:
        NAME: str = "ViT-B-16"
        COS_LAYER: str = "no"
        NECK: str = "no"
        SIE_COE: float = 3.0
        SIE_CAMERA: bool = True
        SIE_VIEW: bool = True
        STRIDE_SIZE: Tuple[int, int] = (12, 12)

    class INPUT:
        SIZE_TRAIN: Tuple[int, int] = (256, 128)

    class TEST:
        NECK_FEAT: str = "after"

    class DATASETS:
        NAMES: str = "market"


@dataclass(frozen=True)
class TransReIDConfig:
    """TransReID モデル設定"""
    class MODEL:
        NAME: str = "transformer"
        TRANSFORMER_TYPE: str = "vit_base_patch16_224_TransReID"
        PRETRAIN_CHOICE: str = "self"
        PRETRAIN_PATH: str = "models/vit_transreid_market.pth"
        LAST_STRIDE: int = 1
        COS_LAYER: str = "no"
        NECK: str = "bnneck"
        JPM: bool = True
        SIE_CAMERA: bool = True
        SIE_VIEW: bool = False
        SIE_COE: float = 3.0
        STRIDE_SIZE: Tuple[int, int] = (16, 16)
        DROP_PATH: float = 0.1
        DROP_OUT: float = 0.0
        ATT_DROP_RATE: float = 0.0
        ID_LOSS_TYPE: str = "softmax"
        RE_ARRANGE: bool = False
        SHUFFLE_GROUP: int = 2
        SHIFT_NUM: int = 5
        DIVIDE_LENGTH: int = 4
        DEVIDE_LENGTH: int = 4

    class INPUT:
        SIZE_TRAIN: Tuple[int, int] = (336, 160)

    class TEST:
        NECK_FEAT: str = "before"

    class SOLVER:
        COSINE_SCALE: float = 30.0
        COSINE_MARGIN: float = 0.5

    class DATASETS:
        NAMES: str = "market"


@dataclass(frozen=True)
class LATransformerConfig:
    """LA-Transformer モデル設定"""
    class MODEL:
        NAME: str = "la_with_lmbd_8"
        BACKBONE: str = "vit_base_patch16_224"
        LAMBDA: float = 8

    class INPUT:
        SIZE_TRAIN: Tuple[int, int] = (224, 224)


@dataclass(frozen=True)
class FileProcessingConfig:
    """ファイル処理設定"""
    # サポートする拡張子
    video_extensions: Tuple[str, ...] = ('.mp4', '.avi', '.mov')
    image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')

    # 出力設定
    output_video_codec: str = 'H264'  # より互換性の高いコーデック
    output_image_quality: int = 95

    # 処理設定
    batch_size: int = 1
    max_concurrent_files: int = 4

    # 進捗表示設定
    progress_update_interval: int = 10  # フレーム数


@dataclass(frozen=True)
class ColorPalette:
    """色パレット設定"""
    colors: Tuple[Tuple[int, int, int], ...] = (
        (255, 56, 56),    # 赤
        (36, 114, 222),   # 青
        (72, 204, 55),    # 緑
        (255, 112, 31),   # オレンジ
        (36, 204, 222),   # シアン
        (180, 58, 222),   # マゼンタ
        (255, 178, 29),   # 黄
        (114, 58, 222),   # 紫
    )
    default_color: Tuple[int, int, int] = (255, 255, 255)  # White


# グローバル設定インスタンス
APP_CONFIG = AppConfig()
YOLO_CONFIG = YOLOConfig()
REID_CONFIG = ReIDConfig()
CLIP_REID_CONFIG = CLIPReIDConfig()
TRANS_REID_CONFIG = TransReIDConfig()
LA_TRANSFORMER_CONFIG = LATransformerConfig()
FILE_PROCESSING_CONFIG = FileProcessingConfig()
COLOR_PALETTE = ColorPalette()
