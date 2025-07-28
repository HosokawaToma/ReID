from typing import Tuple, List
from typing import Tuple
from dataclasses import dataclass
import torch
from typing import Tuple

# ================================
# アプリケーション設定
# ================================


@dataclass
class AppConfig:
    device: torch.device = None

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")


APP_CONFIG = AppConfig()

# ================================
# person_image_app設定
# ================================


@dataclass
class PersonImageAppConfig:
    class Default:
        reid_backend: str = "trans_reid"
        data_set_name: str = "market10"

    class Directories:
        input_dir_str: str = "./resources/person_images/input"
        output_dir_str: str = "./resources/person_images/output"

    class PostProcessing:
        re_ranking: bool = True


# ================================
# video_app設定
# ================================


@dataclass
class VideoAppConfig:
    class Default:
        reid_backend: str = "trans_reid"
        data_set_name: str = "market10"

    class Directories:
        input_dir_str: str = "./resources/videos"
        output_dir_str: str = "./resources/videos"

    class PostProcessing:
        re_ranking: bool = True


VIDEO_APP_CONFIG = VideoAppConfig()

# ================================
# YOLO設定
# ================================


@dataclass
class YoloConfig:
    class MODEL:
        model_path: str = "models/yolo11n-pose.pt"
        confidence_threshold: float = 0.7
        person_class_id: int = 1


YOLO_CONFIG = YoloConfig()

# ================================
# データセット設定
# ================================


@dataclass
class DatasetConfig:
    class Market1501:
        name: str = "market"
        num_classes: int = 751
        camera_num: int = 6
        view_num: int = 1


DATASET_CONFIG = DatasetConfig()

# ================================
# ClipReID設定
# ================================


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


CLIP_REID_CONFIG = ClipReIDConfig()

# ================================
# TransReID設定
# ================================


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


TRANS_REID_CONFIG = TransReIDConfig()

# ================================
# LA-Transformer設定
# ================================


@dataclass
class LA_TransformerConfig:
    class MODEL:
        PATH: str = "models/net_best.pth"
        NAME: str = "la_with_lmbd_8"
        BACKBONE: str = "vit_base_patch16_224"
        LAMBDA: float = 8

    class INPUT:
        SIZE_TEST: Tuple[int, int] = (224, 224)
        PIXEL_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
        PIXEL_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


LA_TRANSFORMER_CONFIG = LA_TransformerConfig()

# ================================
# 後処理設定
# ================================


@dataclass
class PostProcessingConfig:
    class EVALUATE:
        MAX_RANK: int = 50
        METRIC: str = "cosine"
        USE_METRIC_CUHK03: bool = False
        USE_CYTHON: bool = False
        USE_RE_RANKING: bool = True  # re-rankingの有効/無効

        class RE_RANKING:
            K1: int = 20
            K2: int = 6
            LAMBDA_VALUE: float = 0.3


POST_PROCESSING_CONFIG = PostProcessingConfig()
