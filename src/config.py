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

# ================================
# YOLO設定
# ================================


@dataclass
class YoloConfig:
    class MODEL:
        model_path: str = "yolo11n.pt"
        confidence_threshold: float = 0.7
        person_class_id: int = 1


YOLO_CONFIG = YoloConfig()

# ================================
# ReIDモデルのパス設定
# ================================


@dataclass
class ReIDModelPathConfig:
    class Path:
        clip_reid: str = "models/Market1501_clipreid_12x12sie_ViT-B-16_60.pth"
        trans_reid: str = "models/vit_transreid_market.pth"
        la_transformer: str = "models/net_best.pth"


REID_MODEL_PATH_CONFIG = ReIDModelPathConfig()

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
        MEAN: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
        STD: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

    class TEST:
        NECK_FEAT: str = "after"

    class DATASETS:
        NAMES: str = "market"


CLIP_REID_CONFIG = ClipReIDConfig()

# ================================
# TransReID設定
# ================================


@dataclass
class TransReIDConfig:
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
        MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
        STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    class TEST:
        NECK_FEAT: str = "before"

    class SOLVER:
        COSINE_SCALE: float = 30.0
        COSINE_MARGIN: float = 0.5

    class DATASETS:
        NAMES: str = "market"


TRANS_REID_CONFIG = TransReIDConfig()

# ================================
# LA-Transformer設定
# ================================


@dataclass
class LA_TransformerConfig:
    class MODEL:
        NAME: str = "la_with_lmbd_8"
        BACKBONE: str = "vit_base_patch16_224"
        LAMBDA: float = 8

    class INPUT:
        SIZE_TRAIN: Tuple[int, int] = (224, 224)
        MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
        STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


LA_TRANSFORMER_CONFIG = LA_TransformerConfig()

# ================================
# 後処理設定
# ================================


@dataclass
class PostProcessingConfig:
    class EVALUATE:
        MAX_RANK: int = 50
        USE_METRIC_CUHK03: bool = False
        USE_CYTHON: bool = False


POST_PROCESSING_CONFIG = PostProcessingConfig()
