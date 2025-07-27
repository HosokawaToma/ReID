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
        PRETRAIN_CHOICE: str = "imagenet"
        METRIC_LOSS_TYPE: str = "triplet"
        IF_LABELSMOOTH: str = "on"
        IF_WITH_CENTER: str = "no"
        NAME: str = "ViT-B-16"
        STRIDE_SIZE: Tuple[int, int] = (16, 16)
        ID_LOSS_WEIGHT: float = 0.25
        TRIPLET_LOSS_WEIGHT: float = 1.0
        I2T_LOSS_WEIGHT: float = 1.0

    class INPUT:
        SIZE_TRAIN: Tuple[int, int] = (256, 128)
        SIZE_TEST: Tuple[int, int] = (256, 128)
        PROB: float = 0.5  # random horizontal flip
        RE_PROB: float = 0.5  # random erasing
        PADDING: int = 10
        PIXEL_MEAN: Tuple[float, float, float] = (0.5, 0.5, 0.5)
        PIXEL_STD: Tuple[float, float, float] = (0.5, 0.5, 0.5)

    class DATALOADER:
        SAMPLER: str = "softmax_triplet"
        NUM_INSTANCE: int = 4
        NUM_WORKERS: int = 8

    class SOLVER:
        class STAGE1:
            IMS_PER_BATCH: int = 64
            OPTIMIZER_NAME: str = "Adam"
            BASE_LR: float = 0.00035
            WARMUP_LR_INIT: float = 0.00001
            LR_MIN: float = 1e-6
            WARMUP_METHOD: str = 'linear'
            WEIGHT_DECAY: float = 1e-4
            WEIGHT_DECAY_BIAS: float = 1e-4
            MAX_EPOCHS: int = 120
            CHECKPOINT_PERIOD: int = 120
            LOG_PERIOD: int = 50
            WARMUP_EPOCHS: int = 5

        class STAGE2:
            IMS_PER_BATCH: int = 64
            OPTIMIZER_NAME: str = "Adam"
            BASE_LR: float = 5e-06
            WARMUP_METHOD: str = 'linear'
            WARMUP_ITERS: int = 10
            WARMUP_FACTOR: float = 0.1
            WEIGHT_DECAY: float = 1e-4
            WEIGHT_DECAY_BIAS: float = 1e-4
            LARGE_FC_LR: bool = False
            MAX_EPOCHS: int = 60
            CHECKPOINT_PERIOD: int = 60
            LOG_PERIOD: int = 50
            EVAL_PERIOD: int = 60
            BIAS_LR_FACTOR: int = 2
            STEPS: Tuple[int, int] = (30, 50)
            GAMMA: float = 0.1

    class TEST:
        EVAL: bool = True
        IMS_PER_BATCH: int = 64
        RE_RANKING: bool = False
        WEIGHT: str = ''
        NECK_FEAT: str = 'before'
        FEAT_NORM: str = 'yes'

CLIP_REID_CONFIG = ClipReIDConfig()

# ================================
# TransReID設定
# ================================


@dataclass
class TransReIDConfig:
    class MODEL:
        PRETRAIN_CHOICE: str = "imagenet"
        PRETRAIN_PATH: str = "models/vit_transreid_market.pth"
        METRIC_LOSS_TYPE: str = "triplet"
        IF_LABELSMOOTH: str = "off"
        IF_WITH_CENTER: str = "no"
        NAME: str = "transformer"
        NO_MARGIN: bool = True
        DEVICE_ID: Tuple[str, ...] = ("7",)
        TRANSFORMER_TYPE: str = "vit_base_patch16_224_TransReID"
        STRIDE_SIZE: Tuple[int, int] = (16, 16)

    class INPUT:
        SIZE_TRAIN: Tuple[int, int] = (256, 128)
        SIZE_TEST: Tuple[int, int] = (256, 128)
        PROB: float = 0.5
        RE_PROB: float = 0.5
        PADDING: int = 10
        PIXEL_MEAN: Tuple[float, float, float] = (0.5, 0.5, 0.5)
        PIXEL_STD: Tuple[float, float, float] = (0.5, 0.5, 0.5)

    class DATASETS:
        NAMES: str = "market1501"
        ROOT_DIR: str = "../../data"

    class DATALOADER:
        SAMPLER: str = "softmax_triplet"
        NUM_INSTANCE: int = 4
        NUM_WORKERS: int = 8

    class SOLVER:
        OPTIMIZER_NAME: str = "SGD"
        MAX_EPOCHS: int = 120
        BASE_LR: float = 0.008
        IMS_PER_BATCH: int = 64
        WARMUP_METHOD: str = "linear"
        LARGE_FC_LR: bool = False
        CHECKPOINT_PERIOD: int = 120
        LOG_PERIOD: int = 50
        EVAL_PERIOD: int = 120
        WEIGHT_DECAY: float = 1e-4
        WEIGHT_DECAY_BIAS: float = 1e-4
        BIAS_LR_FACTOR: int = 2

    class TEST:
        EVAL: bool = True
        IMS_PER_BATCH: int = 256
        RE_RANKING: bool = False
        WEIGHT: str = "../logs/0321_market_vit_base/transformer_120.pth"
        NECK_FEAT: str = "before"
        FEAT_NORM: str = "yes"

    OUTPUT_DIR: str = "../logs/0321_market_vit_base"

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


POST_PROCESSING_CONFIG = PostProcessingConfig()
