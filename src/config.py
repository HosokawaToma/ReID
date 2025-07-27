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
        reid_backend: str = "la_transformer"
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
    """Configuration for CLIP-ReID model, matching YACS defaults."""
    class MODEL:
        # device settings
        DEVICE: str = "cuda"
        DEVICE_ID: str = "0"

        # backbone / pretrained
        PRETRAIN_CHOICE: str = "imagenet"
        PRETRAIN_PATH: str = ""
        NAME: str = "ViT-B-16"
        LAST_STRIDE: int = 1

        # neck and loss settings
        NECK: str = "bnneck"
        IF_WITH_CENTER: str = "no"
        IF_LABELSMOOTH: str = "on"
        ID_LOSS_TYPE: str = "softmax"
        METRIC_LOSS_TYPE: str = "triplet"
        ID_LOSS_WEIGHT: float = 0.25
        TRIPLET_LOSS_WEIGHT: float = 1.0
        I2T_LOSS_WEIGHT: float = 1.0
        COS_LAYER: bool = False
        DIST_TRAIN: bool = False
        NO_MARGIN: bool = False

        # transformer & dropout
        DROP_PATH: float = 0.1
        DROP_OUT: float = 0.0
        ATT_DROP_RATE: float = 0.0
        TRANSFORMER_TYPE: str = "None"
        STRIDE_SIZE: Tuple[int, int] = (12, 12)

        # SIE (camera/view)
        SIE_CAMERA: bool = True
        SIE_VIEW: bool = False
        SIE_COE: float = 1.0

    class INPUT:
        SIZE_TRAIN: Tuple[int, int] = (256, 128)
        SIZE_TEST: Tuple[int, int] = (256, 128)
        PROB: float = 0.5          # horizontal flip prob.
        RE_PROB: float = 0.5       # random erasing prob.
        PADDING: int = 10
        PIXEL_MEAN: Tuple[float, float, float] = (0.5, 0.5, 0.5)
        PIXEL_STD: Tuple[float, float, float] = (0.5, 0.5, 0.5)

    class DATASETS:
        NAMES: str = "market1501"
        ROOT_DIR: str = "./person_images/input/market1501"

    class DATALOADER:
        SAMPLER: str = "softmax_triplet"
        NUM_INSTANCE: int = 4
        NUM_WORKERS: int = 8

    class SOLVER:
        SEED: int = 1234
        MARGIN: float = 0.3

        class STAGE2:
            IMS_PER_BATCH: int = 64
            OPTIMIZER_NAME: str = "Adam"
            BASE_LR: float = 5e-6
            MOMENTUM: float = 0.9

            # weight decay
            WEIGHT_DECAY: float = 1e-4
            WEIGHT_DECAY_BIAS: float = 1e-4

            # FC layer LR factor
            LARGE_FC_LR: bool = False
            BIAS_LR_FACTOR: int = 2

            # learning rate schedule
            STEPS: Tuple[int, int] = (30, 50)
            GAMMA: float = 0.1

            # training epochs
            MAX_EPOCHS: int = 60

            # warmup settings
            WARMUP_METHOD: str = "linear"
            WARMUP_ITERS: int = 10
            WARMUP_FACTOR: float = 0.1
            WARMUP_EPOCHS: int = 5
            WARMUP_LR_INIT: float = 0.01
            LR_MIN: float = 1.6e-5

            # center loss (if used)
            CENTER_LR: float = 0.5
            CENTER_LOSS_WEIGHT: float = 0.0005

            # cosine margin/scale (if using arcface)
            COSINE_MARGIN: float = 0.5
            COSINE_SCALE: float = 30

            # logging / checkpointing
            CHECKPOINT_PERIOD: int = 60
            LOG_PERIOD: int = 50
            EVAL_PERIOD: int = 60

    class TEST:
        EVAL: bool = True
        IMS_PER_BATCH: int = 64
        RE_RANKING: bool = False
        WEIGHT: str = "models/Market1501_clipreid_12x12sie_ViT-B-16_60.pth"
        NECK_FEAT: str = "before"
        FEAT_NORM: str = "yes"
        DIST_MAT: str = "dist_mat.npy"

CLIP_REID_CONFIG = ClipReIDConfig()

# ================================
# TransReID設定
# ================================


@dataclass
class TransReIDConfig:
    """Configuration for the Transformer‑based ReID model."""
    class MODEL:
        # device settings
        DEVICE: str = "cuda"
        DEVICE_ID: str = "4"

        # backbone / pretrained
        NAME: str = "transformer"
        PRETRAIN_CHOICE: str = "imagenet"
        PRETRAIN_PATH: str = "./models/deit_base_distilled_patch16_224-df68dfff.pth"
        LAST_STRIDE: int = 1

        # neck and loss settings
        NECK: str = "bnneck"
        IF_WITH_CENTER: str = "no"
        IF_LABELSMOOTH: str = "off"
        ID_LOSS_TYPE: str = "softmax"
        METRIC_LOSS_TYPE: str = "triplet"
        ID_LOSS_WEIGHT: float = 1.0
        TRIPLET_LOSS_WEIGHT: float = 1.0
        I2T_LOSS_WEIGHT: float = 1.0
        COS_LAYER: bool = False
        DIST_TRAIN: bool = False
        NO_MARGIN: bool = True

        # transformer & dropout
        TRANSFORMER_TYPE: str = "vit_base_patch16_224_TransReID"
        DROP_PATH: float = 0.1
        DROP_OUT: float = 0.0
        ATT_DROP_RATE: float = 0.0
        STRIDE_SIZE: Tuple[int, int] = (12, 12)

        # JPM augmentation
        JPM: bool = True
        SHIFT_NUM: int = 5
        SHUFFLE_GROUP: int = 2
        DEVIDE_LENGTH: int = 4
        RE_ARRANGE: bool = True

        # SIE (camera/view)
        SIE_CAMERA: bool = True
        SIE_VIEW: bool = False
        SIE_COE: float = 3.0

    class INPUT:
        SIZE_TRAIN: Tuple[int, int] = (256, 128)
        SIZE_TEST: Tuple[int, int] = (256, 128)
        PROB: float = 0.5       # horizontal flip prob.
        RE_PROB: float = 0.8    # random erasing prob.
        PADDING: int = 10
        PIXEL_MEAN: Tuple[float, float, float] = (0.5, 0.5, 0.5)
        PIXEL_STD: Tuple[float, float, float] = (0.5, 0.5, 0.5)

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
        # if using larger LR for FC layer
        LARGE_FC_LR: bool = False
        # bias learning-rate factor
        BIAS_LR_FACTOR: int = 2
        # weight decay
        WEIGHT_DECAY: float = 1e-4
        WEIGHT_DECAY_BIAS: float = 1e-4
        # logging / checkpointing
        CHECKPOINT_PERIOD: int = 120
        LOG_PERIOD: int = 50
        EVAL_PERIOD: int = 120

    class TEST:
        EVAL: bool = True
        IMS_PER_BATCH: int = 256
        RE_RANKING: bool = False
        WEIGHT: str = "../logs/0321_market_deit_transreie/transformer_120.pth"
        NECK_FEAT: str = "before"
        FEAT_NORM: str = "yes"
        DIST_MAT: str = "dist_mat.npy"

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
