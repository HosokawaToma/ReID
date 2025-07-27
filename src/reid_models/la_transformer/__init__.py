"""LA Transformer model factory."""

import timm
import torch
from typing import Any
from .model import LATransformer


def make_model(cfg: Any, num_class: int, camera_num: int, view_num: int, device: torch.device) -> LATransformer:
    """
    LA Transformerモデルを作成する関数

    Args:
        cfg: 設定オブジェクト
        num_class: クラス数
        camera_num: カメラ数
        view_num: ビュー数
        device: デバイス

    Returns:
        LATransformer: 作成されたモデル
    """
    # timmからViTベースモデルを取得
    backbone = timm.create_model(
        cfg.MODEL.BACKBONE,
        pretrained=True,
        num_classes=0,  # 分類層を削除
        drop_rate=0.0,
        drop_path_rate=0.0,
        drop_block_rate=None
    )

    # LA Transformerモデルを作成
    model = LATransformer(backbone, cfg.MODEL.LAMBDA)

    return model
