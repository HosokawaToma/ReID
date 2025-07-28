"""データマネージャーモジュール"""
import logging
from typing import List
import numpy as np
import torch


class DataManager:
    """データマネージャークラス"""

    def __init__(self):
        """初期化"""
        print("DataManager初期化開始...")
        self.logger = logging.getLogger(__name__)

        self.query_process_ids: List[int] = []
        self.query_camera_ids: List[int] = []
        self.query_feats: List[torch.Tensor] = []
        self.gallery_process_ids: List[int] = []
        self.gallery_camera_ids: List[int] = []
        self.gallery_feats: List[torch.Tensor] = []

        print("DataManager初期化完了")

    def add_gallery(self, process_id: int, camera_id: int, feats: torch.Tensor) -> None:
        """
        ギャラリー特徴量を追加

        Args:
            process_id: プロセスID
            camera_id: カメラID
            feats: 特徴量
        """
        self.gallery_process_ids.append(process_id)
        self.gallery_camera_ids.append(camera_id)
        self.gallery_feats.append(feats)

    def add_query(self, process_id: int, camera_id: int, feats: torch.Tensor) -> None:
        """
        クエリ特徴量を追加

        Args:
            process_id: プロセスID
            camera_id: カメラID
            feats: 特徴量
        """
        self.query_process_ids.append(process_id)
        self.query_camera_ids.append(camera_id)
        self.query_feats.append(feats)
