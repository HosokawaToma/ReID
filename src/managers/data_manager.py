"""データマネージャーモジュール"""
import logging
from typing import List
import torch


class DataManager:
    """データマネージャークラス"""

    def __init__(self):
        """初期化"""
        self.logger = logging.getLogger(__name__)

        self.query_person_ids: List[int] = []
        self.query_camera_ids: List[int] = []
        self.query_view_ids: List[int] = []
        self.query_feats: torch.Tensor = None
        self.gallery_person_ids: List[int] = []
        self.gallery_camera_ids: List[int] = []
        self.gallery_view_ids: List[int] = []
        self.gallery_feats: torch.Tensor = None

    def add_gallery(self, person_id: int, camera_id: int, view_id: int, feats: torch.Tensor) -> None:
        """
        ギャラリー特徴量を追加

        Args:
            person_id: 人物ID
            camera_id: カメラID
            view_id: ビューID
            feats: 特徴量
        """
        self.gallery_person_ids.append(person_id)
        self.gallery_camera_ids.append(camera_id)
        self.gallery_view_ids.append(view_id)
        if self.gallery_feats is None:
            self.gallery_feats = feats
        else:
            self.gallery_feats = torch.cat([self.gallery_feats, feats], dim=0)
        self.logger.info(
            f"ギャラリー特徴量を追加しました: person_id={person_id}, camera_id={camera_id}, view_id={view_id}")

    def add_query(self, person_id: int, camera_id: int, view_id: int, feats: torch.Tensor) -> None:
        """
        クエリ特徴量を追加

        Args:
            person_id: 人物ID
            camera_id: カメラID
            view_id: ビューID
            feats: 特徴量
        """
        self.query_person_ids.append(person_id)
        self.query_camera_ids.append(camera_id)
        self.query_view_ids.append(view_id)
        if self.query_feats is None:
            self.query_feats = feats
        else:
            self.query_feats = torch.cat([self.query_feats, feats], dim=0)
        self.logger.info(
            f"クエリ特徴量を追加しました: person_id={person_id}, camera_id={camera_id}, view_id={view_id}")
