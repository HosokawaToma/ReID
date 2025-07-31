"""後処理マネージャーモジュール"""
from dataclasses import dataclass
from torchreid import metrics
import logging
import torch
import numpy as np
from typing import List, Tuple


@dataclass
class PostProcessingConfig:
    class EVALUATE:
        MAX_RANK: int = 50
        METRIC: str = "cosine"
        USE_METRIC_CUHK03: bool = False
        USE_CYTHON: bool = False

    class ASSIGN_PERSON_ID:
        SIMILARITY_THRESHOLD: float = 0.5


POST_PROCESSING_CONFIG = PostProcessingConfig()


class PostProcessingManager:
    """後処理マネージャークラス"""

    def __init__(
        self,
        max_rank: int = POST_PROCESSING_CONFIG.EVALUATE.MAX_RANK,
        metric: str = POST_PROCESSING_CONFIG.EVALUATE.METRIC,
        use_metric_cuhk03: bool = POST_PROCESSING_CONFIG.EVALUATE.USE_METRIC_CUHK03,
        use_cython: bool = POST_PROCESSING_CONFIG.EVALUATE.USE_CYTHON,
        similarity_threshold: float = POST_PROCESSING_CONFIG.ASSIGN_PERSON_ID.SIMILARITY_THRESHOLD
    ) -> None:
        """初期化"""
        print("PostProcessingManager初期化開始...")
        self.logger = logging.getLogger(__name__)
        self.max_rank = max_rank
        self.metric = metric
        self.use_metric_cuhk03 = use_metric_cuhk03
        self.use_cython = use_cython
        self.similarity_threshold = similarity_threshold
        self.next_person_id = 1
        self.logger.info(
            f"人物ID割り当ての類似度閾値を{self.similarity_threshold}に設定しました")
        print("PostProcessingManager初期化完了")

    def evaluate(
        self,
        query_feats: List[torch.Tensor],
        gallery_feats: List[torch.Tensor],
        query_person_ids: List[int],
        gallery_person_ids: List[int],
        query_camera_ids: List[int],
        gallery_camera_ids: List[int],
    ) -> Tuple[np.ndarray, float]:
        """
        特徴量の評価を実行する

        Args:
            query_feats: クエリ特徴量のリスト
            gallery_feats: ギャラリー特徴量のリスト
            query_person_ids: クエリ人物IDのリスト
            gallery_person_ids: ギャラリープロセスIDのリスト
            query_camera_ids: クエリカメラIDのリスト
            gallery_camera_ids: ギャラリーカメラIDのリスト

        Returns:
            Tuple[np.ndarray, float]: CMCスコアとmAPスコア
        """
        # 特徴量をテンソルに変換
        q = torch.stack(query_feats, 0)  # (Nq,D)
        g = torch.stack(gallery_feats, 0)  # (Ng,D)

        # 距離行列 (Nq,Ng)
        dist = metrics.compute_distance_matrix(
            q, g, metric=self.metric).cpu().numpy()

        # IDとカメラIDをNumPy配列に変換
        q_pids = np.asarray(query_person_ids, dtype=np.int64)
        g_pids = np.asarray(gallery_person_ids, dtype=np.int64)
        q_camids = np.asarray(query_camera_ids, dtype=np.int64)
        g_camids = np.asarray(gallery_camera_ids, dtype=np.int64)

        # 評価実行
        cmc, mAP = metrics.evaluate_rank(
            dist,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            max_rank=self.max_rank,
            use_metric_cuhk03=self.use_metric_cuhk03,
            use_cython=self.use_cython
        )

        return cmc, mAP

    def assign_person_id(
        self,
        query_feat: torch.Tensor,
        gallery_feats: List[torch.Tensor],
        gallery_person_ids: List[int]
    ) -> int:
        """
        人物IDを割り当てる
        """
        if query_feat is None or not gallery_feats:
            self.next_person_id += 1
            return self.next_person_id

        q = query_feat.float()
        G = torch.stack([g.float() for g in gallery_feats], dim=0)

        sims = G @ q
        if sims.numel() == 0:
            self.next_person_id += 1
            return self.next_person_id

        rank1_idx = int(torch.argmax(sims).item())
        best_sim = float(sims[rank1_idx])

        if rank1_idx >= len(gallery_person_ids):
            raise ValueError("gallery_feats と gallery_person_ids の対応が不一致です。")

        best_id = int(gallery_person_ids[rank1_idx])
        if best_sim >= self.similarity_threshold:
            return best_id
        else:
            self.next_person_id += 1
            return self.next_person_id
