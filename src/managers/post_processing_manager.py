"""後処理マネージャーモジュール"""
from sklearn.metrics import roc_curve, precision_recall_curve
from dataclasses import dataclass
from torchreid import metrics
from post_processing.k_reciprocal_encoding import re_ranking
import logging
import torch
import numpy as np
from typing import List, Tuple


@dataclass
class PostProcessingConfig:
    class EVALUATE:
        K_RECIPROCAL_RE_RANKING: bool = False
        MAX_RANK: int = 50
        METRIC: str = "cosine"
        USE_METRIC_CUHK03: bool = False
        USE_CYTHON: bool = False

    class ASSIGN_PERSON_ID:
        SIMILARITY_THRESHOLD: float = 0.8


POST_PROCESSING_CONFIG = PostProcessingConfig()


class PostProcessingManager:
    """後処理マネージャークラス"""

    def __init__(
        self,
        k_reciprocal_re_ranking: bool = POST_PROCESSING_CONFIG.EVALUATE.K_RECIPROCAL_RE_RANKING,
        max_rank: int = POST_PROCESSING_CONFIG.EVALUATE.MAX_RANK,
        metric: str = POST_PROCESSING_CONFIG.EVALUATE.METRIC,
        use_metric_cuhk03: bool = POST_PROCESSING_CONFIG.EVALUATE.USE_METRIC_CUHK03,
        use_cython: bool = POST_PROCESSING_CONFIG.EVALUATE.USE_CYTHON,
        similarity_threshold: float = POST_PROCESSING_CONFIG.ASSIGN_PERSON_ID.SIMILARITY_THRESHOLD
    ) -> None:
        """初期化"""
        print("PostProcessingManager初期化開始...")
        self.logger = logging.getLogger(__name__)
        self.k_reciprocal_re_ranking = k_reciprocal_re_ranking
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
        query_feats: torch.Tensor,
        gallery_feats: torch.Tensor,
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
        q = query_feats
        g = gallery_feats

        dist = metrics.compute_distance_matrix(
            q, g, metric=self.metric).cpu().numpy()

        q_pids = np.asarray(query_person_ids, dtype=np.int64)
        g_pids = np.asarray(gallery_person_ids, dtype=np.int64)
        q_camids = np.asarray(query_camera_ids, dtype=np.int64)
        g_camids = np.asarray(gallery_camera_ids, dtype=np.int64)

        if self.k_reciprocal_re_ranking:
            q_q_dist = metrics.compute_distance_matrix(
                q, q, metric=self.metric).cpu().numpy()
            g_g_dist = metrics.compute_distance_matrix(
                g, g, metric=self.metric).cpu().numpy()
            dist = re_ranking(dist, q_q_dist, g_g_dist)

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
        gallery_feats: torch.Tensor,
        gallery_person_ids: List[int]
    ) -> int:
        """
        人物IDを割り当てる（GPUベクトル化処理）
        """
        if query_feat is None or not gallery_feats:
            self.next_person_id += 1
            return self.next_person_id

        similarities = torch.nn.functional.cosine_similarity(
            query_feat, gallery_feats, dim=1, eps=1e-8)

        valid_indices = similarities > self.similarity_threshold

        if valid_indices.any():
            best_idx = torch.argmax(similarities).item()
            best_sim = similarities[best_idx].item()

            if best_sim > self.similarity_threshold:
                return gallery_person_ids[best_idx]

        self.next_person_id += 1
        return self.next_person_id

    def compute_roc_eer_f1(
        self,
        query_feats: torch.Tensor,
        gallery_feats: torch.Tensor,
        query_person_ids: List[int],
        gallery_person_ids: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
        """
        Args:
            query_feats: List of (D,) L2-normalized torch.Tensor for each query.
            gallery_feats: List of (D,) L2-normalized torch.Tensor for each gallery.
            query_person_ids: List of int IDs for each query vector.
            gallery_person_ids: List of int IDs for each gallery vector.
        Returns:
            fpr: array of false positive rates for various thresholds.
            tpr: array of true positive rates for various thresholds.
            roc_thresholds: array of thresholds used to compute fpr/tpr.
            eer: Equal Error Rate (scalar).
            eer_threshold: threshold at which EER occurs.
            best_f1: maximum F1 score over all PR thresholds.
            best_f1_threshold: threshold achieving best_f1.
        """
        # 1) stack into matrices
        q = query_feats         # shape (Nq, D)
        g = gallery_feats       # shape (Ng, D)

        # 2) compute cosine similarity matrix (since both are L2-normalized)
        #    sims[i,j] = cos(query_i, gallery_j)
        sims = q @ g.t()                     # shape (Nq, Ng)

        # 3) flatten scores and build label vector
        #    label = 1 if same person id, else 0
        q_ids = np.array(query_person_ids)      # (Nq,)
        g_ids = np.array(gallery_person_ids)    # (Ng,)
        # broadcast compare
        labels = (q_ids[:, None] == g_ids[None, :]).astype(int)  # (Nq, Ng)

        scores = sims.cpu().numpy().ravel()     # (Nq*Ng,)
        y_true = labels.ravel()                # (Nq*Ng,)

        # 4) ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, scores)

        # 5) compute EER: find point where FPR + TPR = 1
        abs_diffs = np.abs(fpr + tpr - 1)
        idx_eer = np.nanargmin(abs_diffs)
        eer = (fpr[idx_eer] + (1 - tpr[idx_eer])) / 2
        eer_threshold = roc_thresholds[idx_eer]

        # 6) precision–recall & F1
        precision, recall, pr_thresholds = precision_recall_curve(y_true, scores)
        # drop last point where threshold is undefined
        # compute F1 = 2 * P * R / (P + R)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
        # ignore the last element (recall goes to 1.0 with no threshold)
        f1_scores = f1_scores[:-1]
        pr_thresholds = pr_thresholds
        idx_best = np.nanargmax(f1_scores)
        best_f1 = f1_scores[idx_best]
        best_f1_threshold = pr_thresholds[idx_best]

        return fpr, tpr, roc_thresholds, eer, eer_threshold, best_f1, best_f1_threshold
