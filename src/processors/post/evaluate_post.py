import torch
from typing import List
import numpy as np
from torchreid import metrics
from library.post_processing.k_reciprocal_encoding import re_ranking

class EvaluatePostProcessor:
    def __init__(
        self,
        max_rank: int = 10,
        metric: str = "cosine",
        use_metric_cuhk03: bool = False,
        use_cython: bool = False,
        k_reciprocal_re_ranking: bool = False
    ):
        self.max_rank = max_rank
        self.metric = metric
        self.use_metric_cuhk03 = use_metric_cuhk03
        self.use_cython = use_cython
        self.k_reciprocal_re_ranking = k_reciprocal_re_ranking
        self.cmc = None
        self.mAP = None

    def evaluate(
        self,
        query_feats: torch.Tensor,
        gallery_feats: torch.Tensor,
        query_person_ids: List[int],
        gallery_person_ids: List[int],
        query_camera_ids: List[int],
        gallery_camera_ids: List[int],
    ) -> None:
        if self.k_reciprocal_re_ranking:
            distance_matrix = self._k_reciprocal_re_ranking(
                gallery_feats, query_feats)
        else:
            distance_matrix = metrics.compute_distance_matrix(
                query_feats, gallery_feats, metric=self.metric).cpu().numpy()

        q_pids = np.asarray(query_person_ids)
        g_pids = np.asarray(gallery_person_ids)
        q_camids = np.asarray(query_camera_ids)
        g_camids = np.asarray(gallery_camera_ids)

        cmc, mAP = metrics.evaluate_rank(
            distance_matrix,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            max_rank=self.max_rank,
            use_metric_cuhk03=self.use_metric_cuhk03,
            use_cython=self.use_cython
        )

        self.cmc = cmc
        self.mAP = mAP

    def get_cmc(self) -> np.ndarray:
        return self.cmc

    def get_A1(self) -> float:
        return self.cmc[0]

    def get_A5(self) -> float:
        return self.cmc[4]

    def get_mAP(self) -> float:
        return self.mAP

    def _k_reciprocal_re_ranking(
        self,
        gallery_feats: torch.Tensor,
        query_feats: torch.Tensor,
    ) -> np.ndarray:
        q_g_dist = metrics.compute_distance_matrix(
            query_feats, gallery_feats, metric=self.metric).cpu().numpy()
        q_q_dist = metrics.compute_distance_matrix(
            query_feats, query_feats, metric=self.metric).cpu().numpy()
        g_g_dist = metrics.compute_distance_matrix(
            gallery_feats, gallery_feats, metric=self.metric).cpu().numpy()
        return re_ranking(q_g_dist, q_q_dist, g_g_dist)
