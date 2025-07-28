"""後処理マネージャーモジュール"""
from config import POST_PROCESSING_CONFIG
from torchreid import metrics
from torchreid.reid.utils import re_ranking
import logging
import warnings
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

# 警告を完全に抑制
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class PostProcessingManager:
    """後処理マネージャークラス"""

    def __init__(self):
        """初期化"""
        print("PostProcessingManager初期化開始...")
        self.logger = logging.getLogger(__name__)
        print("PostProcessingManager初期化完了")

    def evaluate(
        self,
        query_feats: List[torch.Tensor],
        gallery_feats: List[torch.Tensor],
        query_process_ids: List[int],
        gallery_process_ids: List[int],
        query_camera_ids: List[int],
        gallery_camera_ids: List[int]
    ) -> Tuple[np.ndarray, float]:
        """
        特徴量の評価を実行する

        Args:
            query_feats: クエリ特徴量のリスト
            gallery_feats: ギャラリー特徴量のリスト
            query_process_ids: クエリプロセスIDのリスト
            gallery_process_ids: ギャラリープロセスIDのリスト
            query_camera_ids: クエリカメラIDのリスト
            gallery_camera_ids: ギャラリーカメラIDのリスト

        Returns:
            Tuple[np.ndarray, float]: CMCスコアとmAPスコア
        """
        print("評価処理開始...")
        try:
            # 特徴量をテンソルに変換
            q = torch.stack(query_feats, 0)  # (Nq,D)
            g = torch.stack(gallery_feats, 0)  # (Ng,D)

            # L2正規化
            q = F.normalize(q, p=2, dim=1)
            g = F.normalize(g, p=2, dim=1)

            # 距離行列 (Nq,Ng)
            dist = metrics.compute_distance_matrix(
                q, g, metric=POST_PROCESSING_CONFIG.EVALUATE.METRIC).cpu().numpy()

            # IDとカメラIDをNumPy配列に変換
            q_pids = np.asarray(query_process_ids, dtype=np.int64)
            g_pids = np.asarray(gallery_process_ids, dtype=np.int64)
            q_camids = np.asarray(query_camera_ids, dtype=np.int64)
            g_camids = np.asarray(gallery_camera_ids, dtype=np.int64)

            # 評価実行
            cmc, mAP = metrics.evaluate_rank(
                dist,
                q_pids,
                g_pids,
                q_camids,
                g_camids,
                max_rank=POST_PROCESSING_CONFIG.EVALUATE.MAX_RANK,
                use_metric_cuhk03=POST_PROCESSING_CONFIG.EVALUATE.USE_METRIC_CUHK03,
                use_cython=POST_PROCESSING_CONFIG.EVALUATE.USE_CYTHON
            )

            print("評価処理完了")
            return cmc, mAP

        except Exception as e:
            print(f"評価処理エラー: {e}")
            self.logger.error(f"評価処理でエラーが発生しました: {e}")
            raise Exception(f"評価処理に失敗しました: {e}")

    def assign_person_id(self, query_feat: torch.Tensor, gallery_feats: List[torch.Tensor], gallery_process_ids: List[int]) -> int:
        """
        人物IDを割り当てる
        """
        if query_feat is None or not gallery_feats:
            return -1

        eps = 1e-12

        # 正規化（安全のため毎回かける）
        q = F.normalize(query_feat.float().view(-1),
                        dim=0, eps=eps)             # [D]
        G = torch.stack([g.float().view(-1)
                        for g in gallery_feats], dim=0)      # [N, D]
        G = F.normalize(G, dim=1, eps=eps)

        # コサイン類似度（= 内積）
        # [N]
        sims = G @ q
        if sims.numel() == 0:
            return -1

        rank1_idx = int(torch.argmax(sims).item())
        best_sim = float(sims[rank1_idx])

        if rank1_idx >= len(gallery_process_ids):
            raise ValueError("gallery_feats と gallery_process_ids の対応が不一致です。")

        best_id = int(gallery_process_ids[rank1_idx])
        return best_id if best_sim >= POST_PROCESSING_CONFIG.ASSIGN_PERSON_ID.SIMILARITY_THRESHOLD else -1

    def k_reciprocal_re_ranking(self, query_feats: List[torch.Tensor], gallery_feats: List[torch.Tensor]) -> np.ndarray:
        """
        K-reciprocal re-rankingを実行する

        Args:
            query_feats: クエリ特徴量のリスト（PyTorchテンソル）
            gallery_feats: ギャラリー特徴量のリスト（PyTorchテンソル）

        Returns:
            np.ndarray: re-ranking後の距離行列
        """
        try:
            # PyTorchテンソルをスタック
            q = torch.stack(query_feats, 0)  # (Nq,D)
            g = torch.stack(gallery_feats, 0)  # (Ng,D)

            # L2正規化
            q_normalized = F.normalize(q, p=2, dim=1)
            g_normalized = F.normalize(g, p=2, dim=1)

            # 距離行列を計算
            q_g_dist = metrics.compute_distance_matrix(
                q_normalized, g_normalized,
                metric=POST_PROCESSING_CONFIG.EVALUATE.METRIC).cpu().numpy()
            q_q_dist = metrics.compute_distance_matrix(
                q_normalized, q_normalized,
                metric=POST_PROCESSING_CONFIG.EVALUATE.METRIC).cpu().numpy()
            g_g_dist = metrics.compute_distance_matrix(
                g_normalized, g_normalized,
                metric=POST_PROCESSING_CONFIG.EVALUATE.METRIC).cpu().numpy()

            # re-rankingを実行
            re_ranked_dist = re_ranking(
                q_g_dist,
                q_q_dist,
                g_g_dist,
                k1=POST_PROCESSING_CONFIG.EVALUATE.RE_RANKING.K1,
                k2=POST_PROCESSING_CONFIG.EVALUATE.RE_RANKING.K2,
                lambda_value=POST_PROCESSING_CONFIG.EVALUATE.RE_RANKING.LAMBDA_VALUE
            )

            print(f"re-ranking完了: 出力形状={re_ranked_dist.shape}")
            return re_ranked_dist

        except Exception as e:
            print(f"re-ranking処理エラー: {e}")
            self.logger.error(f"re-ranking処理でエラーが発生しました: {e}")
            # エラーが発生した場合は元の距離行列を返す
            q = torch.stack(query_feats, 0)
            g = torch.stack(gallery_feats, 0)
            q_normalized = F.normalize(q, p=2, dim=1)
            g_normalized = F.normalize(g, p=2, dim=1)
            return metrics.compute_distance_matrix(
                q_normalized, g_normalized,
                metric=POST_PROCESSING_CONFIG.EVALUATE.METRIC).cpu().numpy()
