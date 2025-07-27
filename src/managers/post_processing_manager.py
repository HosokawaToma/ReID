"""後処理マネージャーモジュール"""
from config import POST_PROCESSING_CONFIG
from torchreid import metrics
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
        query_feats: List[np.ndarray],
        gallery_feats: List[np.ndarray],
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
            q = torch.stack([torch.from_numpy(feat) for feat in query_feats], 0)  # (Nq,D)
            g = torch.stack([torch.from_numpy(feat) for feat in gallery_feats], 0)  # (Ng,D)

            q = F.normalize(q, p=2, dim=1)
            g = F.normalize(g, p=2, dim=1)

            # 距離行列 (Nq,Ng)
            dist = metrics.compute_distance_matrix(q, g, metric=POST_PROCESSING_CONFIG.EVALUATE.METRIC).cpu().numpy()

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
