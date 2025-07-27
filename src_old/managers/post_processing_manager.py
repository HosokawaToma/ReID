"""K-Reciprocal Re-ranking Manager"""
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np

from post_processing.k_reciprocal_encoding import re_ranking

logger = logging.getLogger(__name__)


class KReciprocalManager:
    """K-Reciprocal Re-ranking による人物識別精度向上管理クラス"""

    def __init__(self, k1: int = 20, k2: int = 6, lambda_value: float = 0.3):
        """
        K-Reciprocal Re-ranking Manager を初期化する

        :param k1: k-reciprocal encoding のパラメータ k1 (デフォルト: 20)
        :param k2: query expansion のパラメータ k2 (デフォルト: 6)
        :param lambda_value: 最終距離計算の重み付けパラメータ (デフォルト: 0.3)
        """
        self.k1 = k1
        self.k2 = k2
        self.lambda_value = lambda_value

        # 特徴量データベース
        self.query_features: List[np.ndarray] = []
        self.gallery_features: List[np.ndarray] = []
        self.query_ids: List[int] = []
        self.gallery_ids: List[int] = []

        logger.info(
            f"K-Reciprocal Manager 初期化完了 (k1={k1}, k2={k2}, lambda={lambda_value})")

    def add_query_features(self, person_id: int, features: np.ndarray) -> None:
        """
        クエリ特徴量を追加する

        :param person_id: 人物ID
        :param features: 特徴量ベクトル
        """
        if features is None or features.size == 0:
            logger.warning(f"無効な特徴量が提供されました (ID: {person_id})")
            return

        self.query_features.append(features.copy())
        self.query_ids.append(person_id)
        logger.debug(f"クエリ特徴量追加: ID={person_id}, 形状={features.shape}")

    def add_gallery_features(self, person_id: int, features: np.ndarray) -> None:
        """
        ギャラリー特徴量を追加する

        :param person_id: 人物ID
        :param features: 特徴量ベクトル
        """
        if features is None or features.size == 0:
            logger.warning(f"無効な特徴量が提供されました (ID: {person_id})")
            return

        self.gallery_features.append(features.copy())
        self.gallery_ids.append(person_id)
        logger.debug(f"ギャラリー特徴量追加: ID={person_id}, 形状={features.shape}")

    def compute_distance_matrix(self, features1: List[np.ndarray], features2: List[np.ndarray]) -> np.ndarray:
        """
        特徴量リスト間の距離行列を計算する

        :param features1: 特徴量リスト1
        :param features2: 特徴量リスト2
        :return: 距離行列 (shape: [len(features1), len(features2)])
        """
        if not features1 or not features2:
            return np.array([])

        # 特徴量を行列に変換
        feat_matrix1 = np.vstack(features1)  # (num_query, feature_dim)
        feat_matrix2 = np.vstack(features2)  # (num_gallery, feature_dim)

        # コサイン距離を計算 (1 - コサイン類似度)
        # コサイン類似度 = dot(a, b) / (||a|| * ||b||)
        # 特徴量は既にL2正規化されているため、内積がコサイン類似度
        similarity_matrix = np.dot(feat_matrix1, feat_matrix2.T)
        distance_matrix = 1.0 - similarity_matrix

        # 距離を [0, 1] の範囲にクリップ
        distance_matrix = np.clip(distance_matrix, 0.0, 1.0)

        return distance_matrix

    def perform_re_ranking(self) -> Optional[np.ndarray]:
        """
        K-Reciprocal Re-ranking を実行する

        :return: Re-ranking後の距離行列 (shape: [num_query, num_gallery])
                 データが不十分な場合は None
        """
        if len(self.query_features) == 0 or len(self.gallery_features) == 0:
            logger.warning("Re-ranking実行に必要なデータが不足しています")
            return None

        try:
            logger.info(
                f"Re-ranking開始: クエリ={len(self.query_features)}, ギャラリー={len(self.gallery_features)}")

            # 距離行列を計算
            q_g_dist = self.compute_distance_matrix(
                self.query_features, self.gallery_features)
            q_q_dist = self.compute_distance_matrix(
                self.query_features, self.query_features)
            g_g_dist = self.compute_distance_matrix(
                self.gallery_features, self.gallery_features)

            # K-Reciprocal Re-ranking を実行
            final_dist = re_ranking(
                q_g_dist, q_q_dist, g_g_dist,
                k1=self.k1, k2=self.k2, lambda_value=self.lambda_value
            )

            logger.info(f"Re-ranking完了: 出力形状={final_dist.shape}")
            return final_dist

        except Exception as e:
            logger.error(f"Re-ranking実行エラー: {e}")
            return None

    def get_improved_matches(self, query_idx: int, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Re-ranking後の改善されたマッチング結果を取得する

        :param query_idx: クエリのインデックス
        :param top_k: 上位何件を返すか
        :return: (gallery_person_id, distance) のタプルリスト
        """
        re_ranked_dist = self.perform_re_ranking()

        if re_ranked_dist is None or query_idx >= len(self.query_ids):
            return []

        try:
            # 指定されたクエリの距離を取得
            query_distances = re_ranked_dist[query_idx]

            # 距離でソート（昇順）
            sorted_indices = np.argsort(query_distances)

            # 上位top_k件を取得
            results = []
            for i in range(min(top_k, len(sorted_indices))):
                gallery_idx = sorted_indices[i]
                if gallery_idx < len(self.gallery_ids):
                    gallery_person_id = self.gallery_ids[gallery_idx]
                    distance = query_distances[gallery_idx]
                    results.append((gallery_person_id, float(distance)))

            return results

        except Exception as e:
            logger.error(f"マッチング結果取得エラー: {e}")
            return []

    def clear_database(self) -> None:
        """特徴量データベースをクリアする"""
        self.query_features.clear()
        self.gallery_features.clear()
        self.query_ids.clear()
        self.gallery_ids.clear()
        logger.info("特徴量データベースをクリアしました")

    def get_statistics(self) -> Dict[str, int]:
        """現在の統計情報を取得する"""
        return {
            "query_count": len(self.query_features),
            "gallery_count": len(self.gallery_features),
            "total_features": len(self.query_features) + len(self.gallery_features)
        }
