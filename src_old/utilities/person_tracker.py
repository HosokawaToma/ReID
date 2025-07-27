import numpy as np
from typing import Dict, Optional

from config import ReIDConfig


class PersonTracker:
    """人物追跡とID管理を行うクラス"""

    def __init__(self, backend: str, reid_config: ReIDConfig):
        """
        :param backend: 使用するRe-IDモデルのバックエンド名
        :param reid_config: Re-IDの設定オブジェクト
        """
        self.backend = backend
        self.similarity_threshold = reid_config.model_thresholds[backend]
        # a·old + (1-a)·new
        self.ema_alpha = reid_config.model_ema_alpha[backend]
        self.person_database: Dict[int, np.ndarray] = {}
        self.next_person_id = 1
        self.use_re_ranking = reid_config.use_re_ranking

    def assign_person_id(self, features: np.ndarray, k_reciprocal_manager=None) -> int:
        """
        人物IDを割り当てる

        :param features: 検出された人物の特徴量ベクトル（正規化済み）
        :param k_reciprocal_manager: K-Reciprocal Re-ranking管理オブジェクト（オプション）
        :return: 割り当てられた人物ID
        """
        if not self.person_database:
            self.person_database[self.next_person_id] = features
            self.next_person_id += 1
            return self.next_person_id - 1

        # K-Reciprocal Re-rankingを使用する場合
        if self.use_re_ranking and k_reciprocal_manager is not None:
            best_id, best_sim = self._assign_with_reranking(features, k_reciprocal_manager)
        else:
            # 従来のコサイン類似度による割り当て
            sims = {
                pid: float(np.dot(features, vec))
                for pid, vec in self.person_database.items()
            }
            best_id, best_sim = max(sims.items(), key=lambda x: x[1])

        print(f"best_sim={best_sim:.3f}  assigned={best_id}")

        if best_sim >= self.similarity_threshold:
            # EMA 更新
            self.person_database[best_id] = (
                self.ema_alpha * self.person_database[best_id]
                + (1 - self.ema_alpha) * features
            )
            self.person_database[best_id] /= (
                np.linalg.norm(self.person_database[best_id]) + 1e-12
            )
            return best_id

        self.person_database[self.next_person_id] = features
        self.next_person_id += 1
        return self.next_person_id - 1

    def _assign_with_reranking(self, features: np.ndarray, k_reciprocal_manager) -> tuple:
        """
        K-Reciprocal Re-rankingを使用して人物IDを割り当てる

        :param features: 検出された人物の特徴量ベクトル
        :param k_reciprocal_manager: K-Reciprocal Re-ranking管理オブジェクト
        :return: (best_id, best_similarity)
        """
        try:
            # 現在の特徴量をクエリとして追加
            temp_query_id = len(k_reciprocal_manager.query_features)
            k_reciprocal_manager.add_query_features(temp_query_id, features)

            # 既存の人物データベースをギャラリーとして追加
            # 前回のギャラリーデータをクリア
            k_reciprocal_manager.gallery_features.clear()
            k_reciprocal_manager.gallery_ids.clear()
            for pid, existing_features in self.person_database.items():
                k_reciprocal_manager.add_gallery_features(pid, existing_features)

            # Re-rankingを実行
            improved_matches = k_reciprocal_manager.get_improved_matches(temp_query_id, top_k=1)

            if improved_matches:
                best_id, best_distance = improved_matches[0]
                # 距離を類似度に変換（距離が小さいほど類似度が高い）
                best_sim = 1.0 - best_distance
                return best_id, best_sim
            else:
                # Re-rankingが失敗した場合は従来の方法にフォールバック
                sims = {
                    pid: float(np.dot(features, vec))
                    for pid, vec in self.person_database.items()
                }
                return max(sims.items(), key=lambda x: x[1])

        except Exception as e:
            print(f"Re-ranking処理でエラーが発生しました: {e}")
            # エラーが発生した場合は従来の方法にフォールバック
            sims = {
                pid: float(np.dot(features, vec))
                for pid, vec in self.person_database.items()
            }
            return max(sims.items(), key=lambda x: x[1])
