import numpy as np
import torch
import torch.nn.functional as F
from torchreid import metrics
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from post_processing.k_reciprocal_encoding import re_ranking

class ReIDEvaluator:
    def __init__(self, max_rank: int = 50, metric: str = "cosine",
                 normalize: bool = True, rerank: bool = False,
                 k1: int = 20, k2: int = 6, lambda_value: float = 0.3):
        self.max_rank = max_rank
        self.metric = metric  # "cosine" or "euclidean"
        self.normalize = normalize
        self.rerank = rerank
        self.k1, self.k2, self.lambda_value = k1, k2, lambda_value

        self.q_feats, self.q_pids, self.q_camids = [], [], []
        self.g_feats, self.g_pids, self.g_camids = [], [], []

    # ---- data input ----
    def add_query(self, feat: np.ndarray | torch.Tensor, pid: int, camid: int):
        self.q_feats.append(torch.as_tensor(feat, dtype=torch.float32))
        self.q_pids.append(int(pid))
        self.q_camids.append(int(camid))

    def add_gallery(self, feat: np.ndarray | torch.Tensor, pid: int, camid: int):
        self.g_feats.append(torch.as_tensor(feat, dtype=torch.float32))
        self.g_pids.append(int(pid))
        self.g_camids.append(int(camid))

    # ---- evaluate ----
    def evaluate(self):
        assert self.q_feats and self.g_feats, "query / gallery を投入してください。"

        q = torch.stack(self.q_feats, 0)  # (Nq,D)
        g = torch.stack(self.g_feats, 0)  # (Ng,D)

        if self.normalize:
            q = F.normalize(q, dim=1)
            g = F.normalize(g, dim=1)

        # 距離行列 (Nq,Ng)
        dist = metrics.compute_distance_matrix(q, g, metric=self.metric).cpu().numpy()

        if self.rerank:
            # k-reciprocal re-ranking（Zhong+ CVPR'17）
            # 独自実装を使用
            q_q_dist = metrics.compute_distance_matrix(q, q, metric=self.metric).cpu().numpy()
            g_g_dist = metrics.compute_distance_matrix(g, g, metric=self.metric).cpu().numpy()
            dist = re_ranking(dist, q_q_dist, g_g_dist, k1=self.k1, k2=self.k2, lambda_value=self.lambda_value)

        q_pids = np.asarray(self.q_pids, dtype=np.int64)
        g_pids = np.asarray(self.g_pids, dtype=np.int64)
        q_camids = np.asarray(self.q_camids, dtype=np.int64)
        g_camids = np.asarray(self.g_camids, dtype=np.int64)

        # evaluate_rank関数の戻り値を確認して適切に処理
        eval_result = metrics.evaluate_rank(
            dist, q_pids, g_pids, q_camids, g_camids,
            max_rank=self.max_rank, use_metric_cuhk03=False, use_cython=True
        )

        # 戻り値の数に応じて処理を分岐
        if len(eval_result) == 2:
            cmc, mAP = eval_result
        elif len(eval_result) == 3:
            cmc, mAP, _ = eval_result
        else:
            raise ValueError(f"予期しない戻り値の数: {len(eval_result)}")

        return {
            "cmc": cmc,                  # array shape (max_rank,)
            "rank1": float(cmc[0]),
            "mAP": float(mAP),
            "distmat": dist,             # 再利用したい場合に
        }

    def compute_rank_k_accuracy(self, results: Dict[str, Any], k_values: List[int] = None) -> Dict[str, float]:
        """Rank-k Accuracyを計算

        Args:
            results: evaluate()の結果
            k_values: 計算するランクのリスト (デフォルト: [1, 5, 10, 20])

        Returns:
            Rank-k Accuracyの辞書
        """
        if k_values is None:
            k_values = [1, 5, 10, 20]

        cmc = results["cmc"]
        rank_k_acc = {}

        for k in k_values:
            if k <= len(cmc):
                rank_k_acc[f"rank{k}"] = float(cmc[k-1])
            else:
                rank_k_acc[f"rank{k}"] = float(cmc[-1])  # 最大ランクの値を使用

        return rank_k_acc

    def plot_cmc_curve(self, results: Dict[str, Any], save_path: str = None,
                       title: str = "CMC Curve", max_rank_display: int = 50) -> None:
        """CMC曲線をプロット

        Args:
            results: evaluate()の結果
            save_path: 保存パス (Noneの場合は表示のみ)
            title: グラフのタイトル
            max_rank_display: 表示する最大ランク
        """
        cmc = results["cmc"]
        mAP = results["mAP"]

        # 表示するランク数を制限
        display_ranks = min(max_rank_display, len(cmc))
        ranks = np.arange(1, display_ranks + 1)
        cmc_display = cmc[:display_ranks]

        plt.figure(figsize=(10, 6))
        plt.plot(ranks, cmc_display * 100, 'b-', linewidth=2, marker='o', markersize=3)
        plt.xlabel('Rank')
        plt.ylabel('Matching Rate (%)')
        plt.title(f'{title} (mAP: {mAP:.1%})')
        plt.grid(True, alpha=0.3)
        plt.xlim(1, display_ranks)
        plt.ylim(0, 100)

        # 主要なランクにアノテーションを追加
        key_ranks = [1, 5, 10, 20]
        for rank in key_ranks:
            if rank <= display_ranks:
                plt.annotate(f'Rank-{rank}: {cmc[rank-1]:.1%}',
                           xy=(rank, cmc[rank-1] * 100),
                           xytext=(rank + display_ranks * 0.05, cmc[rank-1] * 100 + 2),
                           fontsize=9,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"CMC曲線を保存しました: {save_path}")
        else:
            plt.show()

        plt.close()

    def save_evaluation_results(self, results: Dict[str, Any], output_dir: str,
                               filename: str = "evaluation_results.json") -> None:
        """評価結果をJSONファイルに保存

        Args:
            results: evaluate()の結果
            output_dir: 出力ディレクトリ
            filename: ファイル名
        """
        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # NumPy配列をリストに変換してJSON保存可能にする
        save_data = {
            "mAP": float(results["mAP"]),
            "rank1": float(results["rank1"]),
            "cmc": results["cmc"].tolist(),
            "rank_k_accuracy": self.compute_rank_k_accuracy(results),
            "evaluation_settings": {
                "max_rank": self.max_rank,
                "metric": self.metric,
                "normalize": self.normalize,
                "rerank": self.rerank,
                "k1": self.k1,
                "k2": self.k2,
                "lambda_value": self.lambda_value
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"評価結果を保存しました: {output_path}")

    def print_evaluation_summary(self, results: Dict[str, Any]) -> None:
        """評価結果のサマリーを表示

        Args:
            results: evaluate()の結果
        """
        print("\n=== ReID評価結果 ===")
        print(f"mAP: {results['mAP']:.1%}")

        rank_k_acc = self.compute_rank_k_accuracy(results)
        for rank_name, accuracy in rank_k_acc.items():
            print(f"{rank_name.upper()}: {accuracy:.1%}")

        print(f"評価設定:")
        print(f"  - 距離メトリック: {self.metric}")
        print(f"  - 正規化: {self.normalize}")
        print(f"  - Re-ranking: {self.rerank}")
        if self.rerank:
            print(f"    - k1: {self.k1}, k2: {self.k2}, lambda: {self.lambda_value}")

    def reset(self) -> None:
        """評価データをリセット"""
        self.q_feats.clear()
        self.q_pids.clear()
        self.q_camids.clear()
        self.g_feats.clear()
        self.g_pids.clear()
        self.g_camids.clear()
