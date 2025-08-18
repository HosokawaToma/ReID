import torch
from typing import List, Tuple
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve

class ComputeRocEerF1PostProcessor:
    def __init__(self):
        pass

    def compute_roc_eer_f1(
        self,
        query_feats: torch.Tensor,
        gallery_feats: torch.Tensor,
        query_person_ids: List[int],
        gallery_person_ids: List[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
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
        precision, recall, pr_thresholds = precision_recall_curve(
            y_true, scores)
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
