import torch
from typing import List

class AssignPersonIdPostProcessor:
    def __init__(self, similarity_threshold: float):
        self.similarity_threshold = similarity_threshold
        self.next_person_id = 0

    def assign_person_id(
        self,
        query_feat: torch.Tensor,
        gallery_feats: torch.Tensor,
        gallery_person_ids: List[int]
    ) -> int:
        """人物IDを割り当てる"""
        if query_feat is None or gallery_feats is None or gallery_feats.numel() == 0:
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
