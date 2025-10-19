import torch
from copy import deepcopy


class ApplicationReIDAssignID:
    def __init__(self, similarity_threshold: float = 0.891):
        self.similarity_threshold = similarity_threshold
        self.next_person_id = 1
        self.gallery_features = None
        self.gallery_person_ids = []

    def assign_person_id(
        self,
        query_feature: torch.Tensor
    ) -> int:
        """人物IDを割り当てる"""
        if self.gallery_features is None:
            self.gallery_features = query_feature.clone()
            self.gallery_person_ids.append(deepcopy(self.next_person_id))
            return self.next_person_id

        similarities = torch.nn.functional.cosine_similarity(
            query_feature, self.gallery_features, dim=1, eps=1e-8)
        best_sim, best_idx = torch.max(similarities, dim=0)

        if best_sim.item() > self.similarity_threshold:
            
            self.gallery_features = torch.cat([self.gallery_features, query_feature.clone()], dim=0)
            self.gallery_person_ids.append(deepcopy(self.gallery_person_ids[best_idx]))
            return self.gallery_person_ids[best_idx]

        self.next_person_id += 1
        self.gallery_features = torch.cat([self.gallery_features, query_feature.clone()], dim=0)
        self.gallery_person_ids.append(deepcopy(self.next_person_id))
        return self.next_person_id
