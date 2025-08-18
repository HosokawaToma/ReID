import torch
from typing import List
from data_class.person_data_set_features import PersonDataSetFeatures

class AssignPersonIdPostProcessor:
    def __init__(self, similarity_threshold: float):
        self.similarity_threshold = similarity_threshold
        self.next_person_id = 0
        self.gallery_features = PersonDataSetFeatures(
            persons_id=[], cameras_id=[], views_id=[], features=torch.tensor([]))

    def assign_person_id(
        self,
        query_feat: torch.Tensor
    ) -> int:
        """人物IDを割り当てる"""
        if query_feat is None or self.gallery_features.features is None or self.gallery_features.features.numel() == 0:
            self.next_person_id += 1
            return_person_id = self.next_person_id
        else:
            similarities = torch.nn.functional.cosine_similarity(
                query_feat, self.gallery_features.features, dim=1, eps=1e-8)

            valid_indices = similarities > self.similarity_threshold

            if valid_indices.any():
                best_idx = torch.argmax(similarities).item()
                best_sim = similarities[best_idx].item()

                if best_sim > self.similarity_threshold:
                    return_person_id = self.gallery_features.persons_id[best_idx]

            if return_person_id is None:
                self.next_person_id += 1
                return_person_id = self.next_person_id

        self.gallery_features.persons_id.append(return_person_id)
        self.gallery_features.cameras_id.append(0)
        self.gallery_features.views_id.append(0)
        self.gallery_features.features = torch.cat([self.gallery_features.features, query_feat], dim=0)

        return return_person_id
