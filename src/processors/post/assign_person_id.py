import torch
from data_class.person_data_set_features import PersonDataSetFeatures

SIMILARITY_THRESHOLD = 0.85

class AssignPersonIdPostProcessor:
    def __init__(self, device: str):
        self.similarity_threshold = SIMILARITY_THRESHOLD
        self.next_person_id = 1
        self.gallery_data_set_features = PersonDataSetFeatures(device=device)

    def assign_person_id(
        self,
        query_feature: torch.Tensor
    ) -> int:
        """人物IDを割り当てる"""
        gallery_features = self.gallery_data_set_features.get_features()
        gallery_person_ids = self.gallery_data_set_features.get_person_ids()

        if gallery_features.numel() == 0:
            return_person_id = self.next_person_id
            self.next_person_id += 1
        else:
            similarities = torch.nn.functional.cosine_similarity(
                query_feature, gallery_features, dim=1, eps=1e-8)

            best_sim, best_idx = torch.max(similarities, dim=0)

            if best_sim.item() > self.similarity_threshold:
                return_person_id = gallery_person_ids[best_idx]
            else:
                return_person_id = self.next_person_id
                self.next_person_id += 1

        self.gallery_data_set_features.add_feature(query_feature)
        self.gallery_data_set_features.add_person_id(return_person_id)
        self.gallery_data_set_features.add_camera_id(0)
        self.gallery_data_set_features.add_view_id(0)

        return return_person_id
