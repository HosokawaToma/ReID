import torch
from data_class.person_data_set_features import PersonDataSetFeatures

class AssignPersonIdPostProcessor:
    def __init__(self, similarity_threshold: float):
        self.similarity_threshold = similarity_threshold
        self.next_person_id = 0
        self.gallery_data_set_features = PersonDataSetFeatures()

    def assign_person_id(
        self,
        query_feature: torch.Tensor
    ) -> int:
        """人物IDを割り当てる"""
        gallery_features = self.gallery_data_set_features.get_features()
        gallery_person_ids = self.gallery_data_set_features.get_person_ids()

        if query_feature is None or gallery_features is None or gallery_features.numel() == 0:
            self.next_person_id += 1
            return_person_id = self.next_person_id
        else:
            similarities = torch.nn.functional.cosine_similarity(
                query_feature, gallery_features, dim=1, eps=1e-8)

            valid_indices = similarities > self.similarity_threshold

            if valid_indices.any():
                best_idx = torch.argmax(similarities).item()
                best_sim = similarities[best_idx].item()

                if best_sim > self.similarity_threshold:
                    return_person_id = gallery_person_ids[best_idx]

            if return_person_id is None:
                self.next_person_id += 1
                return_person_id = self.next_person_id

        self.gallery_data_set_features.add_feature(query_feature)
        self.gallery_data_set_features.add_person_id(return_person_id)
        self.gallery_data_set_features.add_camera_id(0)
        self.gallery_data_set_features.add_view_id(0)

        return return_person_id
