import torch


class ServiceIdentifyPersonAssignId:
    def __init__(self):
        self.person_ids = []
        self.features = torch.Tensor([])
        self.next_person_id = 1
        self.similarity_threshold = 0.891

    def assign(self, query_feature: torch.Tensor) -> int:
        gallery_features = self.features
        gallery_person_ids = self.person_ids

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

        self.features = torch.cat([self.features, query_feature], dim=0)
        self.person_ids.append(return_person_id)

        return return_person_id
