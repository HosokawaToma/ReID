from entities.yolo.keypoints import EntityYoloKeypoint

class ApplicationYoloPoseVerification:
    def __init__(
        self,
        confidence_threshold: float = 0.75
    ):
        self.confidence_threshold = confidence_threshold

    def verify(self, keypoints: list[EntityYoloKeypoint]) -> bool:
        if len(keypoints) != 17:
            return False
        sum_confidence = sum(keypoint.confidence for keypoint in keypoints)
        mean_confidence = sum_confidence / len(keypoints)
        if mean_confidence < self.confidence_threshold:
            return False
        return True
