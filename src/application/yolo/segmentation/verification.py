from entities.yolo.mask import EntityYoloMask


class ApplicationYoloSegmentationVerification:
    def verify(self, masks: list[EntityYoloMask]):
        if len(masks) > 1:
            return False
        return True
