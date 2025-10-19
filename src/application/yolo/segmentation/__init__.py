import numpy as np
from ultralytics import YOLO
from entities.yolo.mask import EntityYoloMask
from ultralytics.engine.results import Results, Masks


class ApplicationYoloSegmentation:
    def __init__(self):
        self.model = YOLO("models/yolo11x-seg.pt")

    def extract(self, frame):
        results = self.model(frame, classes=[0], verbose=False)
        result: Results = results[0]
        if result.masks is None:
            return []
        masks = []
        for mask in result.masks:
            masks.append(EntityYoloMask(result.orig_img, mask.xy.pop()))
        return masks
