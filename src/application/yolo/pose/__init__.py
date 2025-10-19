import numpy as np
from ultralytics import YOLO
from entities.yolo.keypoints import EntityYoloKeypoint
from ultralytics.engine.results import Results

class ApplicationYoloPose:
    def __init__(
        self):
        self.yolo_pose = YOLO("models/yolo11x-pose.pt")

    def extract(self, frame: np.ndarray) -> list[EntityYoloKeypoint]:
        results = self.yolo_pose(frame, verbose=False)
        result: Results = results[0]
        if result.keypoints is None:
            return []
        if len(result.keypoints) == 0:
            return []
        xys = result.keypoints.xy[0].tolist()
        confs = result.keypoints.conf[0].tolist()
        keypoints = []
        for xy, confidence in zip(xys, confs):
            keypoints.append(EntityYoloKeypoint(xy, confidence))
        return keypoints
