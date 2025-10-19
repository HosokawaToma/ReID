from dataclasses import dataclass


@dataclass
class EntityYoloDetection:
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int
    confidence: float
