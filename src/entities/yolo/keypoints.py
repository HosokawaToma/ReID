from dataclasses import dataclass
import numpy as np

@dataclass
class EntityYoloKeypoint:
    xy: np.ndarray
    confidence: float
