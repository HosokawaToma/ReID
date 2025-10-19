from dataclasses import dataclass
import numpy as np
@dataclass
class EntityYoloMask:
    original_frame: np.ndarray
    xy: np.ndarray
