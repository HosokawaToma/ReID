import numpy as np
from dataclasses import dataclass

@dataclass
class PersonDetections:
    bounding_box: np.ndarray
    person_crop: np.ndarray
    person_id: int

    def __init__(self, bounding_box: np.ndarray, person_crop: np.ndarray, person_id: int):
        self.bounding_box = bounding_box
        self.person_crop = person_crop
        self.person_id = person_id
