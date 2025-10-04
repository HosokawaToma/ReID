from dataclasses import dataclass
from ultralytics.engine.results import Masks


@dataclass
class YoloMasks:
    _masks: Masks

    def __init__(self, masks: Masks):
        self._masks = masks

    def get_masks(self) -> Masks:
        return self._masks
