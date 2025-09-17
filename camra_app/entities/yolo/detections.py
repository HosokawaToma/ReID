from dataclasses import dataclass

from entities.yolo.bounding_box import EntityYoloBoundingBox


@dataclass
class EntityYoloDetections:
    _detection_id: int
    _bounding_box: EntityYoloBoundingBox
    _confidence: float
    _class: int

    def __init__(
        self,
        detection_id: int,
        bounding_box: EntityYoloBoundingBox,
        confidence: float,
        cls: int,
    ):
        self._detection_id = detection_id
        self._bounding_box = bounding_box
        self._confidence = confidence
        self._class = cls

    def get_confidence(self) -> float:
        return self._confidence

    def get_class(self) -> int:
        return self._class

    def get_detection_id(self) -> int:
        return self._detection_id

    def get_bounding_box(self) -> EntityYoloBoundingBox:
        return self._bounding_box
