from dataclasses import dataclass


@dataclass
class YoloKeypoint:
    _x: float
    _y: float
    _confidence: float

    def __init__(self, x: float, y: float, confidence: float):
        self._x = x
        self._y = y
        self._confidence = confidence

    def get_x(self) -> float:
        return self._x

    def get_y(self) -> float:
        return self._y

    def get_coordinate(self) -> tuple[float, float]:
        return self._x, self._y

    def get_confidence(self) -> float:
        return self._confidence
