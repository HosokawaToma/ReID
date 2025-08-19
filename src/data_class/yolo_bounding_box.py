from dataclasses import dataclass


@dataclass
class YoloBoundingBox:
    _x1: int
    _y1: int
    _x2: int
    _y2: int

    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        self._x1 = x1
        self._y1 = y1
        self._x2 = x2
        self._y2 = y2

    def get_coordinate(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2
