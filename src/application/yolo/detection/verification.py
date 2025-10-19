from src.entities.yolo.detection import EntityYoloDetection


class YoloDetectionVerification:
    def __init__(
        self,
        region_points: list[tuple[tuple[int, int], tuple[int, int]]] = [((0, 0), (100, 100))],
    ):
        self.region_points = region_points

    def verify(self, yolo_detection: EntityYoloDetection) -> bool:
        x1, y1, x2, y2 = yolo_detection.x1, yolo_detection.y1, yolo_detection.x2, yolo_detection.y2
        for region_point_1, region_point_2 in self.region_points:
            region_x1, region_y1 = region_point_1
            region_x2, region_y2 = region_point_2
            if x1 < region_x1 and y1 < region_y1 and x2 > region_x2 and y2 > region_y2:
                return True
        return False
