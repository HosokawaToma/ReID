import cv2
import numpy as np
from typing import Generator

class ApplicationVideoLoader:
    def __init__(self, video_file_path: str):
        self.video_file_path = video_file_path
        self.video_capture = cv2.VideoCapture(video_file_path)

    def get_frame_rate(self):
        return int(self.video_capture.get(cv2.CAP_PROP_FPS))

    def get_frame_width(self):
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    def get_frame_height(self):
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def load_video_frame(self) -> Generator[np.ndarray, None, None]:
        while self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if not ret:
                break
            if frame is None:
                continue
            yield frame
