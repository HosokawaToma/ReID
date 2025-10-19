import cv2
import numpy as np


class ApplicationVideoWriter:
    def __init__(self, video_file_path: str, frame_rate: int, frame_width: int, frame_height: int):
        self.video_file_path = video_file_path
        self.video_capture = cv2.VideoWriter(
            video_file_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))

    def write_video_frame(self, frame: np.ndarray):
        self.video_capture.write(frame)
