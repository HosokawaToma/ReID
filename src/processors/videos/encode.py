from pathlib import Path
import cv2
import numpy as np
from typing import Callable

VIDEOS_DIR_STR = "resources/videos"


class VideosEncodeProcessor:
    def __init__(self):
        self.videos_dir_path = Path(VIDEOS_DIR_STR)
        self.input_dir_path = self.videos_dir_path / "input"
        self.output_dir_path = self.videos_dir_path / "output"

    def encode_videos(self, process_frame: Callable[[np.ndarray], np.ndarray]) -> None:
        video_files = list(self.input_dir_path.glob("*.mp4"))
        for video_file_path in video_files:
            self._process_video(video_file_path, process_frame)

    def _process_video(self, video_file_path: Path, process_frame: Callable[[np.ndarray], np.ndarray]) -> None:
        cap = cv2.VideoCapture(str(video_file_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_file_name = f"{video_file_path.stem}_reid.mp4"
        output_path = self.output_dir_path / output_file_name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = process_frame(frame)
            out.write(processed_frame)
        cap.release()
        out.release()
