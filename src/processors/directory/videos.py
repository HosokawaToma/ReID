from pathlib import Path
from typing import List

VIDEOS_DIR_STR = "resources/videos"

class VideosDirectoryProcessor:
    def __init__(self):
        self._videos_dir_path = Path(VIDEOS_DIR_STR)
        self._input_dir_path = self._videos_dir_path / "input"
        self._output_dir_path = self._videos_dir_path / "output"

    def validate_directories(self) -> bool:
        if not self._videos_dir_path.exists():
            return False

        if not self._videos_dir_path.is_dir():
            return False

    def create_output_directory(self) -> None:
        self._output_dir_path.mkdir(parents=True, exist_ok=True)

    def get_video_file_paths(self) -> List[Path]:
        return sorted(list(self._input_dir_path.glob("*.mp4")))

    def get_output_dir_path(self) -> Path:
        return self._output_dir_path
