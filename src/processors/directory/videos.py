from pathlib import Path

VIDEOS_DIR_STR = "resources/videos"

class VideosDirectoryProcessor:
    def __init__(self):
        self.videos_dir_path = Path(VIDEOS_DIR_STR)
        self.input_dir_path = self.videos_dir_path / "input"
        self.output_dir_path = self.videos_dir_path / "output"

    def validate_directories(self) -> bool:
        if not self.videos_dir_path.exists():
            return False

        if not self.videos_dir_path.is_dir():
            return False

    def create_output_directory(self) -> None:
        self.output_dir_path.mkdir(parents=True, exist_ok=True)
