from pathlib import Path
from typing import List

VIDEOS_DIR_STR = "resources/videos"

class VideosDirectoryProcessor:
    def __init__(self):
        self.videos_dir_path = Path(VIDEOS_DIR_STR)
        self.input_dir_path = self.videos_dir_path / "input"
        self.output_dir_path = self.videos_dir_path / "output"
        self.clahe_dir_path = self.output_dir_path / "clahe"
        self.retinex_dir_path = self.output_dir_path / "retinex"
        self.homorphic_filter_dir_path = self.output_dir_path / "homorphic_filter"
        self.logarithmic_transform_dir_path = self.output_dir_path / "logarithmic_transform"
        self.wavelet_dir_path = self.output_dir_path / "wavelet"
        self.ace_dir_path = self.output_dir_path / "ace"
        self.anisotropic_diffusion_dir_path = self.output_dir_path / "anisotropic_diffusion"

    def validate_directories(self) -> bool:
        if not self.videos_dir_path.exists():
            return False

        if not self.videos_dir_path.is_dir():
            return False

    def create_output_directory(self) -> None:
        self.output_dir_path.mkdir(parents=True, exist_ok=True)
        self.clahe_dir_path.mkdir(parents=True, exist_ok=True)
        self.retinex_dir_path.mkdir(parents=True, exist_ok=True)
        self.homorphic_filter_dir_path.mkdir(parents=True, exist_ok=True)
        self.logarithmic_transform_dir_path.mkdir(parents=True, exist_ok=True)
        self.wavelet_dir_path.mkdir(parents=True, exist_ok=True)
        self.ace_dir_path.mkdir(parents=True, exist_ok=True)
        self.anisotropic_diffusion_dir_path.mkdir(parents=True, exist_ok=True)

    def get_video_file_paths(self) -> List[Path]:
        return list(self.input_dir_path.glob("*.mp4"))
