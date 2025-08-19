from pathlib import Path
from typing import List

class DataSetDirectoryProcessor:
    def __init__(self, use_data_set_name: str):
        self.DATA_SET_DIR_PATH = Path("resources/data_sets")
        self.use_data_set_name = use_data_set_name
        self.use_data_set_dir_path = self.DATA_SET_DIR_PATH / self.use_data_set_name
        self.input_dir_path = self.use_data_set_dir_path / "input"
        self.output_dir_path = self.use_data_set_dir_path / "output"
        self.gallery_dir_path = self.input_dir_path / "gallery"
        self.query_dir_path = self.input_dir_path / "query"

    def create_output_directory(self) -> None:
        """ディレクトリの作成"""
        self.output_dir_path.mkdir(parents=True, exist_ok=True)

    def validate_directories(self) -> bool:
        """ディレクトリの存在確認と作成"""

        if not self.use_data_set_dir_path.exists():
            return False

        if not self.use_data_set_dir_path.is_dir():
            return False

        if not self.input_dir_path.exists():
            return False

        if not self.input_dir_path.is_dir():
            return False

        if not self.gallery_dir_path.exists():
            return False

        if not self.gallery_dir_path.is_dir():
            return False

        if not self.query_dir_path.exists():
            return False

        if not self.query_dir_path.is_dir():
            return False

        return True

    def get_data_set_gallery_image_file_paths(self) -> List[Path]:
        """データセットのギャラリー画像ファイルパスを取得"""
        return list(self.gallery_dir_path.glob("*.jpg"))

    def get_data_set_query_image_file_paths(self) -> List[Path]:
        """データセットのクエリ画像ファイルパスを取得"""
        return list(self.query_dir_path.glob("*.jpg"))
