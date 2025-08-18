from pathlib import Path

DATA_SET_DIR_STR = "resources/data_sets"

class DataSetDirectoryProcessor:
    def __init__(self, use_data_set_name: str):
        self.data_set_dir_path = Path(DATA_SET_DIR_STR)
        self.use_data_set_dir_path = Path(self.data_set_dir_path / use_data_set_name)
        self.input_dir_path = Path(self.data_set_dir_path / "input")
        self.data_set_gallery_dir_path = Path(self.use_data_set_dir_path / "gallery")
        self.data_set_query_dir_path = Path(self.use_data_set_dir_path / "query")
        self.output_dir_path = Path(self.use_data_set_dir_path / "output")

    def create_output_directory(self) -> None:
        """ディレクトリの作成"""
        self.output_dir_path.mkdir(parents=True, exist_ok=True)

    def validate_directories(self) -> bool:
        """ディレクトリの存在確認と作成"""

        if not self.data_set_dir_path.exists():
            return False

        if not self.data_set_dir_path.is_dir():
            return False

        if not self.input_dir_path.exists():
            return False

        if not self.input_dir_path.is_dir():
            return False

        if not self.data_set_gallery_dir_path.exists():
            return False

        if not self.data_set_gallery_dir_path.is_dir():
            return False

        if not self.data_set_query_dir_path.exists():
            return False

        if not self.data_set_query_dir_path.is_dir():
            return False

        if not self.output_dir_path.exists():
            return False

        if not self.output_dir_path.is_dir():
            return False

        return True
