import logging
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
        self.logger = logging.getLogger(__name__)

    def create_output_directory(self) -> None:
        """ディレクトリの作成"""
        self.logger.info("出力ディレクトリの作成を開始します...")
        self.output_dir_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"出力ディレクトリを準備しました")

    def validate_directories(self) -> None:
        """ディレクトリの存在確認と作成"""
        self.logger.info("ディレクトリの存在確認と作成を開始します...")

        if not self.data_set_dir_path.exists():
            raise FileNotFoundError(
                f"データセットディレクトリが存在しません: {self.data_set_dir_path}")

        if not self.data_set_dir_path.is_dir():
            raise NotADirectoryError(
                f"指定されたパスはディレクトリではありません: {self.data_set_dir_path}")

        if not self.input_dir_path.exists():
            raise FileNotFoundError(f"入力ディレクトリが存在しません: {self.input_dir_path}")

        if not self.input_dir_path.is_dir():
            raise NotADirectoryError(
                f"指定されたパスはディレクトリではありません: {self.input_dir_path}")

        if not self.data_set_gallery_dir_path.exists():
            raise FileNotFoundError(
                f"galleryディレクトリが存在しません: {self.data_set_gallery_dir_path}")

        if not self.data_set_gallery_dir_path.is_dir():
            raise NotADirectoryError(
                f"指定されたパスはディレクトリではありません: {self.data_set_gallery_dir_path}")

        if not self.data_set_query_dir_path.exists():
            raise FileNotFoundError(
                f"queryディレクトリが存在しません: {self.data_set_query_dir_path}")

        if not self.data_set_query_dir_path.is_dir():
            raise NotADirectoryError(
                f"指定されたパスはディレクトリではありません: {self.data_set_query_dir_path}")

        if not self.output_dir_path.exists():
            raise FileNotFoundError(
                f"出力ディレクトリが存在しません: {self.output_dir_path}")

        if not self.output_dir_path.is_dir():
            raise NotADirectoryError(
                f"指定されたパスはディレクトリではありません: {self.output_dir_path}")

        self.logger.info("ディレクトリの存在確認と作成が完了しました")
