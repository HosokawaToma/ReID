"""データセットマネージャーモジュール"""
import cv2
import numpy as np
from pathlib import Path
import logging


class DataSetManager:
    """データセットマネージャークラス"""

    def __init__(self, data_set_name: str = "market1501"):
        """初期化"""
        print("DataSetManager初期化開始...")
        self.logger = logging.getLogger(__name__)
        self.data_set_name = data_set_name
        self.logger.info(f"データセットを{self.data_set_name}に設定しました")
        print("DataSetManager初期化完了")

    def _load_image_market1501(self, file_path: Path) -> tuple[int, int, int, np.ndarray]:
        """
        画像ファイルを読み込み、ファイル名からID情報を抽出する

        :param file_path: 画像ファイルのパス
        :return: (person_id, camera_id, view_id, image)のタプル
        """
        try:
            image = cv2.imread(str(file_path))
            if image is None:
                raise ValueError(f"画像の読み込みに失敗しました: {file_path}")

            file_stem = file_path.stem
            parts = file_stem.split("_")

            if len(parts) < 2:
                raise ValueError(f"ファイル名の形式が正しくありません: {file_path}")

            person_id = int(parts[0])

            camera_view_part = parts[1]
            if not camera_view_part.startswith('c') or 's' not in camera_view_part:
                raise ValueError(f"カメラ・ビュー情報の形式が正しくありません: {camera_view_part}")

            camera_view_parts = camera_view_part[1:].split('s')
            if len(camera_view_parts) != 2:
                raise ValueError(f"カメラ・ビュー情報の形式が正しくありません: {camera_view_part}")

            camera_id = int(camera_view_parts[0]) - 1
            view_id = int(camera_view_parts[1]) - 1

            return person_id, camera_id, view_id, image

        except Exception as e:
            self.logger.error(f"画像読み込みエラー: {file_path} - {e}")
            raise

    def _load_image_osaka(self, file_path: Path) -> tuple[int, int, int, np.ndarray]:
        """
        画像ファイルを読み込み、ファイル名からID情報を抽出する

        :param file_path: 画像ファイルのパス
        :return: (person_id, camera_id, view_id, image)のタプル
        """
        try:
            image = cv2.imread(str(file_path))
            if image is None:
                raise ValueError(f"画像の読み込みに失敗しました: {file_path}")

            file_stem = file_path.stem
            parts = file_stem.split("_")

            if len(parts) < 2:
                raise ValueError(f"ファイル名の形式が正しくありません: {file_path}")

            person_id = int(parts[0])
            camera_id = int(parts[1])

            return person_id, camera_id, 0, image

        except Exception as e:
            self.logger.error(f"画像読み込みエラー: {file_path} - {e}")
            raise

    def load_image(self, file_path: Path) -> tuple[int, int, int, np.ndarray]:
        """
        画像ファイルを読み込み、ファイル名からID情報を抽出する

        :param file_path: 画像ファイルのパス
        :return: (person_id, camera_id, view_id, image)のタプル
        """
        if self.data_set_name == "market1501":
            return self._load_image_market1501(file_path)
        elif self.data_set_name == "osaka":
            return self._load_image_osaka(file_path)
        else:
            raise ValueError(f"不明なデータセットです: {self.data_set_name}")
