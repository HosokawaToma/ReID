"""ファイル処理の共通ロジック"""
import os
import logging
from pathlib import Path
from typing import List, Dict
from config import FILE_PROCESSING_CONFIG


class FileProcessor:
    """ファイル処理の共通ロジック"""

    def __init__(self):
        """初期化"""
        self.logger = logging.getLogger(__name__)

    def discover_files(self, input_dir: str, extensions: List[str]) -> List[str]:
        """指定されたディレクトリから対象ファイルを検出

        Args:
            input_dir: 入力ディレクトリパス
            extensions: 対象拡張子のリスト

        Returns:
            検出されたファイルパスのリスト
        """
        if not os.path.exists(input_dir):
            self.logger.error(f"入力ディレクトリが存在しません: {input_dir}")
            return []

        if not os.path.isdir(input_dir):
            self.logger.error(f"指定されたパスはディレクトリではありません: {input_dir}")
            return []

        discovered_files = []
        input_path = Path(input_dir)

        # 拡張子を小文字に統一
        extensions_lower = [ext.lower() for ext in extensions]

        try:
            # ディレクトリ内のファイルを再帰的に検索
            for file_path in input_path.rglob('*'):
                if file_path.is_file():
                    file_extension = file_path.suffix.lower()
                    if file_extension in extensions_lower:
                        discovered_files.append(str(file_path))

            self.logger.info(f"発見されたファイル数: {len(discovered_files)} (ディレクトリ: {input_dir})")

            # ファイル名でソート
            discovered_files.sort()

        except Exception as e:
            self.logger.error(f"ファイル検索中にエラーが発生しました: {e}")
            return []

        return discovered_files

    def validate_files(self, file_paths: List[str]) -> List[str]:
        """ファイルの存在と読み込み可能性を検証

        Args:
            file_paths: 検証対象のファイルパスリスト

        Returns:
            検証に通ったファイルパスのリスト
        """
        valid_files = []

        for file_path in file_paths:
            try:
                # ファイルの存在確認
                if not os.path.exists(file_path):
                    self.logger.warning(f"ファイルが存在しません: {file_path}")
                    continue

                # ファイルかどうか確認
                if not os.path.isfile(file_path):
                    self.logger.warning(f"指定されたパスはファイルではありません: {file_path}")
                    continue

                # 読み込み権限確認
                if not os.access(file_path, os.R_OK):
                    self.logger.warning(f"ファイルの読み込み権限がありません: {file_path}")
                    continue

                # ファイルサイズ確認（0バイトファイルを除外）
                if os.path.getsize(file_path) == 0:
                    self.logger.warning(f"ファイルサイズが0バイトです: {file_path}")
                    continue

                valid_files.append(file_path)

            except Exception as e:
                self.logger.error(f"ファイル検証中にエラーが発生しました ({file_path}): {e}")
                continue

        self.logger.info(f"検証に通ったファイル数: {len(valid_files)}/{len(file_paths)}")
        return valid_files

    def create_output_structure(self, output_dir: str, input_files: List[str]) -> Dict[str, str]:
        """出力ディレクトリ構造を作成

        Args:
            output_dir: 出力ディレクトリパス
            input_files: 入力ファイルパスのリスト

        Returns:
            入力ファイルパスと出力ファイルパスのマッピング辞書
        """
        output_mapping = {}

        try:
            # 出力ディレクトリを作成
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # 処理済みファイル保存用のディレクトリを作成
            processed_dir = output_path / "processed"
            processed_dir.mkdir(exist_ok=True)

            # サマリー保存用のディレクトリを作成
            summary_dir = output_path / "summary"
            summary_dir.mkdir(exist_ok=True)

            self.logger.info(f"出力ディレクトリ構造を作成しました: {output_dir}")

            # 各入力ファイルに対応する出力パスを生成（指定されたディレクトリに直接保存）
            for input_file in input_files:
                input_path = Path(input_file)

                # 出力ファイル名を生成（拡張子は処理時に決定）
                output_filename = input_path.stem
                output_file_path = output_path / output_filename

                output_mapping[input_file] = str(output_file_path)

            self.logger.info(f"出力パスマッピングを作成しました: {len(output_mapping)}件")

        except Exception as e:
            self.logger.error(f"出力ディレクトリ構造の作成中にエラーが発生しました: {e}")
            return {}

        return output_mapping

    def get_video_extensions(self) -> List[str]:
        """サポートされている動画拡張子を取得"""
        return list(FILE_PROCESSING_CONFIG.video_extensions)

    def get_image_extensions(self) -> List[str]:
        """サポートされている画像拡張子を取得"""
        return list(FILE_PROCESSING_CONFIG.image_extensions)

    def is_video_file(self, file_path: str) -> bool:
        """ファイルが動画ファイルかどうか判定"""
        file_extension = Path(file_path).suffix.lower()
        return file_extension in FILE_PROCESSING_CONFIG.video_extensions

    def is_image_file(self, file_path: str) -> bool:
        """ファイルが画像ファイルかどうか判定"""
        file_extension = Path(file_path).suffix.lower()
        return file_extension in FILE_PROCESSING_CONFIG.image_extensions
