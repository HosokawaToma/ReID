"""結果保存の共通ロジック"""
import os
import json
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from config import FILE_PROCESSING_CONFIG


class ResultSaver:
    """結果保存の共通ロジック"""

    def __init__(self):
        """初期化"""
        self.logger = logging.getLogger(__name__)

    def save_processed_image(self, image: np.ndarray, output_path: str) -> bool:
        """処理済み画像を保存

        Args:
            image: 保存する画像（numpy配列）
            output_path: 出力ファイルパス（拡張子なし）

        Returns:
            保存成功時True、失敗時False
        """
        try:
            # 出力パスに拡張子を追加
            output_file = f"{output_path}.jpg"

            # 出力ディレクトリを作成
            output_dir = Path(output_file).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # 画像品質設定
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, FILE_PROCESSING_CONFIG.output_image_quality]

            # 画像を保存
            success = cv2.imwrite(output_file, image, encode_params)

            if success:
                self.logger.debug(f"画像を保存しました: {output_file}")
                return True
            else:
                self.logger.error(f"画像の保存に失敗しました: {output_file}")
                return False

        except Exception as e:
            self.logger.error(f"画像保存中にエラーが発生しました ({output_path}): {e}")
            return False

    def save_processed_video(self, frames: List[np.ndarray], output_path: str, fps: float) -> bool:
        """処理済み動画を保存

        Args:
            frames: 保存するフレームのリスト
            output_path: 出力ファイルパス（拡張子なし）
            fps: フレームレート

        Returns:
            保存成功時True、失敗時False
        """
        if not frames:
            self.logger.warning(f"保存するフレームがありません: {output_path}")
            return False

        video_writer = None
        try:
            # 出力パスに拡張子を追加
            output_file = f"{output_path}.mp4"

            # 出力ディレクトリを作成
            output_dir = Path(output_file).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # 最初のフレームから動画の設定を取得
            first_frame = frames[0]
            height, width = first_frame.shape[:2]

            # FPSの妥当性チェック
            if fps <= 0 or fps > 120:
                self.logger.warning(f"無効なFPS値: {fps}、デフォルト値30を使用します")
                fps = 30.0

            # 複数のコーデックを試行
            codecs_to_try = [
                ('H264', cv2.VideoWriter_fourcc(*'H264')),
                ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
                ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
                ('MJPG', cv2.VideoWriter_fourcc(*'MJPG'))
            ]

            video_writer = None
            successful_codec = None

            for codec_name, fourcc in codecs_to_try:
                try:
                    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

                    if video_writer.isOpened():
                        # テスト書き込みを実行
                        test_frame = first_frame.copy()
                        video_writer.write(test_frame)
                        successful_codec = codec_name
                        self.logger.debug(f"コーデック {codec_name} で初期化成功")
                        break
                    else:
                        if video_writer:
                            video_writer.release()
                        video_writer = None

                except Exception as codec_error:
                    self.logger.debug(f"コーデック {codec_name} の初期化に失敗: {codec_error}")
                    if video_writer:
                        video_writer.release()
                    video_writer = None
                    continue

            if not video_writer or not video_writer.isOpened():
                self.logger.error(f"すべてのコーデックで VideoWriter の初期化に失敗しました: {output_file}")
                return False

            self.logger.info(f"動画保存開始: {output_file} (コーデック: {successful_codec}, {len(frames)}フレーム, {fps}fps)")

            # フレームを書き込み（最初のフレームは既に書き込み済み）
            for i, frame in enumerate(frames[1:], 1):
                try:
                    # フレームの妥当性チェック
                    if frame is None or frame.size == 0:
                        self.logger.warning(f"無効なフレーム {i} をスキップします")
                        continue

                    # フレームサイズの一貫性チェック
                    frame_height, frame_width = frame.shape[:2]
                    if frame_width != width or frame_height != height:
                        self.logger.warning(f"フレーム {i} のサイズが不一致: {frame_width}x{frame_height} != {width}x{height}")
                        # リサイズして対応
                        frame = cv2.resize(frame, (width, height))

                    video_writer.write(frame)

                    # 進捗ログ（大量のフレームの場合）
                    if len(frames) > 100 and i % 50 == 0:
                        self.logger.debug(f"動画書き込み進捗: {i}/{len(frames)} フレーム")

                except Exception as frame_error:
                    self.logger.warning(f"フレーム {i} の書き込み中にエラー: {frame_error}")
                    continue

            # リソースを解放
            video_writer.release()
            video_writer = None

            # 出力ファイルの存在と妥当性を確認
            if not os.path.exists(output_file):
                self.logger.error(f"動画ファイルが作成されませんでした: {output_file}")
                return False

            file_size = os.path.getsize(output_file)
            if file_size == 0:
                self.logger.error(f"動画ファイルが空です: {output_file}")
                return False

            self.logger.info(f"動画保存完了: {output_file} ({len(frames)}フレーム, {fps}fps, {file_size/1024/1024:.2f}MB)")
            return True

        except Exception as e:
            self.logger.error(f"動画保存中にエラーが発生しました ({output_path}): {e}")
            return False

        finally:
            # リソースの確実な解放
            if video_writer is not None:
                try:
                    video_writer.release()
                except Exception as release_error:
                    self.logger.warning(f"VideoWriter解放エラー: {release_error}")

    def save_results_summary(self, results: Dict[str, Any], output_path: str) -> bool:
        """結果サマリーをJSONで保存

        Args:
            results: 保存する結果データ
            output_path: 出力ファイルパス（拡張子なし、summaryディレクトリに保存）

        Returns:
            保存成功時True、失敗時False
        """
        try:
            # 出力パスを調整（summaryディレクトリに保存）
            output_dir = Path(output_path).parent / "summary"
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / "processing_summary.json"

            # 結果をJSON形式で保存
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=self._json_serializer)

            self.logger.info(f"結果サマリーを保存しました: {output_file}")
            return True

        except Exception as e:
            self.logger.error(f"結果サマリー保存中にエラーが発生しました ({output_path}): {e}")
            return False

    def save_individual_result(self, result: Dict[str, Any], output_path: str, filename: str) -> bool:
        """個別ファイルの処理結果を保存

        Args:
            result: 保存する結果データ
            output_path: 出力ディレクトリパス
            filename: ファイル名（拡張子なし）

        Returns:
            保存成功時True、失敗時False
        """
        try:
            # 出力パスを調整（summaryディレクトリに保存）
            output_dir = Path(output_path).parent / "summary"
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / f"{filename}_result.json"

            # 結果をJSON形式で保存
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=self._json_serializer)

            self.logger.debug(f"個別結果を保存しました: {output_file}")
            return True

        except Exception as e:
            self.logger.error(f"個別結果保存中にエラーが発生しました ({filename}): {e}")
            return False

    def create_output_directories(self, base_output_dir: str) -> Dict[str, str]:
        """出力用ディレクトリ構造を作成

        Args:
            base_output_dir: ベース出力ディレクトリ

        Returns:
            作成されたディレクトリパスの辞書
        """
        try:
            base_path = Path(base_output_dir)

            # 各種ディレクトリを作成
            directories = {
                'base': str(base_path),
                'summary': str(base_path / "summary")
            }

            for dir_type, dir_path in directories.items():
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"{dir_type}ディレクトリを作成: {dir_path}")

            return directories

        except Exception as e:
            self.logger.error(f"出力ディレクトリ作成中にエラーが発生しました: {e}")
            return {}

    def _json_serializer(self, obj: Any) -> Any:
        """JSON シリアライゼーション用のカスタムシリアライザー

        Args:
            obj: シリアライズ対象のオブジェクト

        Returns:
            シリアライズ可能な形式に変換されたオブジェクト
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        else:
            return str(obj)

    def get_output_file_path(self, base_path: str, filename: str, file_type: str) -> str:
        """出力ファイルパスを生成

        Args:
            base_path: ベースパス
            filename: ファイル名（拡張子なし）
            file_type: ファイルタイプ（'image', 'video', 'json'）

        Returns:
            生成された出力ファイルパス
        """
        extensions = {
            'image': '.jpg',
            'video': '.mp4',
            'json': '.json'
        }

        extension = extensions.get(file_type, '')
        return f"{base_path}/{filename}{extension}"

    def cleanup_temp_files(self, temp_dir: str) -> bool:
        """一時ファイルをクリーンアップ

        Args:
            temp_dir: 一時ディレクトリパス

        Returns:
            クリーンアップ成功時True、失敗時False
        """
        try:
            temp_path = Path(temp_dir)
            if temp_path.exists() and temp_path.is_dir():
                # 一時ファイルを削除
                for file_path in temp_path.rglob('*'):
                    if file_path.is_file():
                        file_path.unlink()
                        self.logger.debug(f"一時ファイルを削除: {file_path}")

                # 空のディレクトリを削除
                try:
                    temp_path.rmdir()
                    self.logger.debug(f"一時ディレクトリを削除: {temp_dir}")
                except OSError:
                    # ディレクトリが空でない場合は削除しない
                    pass

            return True

        except Exception as e:
            self.logger.error(f"一時ファイルクリーンアップ中にエラーが発生しました: {e}")
            return False
