"""複数動画ファイル処理アプリケーション"""
import os
import cv2
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from config import APP_CONFIG, REID_CONFIG, COLOR_PALETTE
from utilities.file_processor import FileProcessor
from utilities.progress_tracker import ProgressTracker
from utilities.result_saver import ResultSaver
from exceptions import ProcessingResult, ProcessingSummary, ModelLoadError, ModelInferenceError
from managers.yolo_manager import YoloManager
from managers.reid_manager import ReIDManager
from managers.post_processing_manager import KReciprocalManager
from utilities.person_tracker import PersonTracker


class VideoReIDApp:
    """複数動画ファイル処理アプリケーション"""

    def __init__(self, reid_backend: str, input_dir: str, output_dir: str):
        """初期化

        Args:
            reid_backend: 使用するReIDバックエンド ('clip', 'trans_reid', 'la_transformer')
            input_dir: 入力ディレクトリパス
            output_dir: 出力ディレクトリパス
        """
        self.reid_backend = reid_backend
        self.input_dir = input_dir
        self.output_dir = output_dir

        # ログ設定
        self.logger = logging.getLogger(__name__)

        # コンポーネント初期化
        self.file_processor = FileProcessor()
        self.progress_tracker = ProgressTracker()
        self.result_saver = ResultSaver()

        # AI モデル管理
        self.yolo_manager = None
        self.reid_manager = None
        self.person_tracker = None

        # 処理統計
        self.processing_stats = {
            'total_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'total_persons_detected': 0,
            'unique_person_count': 0,
            'total_processing_time': 0.0,
            'results': []
        }

        # 初期化と検証
        self._validate_directories()
        self._initialize_components()

    def _validate_directories(self) -> None:
        """ディレクトリの存在確認と作成"""
        # 入力ディレクトリの確認
        if not os.path.exists(self.input_dir):
            raise FileNotFoundError(f"入力ディレクトリが存在しません: {self.input_dir}")

        if not os.path.isdir(self.input_dir):
            raise NotADirectoryError(f"指定されたパスはディレクトリではありません: {self.input_dir}")

        # 出力ディレクトリの作成
        self.result_saver.create_output_directories(self.output_dir)

        self.logger.info(f"ディレクトリ検証完了 - 入力: {self.input_dir}, 出力: {self.output_dir}")

    def _initialize_components(self) -> None:
        """AIモデルとコンポーネントの初期化"""
        try:
            self.logger.info("AIモデルの初期化を開始...")

            # YOLO管理の初期化
            self.yolo_manager = YoloManager()

            # ReID管理の初期化
            self.reid_manager = ReIDManager(backend=self.reid_backend)

            # 人物追跡の初期化
            self.person_tracker = PersonTracker(self.reid_backend, REID_CONFIG)

            # K-Reciprocal Re-ranking管理の初期化
            if REID_CONFIG.use_re_ranking:
                self.k_reciprocal_manager = KReciprocalManager(
                    k1=REID_CONFIG.rerank_k1,
                    k2=REID_CONFIG.rerank_k2,
                    lambda_value=REID_CONFIG.rerank_lambda
                )
                self.logger.info("K-Reciprocal Re-ranking管理を初期化しました")
            else:
                self.k_reciprocal_manager = None

            self.logger.info(f"AIモデル初期化完了 - ReIDバックエンド: {self.reid_backend}")

        except Exception as e:
            self.logger.error(f"AIモデル初期化に失敗しました: {e}")
            raise ModelLoadError(f"AIモデル初期化に失敗しました: {e}")

    def process_videos(self) -> Dict[str, Any]:
        """複数動画ファイルを処理

        Returns:
            処理結果のサマリー辞書
        """
        start_time = time.time()

        try:
            # 動画ファイルを検出
            video_extensions = self.file_processor.get_video_extensions()
            discovered_files = self.file_processor.discover_files(self.input_dir, video_extensions)

            if not discovered_files:
                self.logger.warning(f"処理対象の動画ファイルが見つかりませんでした: {self.input_dir}")
                return self._create_empty_summary()

            # ファイル検証
            valid_files = self.file_processor.validate_files(discovered_files)

            if not valid_files:
                self.logger.warning("有効な動画ファイルが見つかりませんでした")
                return self._create_empty_summary()

            # 出力パス構造を作成
            output_mapping = self.file_processor.create_output_structure(self.output_dir, valid_files)

            # 進捗追跡開始
            self.progress_tracker.start_processing(len(valid_files))

            # 各動画ファイルを処理
            for i, video_path in enumerate(valid_files):
                try:
                    self.logger.info(f"処理開始 ({i+1}/{len(valid_files)}): {Path(video_path).name}")

                    output_path = output_mapping.get(video_path, "")
                    if not output_path:
                        raise ValueError(f"出力パスが設定されていません: {video_path}")

                    result = self.process_single_video(video_path, output_path)
                    self.processing_stats['results'].append(result)

                    if result.success:
                        self.processing_stats['successful_files'] += 1
                        self.processing_stats['total_persons_detected'] += result.persons_detected

                        # 成功時の詳細ログ
                        self.logger.info(f"処理成功 ({i+1}/{len(valid_files)}): {Path(video_path).name} - "
                                       f"{result.persons_detected}人検出, {len(result.unique_person_ids)}人ユニーク")
                    else:
                        self.processing_stats['failed_files'] += 1
                        self.logger.warning(f"処理失敗 ({i+1}/{len(valid_files)}): {Path(video_path).name} - {result.error_message}")

                    # 進捗更新
                    status_msg = "処理完了" if result.success else "エラー"
                    self.progress_tracker.update_progress(i + 1, f"{status_msg}: {Path(video_path).name}")

                except KeyboardInterrupt:
                    self.logger.warning("ユーザーによって処理が中断されました")
                    # 中断時も結果を記録
                    error_result = ProcessingResult(
                        file_path=video_path,
                        success=False,
                        persons_detected=0,
                        unique_person_ids=[],
                        processing_time=0.0,
                        error_message="ユーザーによって中断されました"
                    )
                    self.processing_stats['results'].append(error_result)
                    self.processing_stats['failed_files'] += 1
                    raise  # KeyboardInterruptを再発生させる

                except Exception as e:
                    self.logger.error(f"動画処理中に予期しないエラーが発生しました ({Path(video_path).name}): {e}")

                    # エラーの詳細をログに記録
                    import traceback
                    self.logger.debug(f"エラーの詳細:\n{traceback.format_exc()}")

                    error_result = ProcessingResult(
                        file_path=video_path,
                        success=False,
                        persons_detected=0,
                        unique_person_ids=[],
                        processing_time=0.0,
                        error_message=str(e)
                    )
                    self.processing_stats['results'].append(error_result)
                    self.processing_stats['failed_files'] += 1

                    # 進捗更新
                    self.progress_tracker.update_progress(i + 1, f"エラー: {Path(video_path).name}")

                    # 連続エラーが多い場合は警告
                    if self.processing_stats['failed_files'] >= 3 and i < 5:
                        self.logger.warning("連続してエラーが発生しています。入力ファイルや設定を確認してください。")

            # 統計情報を更新
            self.processing_stats['total_files'] = len(valid_files)
            self.processing_stats['unique_person_count'] = len(self.person_tracker.person_database)
            self.processing_stats['total_processing_time'] = time.time() - start_time

            # 処理サマリーを作成
            summary = ProcessingSummary(
                total_files=self.processing_stats['total_files'],
                successful_files=self.processing_stats['successful_files'],
                failed_files=self.processing_stats['failed_files'],
                total_persons_detected=self.processing_stats['total_persons_detected'],
                unique_person_count=self.processing_stats['unique_person_count'],
                total_processing_time=self.processing_stats['total_processing_time'],
                reid_backend=self.reid_backend,
                results=self.processing_stats['results']
            )

            # 結果を保存
            summary_dict = self._summary_to_dict(summary)
            self.result_saver.save_results_summary(summary_dict, output_mapping.get(valid_files[0], ""))

            # 進捗完了
            self.progress_tracker.finish_processing(summary_dict)

            return summary_dict

        except KeyboardInterrupt:
            self.logger.warning("処理がユーザーによって中断されました")
            # 中断時でも部分的な結果を返す
            self.processing_stats['total_files'] = len(valid_files) if 'valid_files' in locals() else 0
            self.processing_stats['unique_person_count'] = len(self.person_tracker.person_database) if self.person_tracker else 0
            self.processing_stats['total_processing_time'] = time.time() - start_time

            summary_dict = self._create_partial_summary()
            self.logger.info("中断時の部分的な結果を保存しています...")

            try:
                if 'output_mapping' in locals() and 'valid_files' in locals() and valid_files:
                    self.result_saver.save_results_summary(summary_dict, output_mapping.get(valid_files[0], ""))
            except Exception as save_error:
                self.logger.error(f"中断時の結果保存に失敗しました: {save_error}")

            raise

        except Exception as e:
            self.logger.error(f"動画処理中に予期しないエラーが発生しました: {e}")

            # エラーの詳細をログに記録
            import traceback
            self.logger.debug(f"エラーの詳細:\n{traceback.format_exc()}")

            # エラー時でも部分的な結果を返す
            try:
                self.processing_stats['total_files'] = len(valid_files) if 'valid_files' in locals() else 0
                self.processing_stats['unique_person_count'] = len(self.person_tracker.person_database) if self.person_tracker else 0
                self.processing_stats['total_processing_time'] = time.time() - start_time

                return self._create_partial_summary()
            except Exception as summary_error:
                self.logger.error(f"エラー時のサマリー作成に失敗しました: {summary_error}")
                return self._create_empty_summary()

    def process_single_video(self, video_path: str, output_path: str) -> ProcessingResult:
        """単一動画ファイルを処理

        Args:
            video_path: 動画ファイルパス
            output_path: 出力ファイルパス（拡張子なし）

        Returns:
            処理結果
        """
        start_time = time.time()
        processed_frames = []
        persons_detected = 0
        unique_person_ids = set()
        cap = None
        error_frames = 0

        try:
            self.logger.info(f"動画処理開始: {video_path}")

            # ファイルアクセス権限チェック
            if not os.access(video_path, os.R_OK):
                raise PermissionError(f"動画ファイルの読み込み権限がありません: {video_path}")

            # ファイルサイズチェック
            file_size = os.path.getsize(video_path)
            if file_size == 0:
                raise ValueError(f"動画ファイルが空です: {video_path}")

            self.logger.debug(f"動画ファイルサイズ: {file_size / (1024*1024):.2f} MB")

            # 動画を開く
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"動画ファイルを開けませんでした: {video_path}")

            # 動画情報を取得
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 動画情報の妥当性チェック
            if fps <= 0 or total_frames <= 0 or width <= 0 or height <= 0:
                self.logger.warning(f"動画情報が不正です - FPS: {fps}, フレーム数: {total_frames}, サイズ: {width}x{height}")
                # デフォルト値を設定
                fps = fps if fps > 0 else 30.0

            self.logger.info(f"動画情報 - FPS: {fps}, 総フレーム数: {total_frames}, サイズ: {width}x{height}")

            frame_count = 0
            consecutive_errors = 0
            max_consecutive_errors = 10  # 連続エラーの上限
            target_size = None  # フレームサイズの統一用

            while True:
                ret, frame = cap.read()
                if not ret:
                    if frame_count == 0:
                        raise ValueError("動画から最初のフレームを読み込めませんでした")
                    break

                try:
                    # フレーム妥当性チェック
                    if frame is None or frame.size == 0:
                        raise ValueError(f"無効なフレーム (フレーム番号: {frame_count})")

                    # 最初のフレームでターゲットサイズを設定
                    if target_size is None:
                        target_size = (frame.shape[1], frame.shape[0])  # (width, height)
                        self.logger.debug(f"ターゲットフレームサイズ: {target_size}")

                    # フレームサイズの統一
                    current_size = (frame.shape[1], frame.shape[0])
                    if current_size != target_size:
                        self.logger.debug(f"フレーム {frame_count} をリサイズ: {current_size} -> {target_size}")
                        frame = cv2.resize(frame, target_size)

                    # フレーム処理
                    processed_frame, frame_persons = self._process_frame(frame, frame_count)

                    # 処理後のフレームサイズも確認
                    if processed_frame.shape[1::-1] != target_size:
                        processed_frame = cv2.resize(processed_frame, target_size)

                    processed_frames.append(processed_frame)

                    persons_detected += len(frame_persons)
                    for person_id in frame_persons:
                        unique_person_ids.add(person_id)

                    frame_count += 1
                    consecutive_errors = 0  # エラーカウンタリセット

                    # 進捗ログ（一定間隔で）
                    if frame_count % 100 == 0:
                        self.logger.debug(f"フレーム処理進捗: {frame_count}/{total_frames}")

                except Exception as e:
                    error_frames += 1
                    consecutive_errors += 1

                    self.logger.warning(f"フレーム {frame_count} の処理中にエラー: {e}")

                    # 連続エラーが多すぎる場合は処理を中断
                    if consecutive_errors >= max_consecutive_errors:
                        raise RuntimeError(f"連続して{max_consecutive_errors}フレームの処理に失敗しました。動画が破損している可能性があります。")

                    # エラーが発生したフレームは元のフレームを使用（サイズ統一）
                    if target_size and frame is not None:
                        if frame.shape[1::-1] != target_size:
                            frame = cv2.resize(frame, target_size)
                        processed_frames.append(frame)
                    else:
                        # フレームが完全に無効な場合は前のフレームを複製
                        if processed_frames:
                            processed_frames.append(processed_frames[-1].copy())
                        else:
                            # 最初のフレームでエラーの場合は黒いフレームを作成
                            if target_size:
                                black_frame = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
                                processed_frames.append(black_frame)

                    frame_count += 1
                    continue

            # リソース解放
            if cap is not None:
                cap.release()
                cap = None

            # 処理結果の検証
            if not processed_frames:
                raise ValueError("処理可能なフレームがありませんでした")

            self.logger.info(f"フレーム処理完了 - 総フレーム数: {frame_count}, エラーフレーム数: {error_frames}")

            # 処理済み動画を保存
            try:
                self.logger.info(f"動画保存開始: {len(processed_frames)}フレーム, FPS: {fps}")

                # フレーム情報の詳細ログ
                if processed_frames:
                    first_frame = processed_frames[0]
                    self.logger.debug(f"フレーム情報 - サイズ: {first_frame.shape}, データ型: {first_frame.dtype}")

                save_success = self.result_saver.save_processed_video(processed_frames, output_path, fps)
                if not save_success:
                    raise RuntimeError("動画の保存に失敗しました")

                # 保存された動画ファイルの検証
                output_file = f"{output_path}.mp4"
                if os.path.exists(output_file):
                    file_size = os.path.getsize(output_file)
                    self.logger.info(f"動画保存完了: {output_file} ({file_size/1024/1024:.2f}MB)")

                    # 保存された動画の読み込みテスト
                    test_cap = cv2.VideoCapture(output_file)
                    if test_cap.isOpened():
                        test_ret, test_frame = test_cap.read()
                        test_cap.release()
                        if test_ret and test_frame is not None:
                            self.logger.debug("保存された動画の読み込みテスト: 成功")
                        else:
                            self.logger.warning("保存された動画の読み込みテスト: 失敗 - フレーム読み込み不可")
                    else:
                        self.logger.warning("保存された動画の読み込みテスト: 失敗 - ファイルを開けません")
                else:
                    raise RuntimeError(f"動画ファイルが作成されませんでした: {output_file}")

            except Exception as e:
                self.logger.error(f"動画保存エラー: {e}")
                # エラーの詳細情報を追加
                import traceback
                self.logger.debug(f"動画保存エラーの詳細:\n{traceback.format_exc()}")
                raise RuntimeError(f"動画の保存に失敗しました: {e}")

            processing_time = time.time() - start_time

            result = ProcessingResult(
                file_path=video_path,
                success=True,
                persons_detected=persons_detected,
                unique_person_ids=list(unique_person_ids),
                processing_time=processing_time,
                error_message=None
            )

            self.logger.info(f"動画処理完了: {video_path} ({processing_time:.2f}秒, {persons_detected}人検出, {len(unique_person_ids)}人ユニーク)")
            return result

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"動画処理エラー ({Path(video_path).name}): {e}"
            self.logger.error(error_msg)

            # 詳細なエラー情報をログに記録
            if isinstance(e, PermissionError):
                self.logger.error("ファイルアクセス権限を確認してください")
            elif isinstance(e, ValueError) and "開けませんでした" in str(e):
                self.logger.error("動画ファイルが破損しているか、サポートされていない形式の可能性があります")
            elif isinstance(e, RuntimeError) and "連続して" in str(e):
                self.logger.error("動画の品質に問題がある可能性があります")

            return ProcessingResult(
                file_path=video_path,
                success=False,
                persons_detected=0,
                unique_person_ids=[],
                processing_time=processing_time,
                error_message=str(e)
            )

        finally:
            # リソースの確実な解放
            if cap is not None:
                try:
                    cap.release()
                except Exception as e:
                    self.logger.warning(f"VideoCapture解放エラー: {e}")

    def _process_frame(self, frame: np.ndarray, frame_number: int) -> Tuple[np.ndarray, List[int]]:
        """単一フレームを処理

        Args:
            frame: 入力フレーム
            frame_number: フレーム番号

        Returns:
            (処理済みフレーム, 検出された人物IDのリスト)
        """
        try:
            # YOLO推論で人物検出
            person_detections = self.yolo_manager.track_and_detect_persons(frame)

            if not person_detections:
                return frame, []

            detected_person_ids = []
            processed_frame = frame.copy()

            # 各検出された人物を処理
            for bounding_box, person_crop in person_detections:
                try:
                    # ReID特徴抽出
                    features = self.reid_manager.extract_features(person_crop, camera_id=0)

                    # 人物ID割り当て
                    person_id = self.person_tracker.assign_person_id(features, self.k_reciprocal_manager)
                    detected_person_ids.append(person_id)

                    # バウンディングボックスとIDを描画
                    processed_frame = self._draw_detection(processed_frame, bounding_box, person_id)

                except Exception as e:
                    self.logger.warning(f"人物処理エラー (フレーム {frame_number}): {e}")
                    continue

            return processed_frame, detected_person_ids

        except Exception as e:
            self.logger.error(f"フレーム処理エラー (フレーム {frame_number}): {e}")
            raise ModelInferenceError(f"フレーム処理に失敗しました: {e}")

    def _draw_detection(self, frame: np.ndarray, bounding_box: np.ndarray, person_id: int) -> np.ndarray:
        """検出結果をフレームに描画

        Args:
            frame: 描画対象のフレーム
            bounding_box: バウンディングボックス [x1, y1, x2, y2]
            person_id: 人物ID

        Returns:
            描画済みフレーム
        """
        x1, y1, x2, y2 = bounding_box.astype(int)

        # 色を選択
        color_index = (person_id - 1) % len(COLOR_PALETTE.colors)
        color = COLOR_PALETTE.colors[color_index]

        # バウンディングボックスを描画
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # IDラベルを描画
        label = f"ID: {person_id}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

        # ラベル背景を描画
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0], y1), color, -1)

        # ラベルテキストを描画
        cv2.putText(frame, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def _create_empty_summary(self) -> Dict[str, Any]:
        """空の処理サマリーを作成"""
        return {
            'total_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'total_persons_detected': 0,
            'unique_person_count': 0,
            'total_processing_time': 0.0,
            'reid_backend': self.reid_backend,
            'results': []
        }

    def _create_partial_summary(self) -> Dict[str, Any]:
        """部分的な処理サマリーを作成（中断時やエラー時用）"""
        return {
            'total_files': self.processing_stats['total_files'],
            'successful_files': self.processing_stats['successful_files'],
            'failed_files': self.processing_stats['failed_files'],
            'total_persons_detected': self.processing_stats['total_persons_detected'],
            'unique_person_count': self.processing_stats['unique_person_count'],
            'total_processing_time': self.processing_stats['total_processing_time'],
            'reid_backend': self.reid_backend,
            'results': self.processing_stats['results'],
            'status': 'partial'  # 部分的な結果であることを示す
        }

    def _summary_to_dict(self, summary: ProcessingSummary) -> Dict[str, Any]:
        """ProcessingSummaryを辞書に変換"""
        return {
            'total_files': summary.total_files,
            'successful_files': summary.successful_files,
            'failed_files': summary.failed_files,
            'total_persons_detected': summary.total_persons_detected,
            'unique_person_count': summary.unique_person_count,
            'total_processing_time': summary.total_processing_time,
            'reid_backend': summary.reid_backend,
            'results': [
                {
                    'file_path': result.file_path,
                    'success': result.success,
                    'persons_detected': result.persons_detected,
                    'unique_person_ids': result.unique_person_ids,
                    'processing_time': result.processing_time,
                    'error_message': result.error_message
                }
                for result in summary.results
            ]
        }


def main():
    """メイン関数 - コマンドライン実行用"""
    import argparse
    import sys

    # ログ設定
    logging.basicConfig(
        level=getattr(logging, APP_CONFIG.log_level),
        format=APP_CONFIG.log_format
    )

    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(
        description='複数動画ファイルの人物再識別処理',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python src/video_app.py --input_dir ./videos --output_dir ./output --reid_backend clip
  python src/video_app.py --input_dir ./videos --output_dir ./output --reid_backend trans_reid
  python src/video_app.py --input_dir ./videos --output_dir ./output --reid_backend la_transformer

サポートされる動画形式:
  MP4, AVI, MOV

ReIDバックエンド:
  clip          - CLIP-ReID (デフォルト)
  trans_reid    - TransReID
  la_transformer - LA-Transformer
        """
    )

    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='入力動画ファイルが格納されているディレクトリパス'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='処理結果を保存するディレクトリパス'
    )

    parser.add_argument(
        '--reid_backend',
        type=str,
        choices=['clip', 'trans_reid', 'la_transformer'],
        default='clip',
        help='使用するReIDバックエンド (デフォルト: clip)'
    )

    parser.add_argument(
        '--log_level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='ログレベル (デフォルト: INFO)'
    )

    try:
        args = parser.parse_args()

        # ログレベルを更新
        if args.log_level != 'INFO':
            logging.getLogger().setLevel(getattr(logging, args.log_level))

        # 引数検証
        if not os.path.exists(args.input_dir):
            print(f"エラー: 入力ディレクトリが存在しません: {args.input_dir}", file=sys.stderr)
            sys.exit(1)

        if not os.path.isdir(args.input_dir):
            print(f"エラー: 指定されたパスはディレクトリではありません: {args.input_dir}", file=sys.stderr)
            sys.exit(1)

        # 出力ディレクトリの親ディレクトリが存在するかチェック
        output_parent = Path(args.output_dir).parent
        if not output_parent.exists():
            print(f"エラー: 出力ディレクトリの親ディレクトリが存在しません: {output_parent}", file=sys.stderr)
            sys.exit(1)

        print(f"=== 動画処理アプリケーション ===")
        print(f"入力ディレクトリ: {args.input_dir}")
        print(f"出力ディレクトリ: {args.output_dir}")
        print(f"ReIDバックエンド: {args.reid_backend}")
        print(f"ログレベル: {args.log_level}")
        print()

        # アプリケーション実行
        app = VideoReIDApp(
            reid_backend=args.reid_backend,
            input_dir=args.input_dir,
            output_dir=args.output_dir
        )

        # 動画処理実行
        summary = app.process_videos()

        # 結果表示
        print(f"\n=== 処理完了 ===")
        print(f"処理結果は以下に保存されました: {args.output_dir}")

        # 終了コード設定
        if summary['failed_files'] > 0:
            print(f"警告: {summary['failed_files']}個のファイルの処理に失敗しました")
            sys.exit(2)  # 部分的な失敗
        else:
            sys.exit(0)  # 成功

    except KeyboardInterrupt:
        print("\n処理が中断されました", file=sys.stderr)
        sys.exit(130)

    except Exception as e:
        print(f"予期しないエラーが発生しました: {e}", file=sys.stderr)
        logging.getLogger(__name__).exception("予期しないエラー")
        sys.exit(1)


if __name__ == "__main__":
    main()
