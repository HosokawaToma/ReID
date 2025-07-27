"""複数静止画処理アプリケーション"""
import os
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np

from config import APP_CONFIG, REID_CONFIG, FILE_PROCESSING_CONFIG
from managers.yolo_manager import YoloManager
from managers.reid_manager import ReIDManager
from managers.post_processing_manager import KReciprocalManager
from utilities.person_tracker import PersonTracker
from utilities.file_processor import FileProcessor
from utilities.result_saver import ResultSaver
from utilities.progress_tracker import ProgressTracker
from exceptions import (
    ProcessingResult, ProcessingSummary,
    ModelLoadError, ModelInferenceError, PersonIdentificationError
)


class ImageReIDApp:
    """複数静止画処理アプリケーション"""

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
        self.yolo_manager = None
        self.reid_manager = None
        self.person_tracker = None
        self.file_processor = FileProcessor()
        self.result_saver = ResultSaver()
        self.progress_tracker = ProgressTracker()

        # 処理結果
        self.processing_results: List[ProcessingResult] = []

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
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            self.logger.info(f"出力ディレクトリを準備しました: {self.output_dir}")
        except Exception as e:
            raise OSError(f"出力ディレクトリの作成に失敗しました: {e}")

    def _initialize_components(self) -> None:
        """コンポーネントの初期化"""
        try:
            self.logger.info("コンポーネントの初期化を開始します...")

            # YOLO管理クラスの初期化
            self.logger.info("YOLOモデルを初期化中...")
            self.yolo_manager = YoloManager()

            # ReID管理クラスの初期化
            self.logger.info(f"ReIDモデル ({self.reid_backend}) を初期化中...")
            self.reid_manager = ReIDManager(backend=self.reid_backend)

            # 人物追跡クラスの初期化
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

            self.logger.info("全てのコンポーネントの初期化が完了しました")

        except Exception as e:
            self.logger.error(f"コンポーネント初期化エラー: {e}")
            raise ModelLoadError(f"コンポーネントの初期化に失敗しました: {e}")

    def process_images(self) -> ProcessingSummary:
        """複数画像ファイルを処理

        Returns:
            処理結果のサマリー
        """
        self.logger.info("画像処理を開始します...")

        # 画像ファイルを検出
        image_extensions = self.file_processor.get_image_extensions()
        discovered_files = self.file_processor.discover_files(self.input_dir, image_extensions)

        if not discovered_files:
            self.logger.warning("処理対象の画像ファイルが見つかりませんでした")
            return self._create_empty_summary()

        # ファイルを検証
        valid_files = self.file_processor.validate_files(discovered_files)

        if not valid_files:
            self.logger.warning("有効な画像ファイルが見つかりませんでした")
            return self._create_empty_summary()

        # 出力構造を作成
        output_mapping = self.file_processor.create_output_structure(self.output_dir, valid_files)

        # 進捗追跡を開始
        self.progress_tracker.start_processing(len(valid_files))

        # 各画像ファイルを処理
        start_time = time.time()

        for i, image_path in enumerate(valid_files):
            try:
                output_path = output_mapping.get(image_path, "")
                result = self.process_single_image(image_path, output_path)
                self.processing_results.append(result)

                # 進捗更新
                self.progress_tracker.update_progress(
                    i + 1, f"処理完了: {Path(image_path).name}"
                )

            except Exception as e:
                self.logger.error(f"画像処理エラー ({image_path}): {e}")
                error_result = ProcessingResult(
                    file_path=image_path,
                    success=False,
                    persons_detected=0,
                    unique_person_ids=[],
                    processing_time=0.0,
                    error_message=str(e)
                )
                self.processing_results.append(error_result)

                # 進捗更新
                self.progress_tracker.update_progress(
                    i + 1, f"エラー: {Path(image_path).name}"
                )

        total_processing_time = time.time() - start_time

        # 処理サマリーを作成
        summary = self._create_processing_summary(total_processing_time)

        # 進捗追跡を完了
        self.progress_tracker.finish_processing(summary.__dict__)

        # 結果を保存
        self._save_results(summary)

        self.logger.info("画像処理が完了しました")
        return summary

    def process_single_image(self, image_path: str, output_path: str) -> ProcessingResult:
        """単一画像ファイルを処理

        Args:
            image_path: 入力画像ファイルパス
            output_path: 出力ファイルパス（拡張子なし）

        Returns:
            処理結果
        """
        start_time = time.time()

        try:
            # 画像を読み込み
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"画像の読み込みに失敗しました: {image_path}")

            self.logger.debug(f"画像を読み込みました: {image_path} (形状: {image.shape})")

            # YOLO推論で人物検出
            person_detections = self.yolo_manager.track_and_detect_persons(image)

            if not person_detections:
                self.logger.debug(f"人物が検出されませんでした: {image_path}")
                # 人物が検出されなくても元画像を保存
                self.result_saver.save_processed_image(image, output_path)
                processing_time = time.time() - start_time
                return ProcessingResult(
                    file_path=image_path,
                    success=True,
                    persons_detected=0,
                    unique_person_ids=[],
                    processing_time=processing_time
                )

            # 検出された人物を処理
            detected_person_ids = []
            processed_image = image.copy()

            for bounding_box, person_crop in person_detections:
                try:
                    # ReID特徴抽出
                    features = self.reid_manager.extract_features(
                        person_crop, camera_id=0, view_id=0
                    )

                    # 人物ID割り当て
                    person_id = self.person_tracker.assign_person_id(features, self.k_reciprocal_manager)
                    detected_person_ids.append(person_id)

                    # バウンディングボックスとIDを描画
                    self._draw_detection(processed_image, bounding_box, person_id)

                    self.logger.debug(f"人物ID {person_id} を割り当てました")

                except Exception as e:
                    self.logger.warning(f"人物処理エラー: {e}")
                    continue

            # 処理済み画像を保存
            self.result_saver.save_processed_image(processed_image, output_path)

            processing_time = time.time() - start_time

            result = ProcessingResult(
                file_path=image_path,
                success=True,
                persons_detected=len(person_detections),
                unique_person_ids=detected_person_ids,
                processing_time=processing_time
            )

            self.logger.debug(f"画像処理完了: {image_path} ({len(detected_person_ids)}人検出)")
            return result

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"画像処理エラー ({image_path}): {e}")
            return ProcessingResult(
                file_path=image_path,
                success=False,
                persons_detected=0,
                unique_person_ids=[],
                processing_time=processing_time,
                error_message=str(e)
            )

    def _draw_detection(self, image: np.ndarray, bounding_box: np.ndarray, person_id: int) -> None:
        """検出結果を画像に描画

        Args:
            image: 描画対象の画像
            bounding_box: バウンディングボックス [x1, y1, x2, y2]
            person_id: 人物ID
        """
        x1, y1, x2, y2 = bounding_box.astype(int)
        img_height, img_width = image.shape[:2]

        # 画像サイズに応じて文字サイズを調整（より小さめに）
        base_font_scale = min(img_width, img_height) / 1500.0
        font_scale = max(0.3, min(0.8, base_font_scale))

        # 線の太さも画像サイズに応じて調整
        line_thickness = max(1, int(min(img_width, img_height) / 800))
        font_thickness = max(1, int(font_scale * 3))

        # バウンディングボックスを描画
        color = (0, 255, 0)  # 緑色
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)

        # 人物IDラベルを描画
        label = f"ID: {person_id}"
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )

        # ラベル背景の余白（より小さく）
        padding = max(2, int(font_scale * 8))

        # ラベル背景のサイズ
        bg_width = label_width + padding * 2
        bg_height = label_height + padding * 2

        # ラベル位置を決定（優先順位: 上 -> 下 -> 右 -> 左）
        positions = [
            # 上側
            (x1, y1 - bg_height, x1 + bg_width, y1),
            # 下側
            (x1, y2, x1 + bg_width, y2 + bg_height),
            # 右側
            (x2, y1, x2 + bg_width, y1 + bg_height),
            # 左側
            (x1 - bg_width, y1, x1, y1 + bg_height)
        ]

        # 画像内に収まる位置を探す
        bg_x1, bg_y1, bg_x2, bg_y2 = None, None, None, None
        for pos_x1, pos_y1, pos_x2, pos_y2 in positions:
            if (0 <= pos_x1 < img_width and 0 <= pos_y1 < img_height and
                0 < pos_x2 <= img_width and 0 < pos_y2 <= img_height):
                bg_x1, bg_y1, bg_x2, bg_y2 = pos_x1, pos_y1, pos_x2, pos_y2
                break

        # どの位置も完全に収まらない場合は、画像境界内に調整
        if bg_x1 is None:
            # 基本位置（上側）から調整
            bg_x1 = max(0, min(x1, img_width - bg_width))
            bg_y1 = max(0, y1 - bg_height)
            bg_x2 = min(img_width, bg_x1 + bg_width)
            bg_y2 = min(img_height, bg_y1 + bg_height)

            # 高さが足りない場合は下に移動
            if bg_y2 - bg_y1 < bg_height:
                bg_y1 = max(0, min(y2, img_height - bg_height))
                bg_y2 = min(img_height, bg_y1 + bg_height)

        # ラベル背景を描画
        if bg_x2 > bg_x1 and bg_y2 > bg_y1:
            cv2.rectangle(image, (int(bg_x1), int(bg_y1)), (int(bg_x2), int(bg_y2)), color, -1)

            # テキスト位置を計算
            text_x = int(bg_x1 + padding)
            text_y = int(bg_y1 + padding + label_height)

            # テキストが画像境界内にあることを確認
            if (0 <= text_x < img_width - label_width and
                label_height <= text_y < img_height):
                cv2.putText(image, label, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    def _create_processing_summary(self, total_processing_time: float) -> ProcessingSummary:
        """処理サマリーを作成

        Args:
            total_processing_time: 総処理時間

        Returns:
            処理サマリー
        """
        successful_results = [r for r in self.processing_results if r.success]
        failed_results = [r for r in self.processing_results if not r.success]

        total_persons_detected = sum(r.persons_detected for r in successful_results)
        all_unique_ids = set()
        for result in successful_results:
            all_unique_ids.update(result.unique_person_ids)

        return ProcessingSummary(
            total_files=len(self.processing_results),
            successful_files=len(successful_results),
            failed_files=len(failed_results),
            total_persons_detected=total_persons_detected,
            unique_person_count=len(all_unique_ids),
            total_processing_time=total_processing_time,
            reid_backend=self.reid_backend,
            results=self.processing_results
        )

    def _create_empty_summary(self) -> ProcessingSummary:
        """空の処理サマリーを作成

        Returns:
            空の処理サマリー
        """
        return ProcessingSummary(
            total_files=0,
            successful_files=0,
            failed_files=0,
            total_persons_detected=0,
            unique_person_count=0,
            total_processing_time=0.0,
            reid_backend=self.reid_backend,
            results=[]
        )

    def _save_results(self, summary: ProcessingSummary) -> None:
        """処理結果を保存

        Args:
            summary: 処理サマリー
        """
        try:
            # サマリーをJSONで保存
            summary_dict = {
                'total_files': summary.total_files,
                'successful_files': summary.successful_files,
                'failed_files': summary.failed_files,
                'total_persons_detected': summary.total_persons_detected,
                'unique_person_count': summary.unique_person_count,
                'total_processing_time': summary.total_processing_time,
                'reid_backend': summary.reid_backend,
                'results': [
                    {
                        'file_path': r.file_path,
                        'success': r.success,
                        'persons_detected': r.persons_detected,
                        'unique_person_ids': r.unique_person_ids,
                        'processing_time': r.processing_time,
                        'error_message': r.error_message
                    }
                    for r in summary.results
                ]
            }

            self.result_saver.save_results_summary(summary_dict, self.output_dir)
            self.logger.info("処理結果を保存しました")

        except Exception as e:
            self.logger.error(f"結果保存エラー: {e}")


def main():
    """メイン関数"""
    # ログ設定
    logging.basicConfig(
        level=getattr(logging, APP_CONFIG.log_level),
        format=APP_CONFIG.log_format
    )

    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='複数静止画処理アプリケーション')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='入力ディレクトリパス')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='出力ディレクトリパス')
    parser.add_argument('--reid_backend', type=str, default='clip',
                       choices=['clip', 'trans_reid', 'la_transformer'],
                       help='使用するReIDバックエンド (デフォルト: clip)')

    args = parser.parse_args()

    try:
        # アプリケーションを初期化
        app = ImageReIDApp(
            reid_backend=args.reid_backend,
            input_dir=args.input_dir,
            output_dir=args.output_dir
        )

        # 画像処理を実行
        summary = app.process_images()

        # 結果を表示
        print(f"\n=== 処理完了 ===")
        print(f"総ファイル数: {summary.total_files}")
        print(f"成功: {summary.successful_files}")
        print(f"失敗: {summary.failed_files}")
        print(f"検出された人物総数: {summary.total_persons_detected}")
        print(f"ユニーク人物数: {summary.unique_person_count}")
        print(f"使用ReIDバックエンド: {summary.reid_backend}")
        print(f"総処理時間: {summary.total_processing_time:.2f}秒")

        if summary.failed_files > 0:
            print(f"\n失敗したファイル:")
            for result in summary.results:
                if not result.success:
                    print(f"  - {result.file_path}: {result.error_message}")

    except Exception as e:
        logging.error(f"アプリケーション実行エラー: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
