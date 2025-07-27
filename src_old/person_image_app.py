"""複数人物画像直接処理アプリケーション"""
import os
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np

from config import APP_CONFIG, REID_CONFIG, FILE_PROCESSING_CONFIG
from managers.reid_manager import ReIDManager
from managers.post_processing_manager import KReciprocalManager
from utilities.person_tracker import PersonTracker
from utilities.file_processor import FileProcessor
from utilities.result_saver import ResultSaver
from utilities.progress_tracker import ProgressTracker
from utilities.reid_evaluator import ReIDEvaluator
from exceptions import (
    ProcessingResult, ProcessingSummary,
    ModelLoadError, ModelInferenceError, PersonIdentificationError
)


class PersonImageReIDApp:
    """複数人物画像直接処理アプリケーション"""

    def __init__(self, reid_backend: str, input_dir: str, output_dir: str, enable_evaluation: bool = True):
        """初期化

        Args:
            reid_backend: 使用するReIDバックエンド ('clip', 'trans_reid', 'la_transformer')
            input_dir: 入力ディレクトリパス
            output_dir: 出力ディレクトリパス
            enable_evaluation: 評価機能を有効にするかどうか
        """
        self.reid_backend = reid_backend
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.enable_evaluation = enable_evaluation

        # ログ設定
        self.logger = logging.getLogger(__name__)

        # 処理結果
        self.processing_results: List[ProcessingResult] = []
        self.person_features: Dict[str, Any] = {}  # 評価用の特徴量保存

        # ディレクトリの確認
        self._validate_directories()

        # コンポーネント初期化
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

            # ファイル処理クラスの初期化
            self.file_processor = FileProcessor()

            #
            self.result_saver = ResultSaver()

            #
            self.progress_tracker = ProgressTracker()

            #
            self.reid_evaluator = ReIDEvaluator(
                max_rank=REID_CONFIG.evaluation_max_rank,
                metric=REID_CONFIG.evaluation_metric,
                normalize=REID_CONFIG.evaluation_normalize,
                rerank=REID_CONFIG.evaluation_rerank,
                k1=REID_CONFIG.evaluation_k1,
                k2=REID_CONFIG.evaluation_k2,
                lambda_value=REID_CONFIG.evaluation_lambda
            )

            # ReID管理クラスの初期化（YOLO管理は不要）
            self.logger.info(f"ReIDモデル ({self.reid_backend}) を初期化中...")
            self.reid_manager = ReIDManager(backend=self.reid_backend)

            # 人物追跡クラスの初期化
            self.person_tracker = PersonTracker(self.reid_backend, REID_CONFIG)

            # K-Reciprocal Re-ranking管理の初期化
            self.k_reciprocal_manager = KReciprocalManager(
                k1=REID_CONFIG.rerank_k1,
                k2=REID_CONFIG.rerank_k2,
                lambda_value=REID_CONFIG.rerank_lambda
            )
            self.logger.info("K-Reciprocal Re-ranking管理を初期化しました")

            self.logger.info("全てのコンポーネントの初期化が完了しました")

        except Exception as e:
            self.logger.error(f"コンポーネント初期化エラー: {e}")
            raise ModelLoadError(f"コンポーネントの初期化に失敗しました: {e}")

    def process_person_images(self) -> ProcessingSummary:
        """複数人物画像ファイルを処理

        Returns:
            処理結果のサマリー
        """
        self.logger.info("人物画像処理を開始します...")

        # queryとgalleryディレクトリを検出
        query_dir = Path(self.input_dir) / "query"
        gallery_dir = Path(self.input_dir) / "gallery"

        # ディレクトリ構造をチェック
        if query_dir.exists() and gallery_dir.exists():
            self.logger.info("query/galleryディレクトリ構造を検出しました")
            valid_files = self._discover_query_gallery_files(query_dir, gallery_dir)
        else:
            self.logger.info("通常のディレクトリ構造で処理します")
            # 従来の処理
            image_extensions = self.file_processor.get_image_extensions()
            discovered_files = self.file_processor.discover_files(self.input_dir, image_extensions)
            valid_files = self.file_processor.validate_files(discovered_files) if discovered_files else []

        if not valid_files:
            self.logger.warning("処理対象の人物画像ファイルが見つかりませんでした")
            return self._create_empty_summary()

        # 出力構造を作成
        output_mapping = self.file_processor.create_output_structure(self.output_dir, valid_files)

        # 進捗追跡を開始
        self.progress_tracker.start_processing(len(valid_files))

        # 各人物画像ファイルを処理
        start_time = time.time()

        for i, image_path in enumerate(valid_files):
            try:
                output_path = output_mapping.get(image_path, "")
                result = self.process_single_person_image(image_path, output_path)
                self.processing_results.append(result)

                # 進捗更新
                self.progress_tracker.update_progress(
                    i + 1, f"処理完了: {Path(image_path).name}"
                )

            except Exception as e:
                self.logger.error(f"人物画像処理エラー ({image_path}): {e}")
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

        # 評価を実行
        if self.enable_evaluation and len(self.person_features) > 1:
            self.logger.info("ReID評価を実行中...")
            evaluation_results = self._perform_evaluation()
            summary.evaluation_results = evaluation_results

        self.logger.info("人物画像処理が完了しました")
        return summary

    def _discover_query_gallery_files(self, query_dir: Path, gallery_dir: Path) -> List[str]:
        """queryとgalleryディレクトリから画像ファイルを検出

        Args:
            query_dir: queryディレクトリパス
            gallery_dir: galleryディレクトリパス

        Returns:
            有効な画像ファイルパスのリスト
        """
        image_extensions = self.file_processor.get_image_extensions()
        valid_files = []

        # queryディレクトリから画像を検出
        if query_dir.exists():
            query_files = self.file_processor.discover_files(str(query_dir), image_extensions)
            query_valid = self.file_processor.validate_files(query_files) if query_files else []
            valid_files.extend(query_valid)
            self.logger.info(f"Query画像: {len(query_valid)}枚")

        # galleryディレクトリから画像を検出
        if gallery_dir.exists():
            gallery_files = self.file_processor.discover_files(str(gallery_dir), image_extensions)
            gallery_valid = self.file_processor.validate_files(gallery_files) if gallery_files else []
            valid_files.extend(gallery_valid)
            self.logger.info(f"Gallery画像: {len(gallery_valid)}枚")

        return valid_files

    def process_single_person_image(self, image_path: str, output_path: str) -> ProcessingResult:
        """単一人物画像ファイルを処理

        Args:
            image_path: 入力人物画像ファイルパス
            output_path: 出力ファイルパス（拡張子なし）

        Returns:
            処理結果
        """
        start_time = time.time()

        try:
            # 人物画像を読み込み
            person_image = cv2.imread(image_path)
            if person_image is None:
                raise ValueError(f"人物画像の読み込みに失敗しました: {image_path}")

            self.logger.debug(f"人物画像を読み込みました: {image_path} (形状: {person_image.shape})")

            # ReID特徴抽出（YOLO検出をスキップ）
            features = self.reid_manager.extract_features(
                person_image, camera_id=0, view_id=0
            )

            # 人物ID割り当て
            person_id = self.person_tracker.assign_person_id(features, self.k_reciprocal_manager)

            # 評価用に特徴量を保存
            if self.enable_evaluation:
                self._store_features_for_evaluation(image_path, features, person_id)

            # 処理済み画像を作成（IDラベルを描画）
            processed_image = person_image.copy()
            self._draw_person_id(processed_image, person_id)

            # 処理済み画像を保存
            self.result_saver.save_processed_image(processed_image, output_path)

            processing_time = time.time() - start_time

            result = ProcessingResult(
                file_path=image_path,
                success=True,
                persons_detected=1,  # 人物画像なので常に1人
                unique_person_ids=[person_id],
                processing_time=processing_time
            )

            self.logger.debug(f"人物画像処理完了: {image_path} (ID: {person_id})")
            return result

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"人物画像処理エラー ({image_path}): {e}")
            return ProcessingResult(
                file_path=image_path,
                success=False,
                persons_detected=0,
                unique_person_ids=[],
                processing_time=processing_time,
                error_message=str(e)
            )

    def _draw_person_id(self, image: np.ndarray, person_id: int) -> None:
        """人物IDを画像に描画

        Args:
            image: 描画対象の画像
            person_id: 人物ID
        """
        img_height, img_width = image.shape[:2]

        # 画像サイズに応じて文字サイズを調整
        base_font_scale = min(img_width, img_height) / 800.0
        font_scale = max(0.4, min(1.5, base_font_scale))

        # 線の太さも画像サイズに応じて調整
        thickness = max(1, int(font_scale * 2))

        # 画像の上部にIDラベルを描画
        label = f"ID: {person_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 255, 0)  # 緑色

        # ラベルサイズを取得
        (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # 余白を画像サイズに応じて調整（より小さく）
        padding = max(3, int(font_scale * 5))

        # 背景矩形のサイズと位置を計算
        bg_width = label_width + padding * 2
        bg_height = label_height + padding * 2

        # 背景矩形の位置を決定（左上角から開始）
        bg_x1 = 5  # 固定の小さな余白
        bg_y1 = 5  # 固定の小さな余白

        # 背景矩形が画像からはみ出さないように調整
        bg_x2 = min(img_width - 5, bg_x1 + bg_width)
        bg_y2 = min(img_height - 5, bg_y1 + bg_height)

        # 実際の背景サイズに合わせて調整
        actual_bg_width = bg_x2 - bg_x1
        actual_bg_height = bg_y2 - bg_y1

        # 背景矩形を描画
        if actual_bg_width > 0 and actual_bg_height > 0:
            cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)

            # テキスト位置を計算（背景矩形内に確実に収まるように）
            text_x = bg_x1 + padding
            text_y = bg_y1 + padding + label_height

            # 背景矩形内に収まらない場合は位置を調整
            if text_x + label_width > bg_x2:
                text_x = max(bg_x1, bg_x2 - label_width)
            if text_y > bg_y2:
                text_y = bg_y2 - 3

            # テキストを描画（最小限の境界チェック）
            if text_x >= 0 and text_y > 0 and text_x < img_width and text_y < img_height:
                cv2.putText(image, label, (text_x, text_y),
                           font, font_scale, color, thickness)

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

    def _store_features_for_evaluation(self, image_path: str, features: np.ndarray, person_id: int) -> None:
        """評価用に特徴量を保存

        Args:
            image_path: 画像ファイルパス
            features: 特徴量
            person_id: 人物ID
        """
        # ファイル名から人物IDを推定（ファイル名にIDが含まれている場合）
        filename = Path(image_path).stem

        # ファイル名から真のIDを抽出（例: "person_001_cam1.jpg" -> 1）
        true_person_id = self._extract_true_person_id(filename)

        # カメラIDを抽出（例: "person_001_cam1.jpg" -> 1）
        camera_id = self._extract_camera_id(filename)

        # query/galleryの判定
        is_query = "query" in Path(image_path).parts
        is_gallery = "gallery" in Path(image_path).parts

        self.person_features[image_path] = {
            'features': features,
            'predicted_id': person_id,
            'true_id': true_person_id,
            'camera_id': camera_id,
            'filename': filename,
            'is_query': is_query,
            'is_gallery': is_gallery
        }

    def _extract_true_person_id(self, filename: str) -> int:
        """ファイル名から真の人物IDを抽出

        Args:
            filename: ファイル名

        Returns:
            真の人物ID（抽出できない場合は-1）
        """
        import re

        # パターン1: person_001, id_001, 001_cam1 など
        patterns = [
            r'person_(\d+)',
            r'id_(\d+)',
            r'^(\d+)_',
            r'_(\d+)_cam',
            r'_(\d+)$'
        ]

        for pattern in patterns:
            match = re.search(pattern, filename.lower())
            if match:
                return int(match.group(1))

        # パターンが見つからない場合、ファイル名をハッシュ化してIDとする
        return hash(filename) % 1000

    def _extract_camera_id(self, filename: str) -> int:
        """ファイル名からカメラIDを抽出

        Args:
            filename: ファイル名

        Returns:
            カメラID（抽出できない場合は0）
        """
        import re

        # パターン: cam1, camera1, c1 など
        patterns = [
            r'cam(\d+)',
            r'camera(\d+)',
            r'c(\d+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, filename.lower())
            if match:
                return int(match.group(1))

        return 0  # デフォルトカメラID

    def _perform_evaluation(self) -> Optional[Dict[str, Any]]:
        """ReID評価を実行

        Returns:
            評価結果（評価できない場合はNone）
        """
        if not self.reid_evaluator or len(self.person_features) < 2:
            return None

        try:
            # 評価データをリセット
            self.reid_evaluator.reset()

            # query/galleryディレクトリベースで分割
            query_count = 0
            gallery_count = 0

            # まずquery/galleryディレクトリベースで分割を試行
            has_query_gallery_structure = any(data.get('is_query') or data.get('is_gallery')
                                             for data in self.person_features.values())

            if has_query_gallery_structure:
                # query/galleryディレクトリ構造がある場合
                self.logger.info("query/galleryディレクトリ構造を使用して評価データを分割します")
                for image_path, data in self.person_features.items():
                    if data.get('is_query', False):
                        # queryディレクトリの画像をクエリに追加
                        self.reid_evaluator.add_query(
                            data['features'],
                            data['true_id'],
                            data['camera_id']
                        )
                        query_count += 1
                    elif data.get('is_gallery', False):
                        # galleryディレクトリの画像をギャラリーに追加
                        self.reid_evaluator.add_gallery(
                            data['features'],
                            data['true_id'],
                            data['camera_id']
                        )
                        gallery_count += 1
            else:
                # 従来の方法: 各人物の最初の画像をクエリ、残りをギャラリーとする
                self.logger.info("従来の方法で評価データを分割します（各人物の最初の画像をクエリ）")
                person_groups = {}
                for image_path, data in self.person_features.items():
                    true_id = data['true_id']
                    if true_id not in person_groups:
                        person_groups[true_id] = []
                    person_groups[true_id].append((image_path, data))

                for true_id, images in person_groups.items():
                    if len(images) > 1:
                        # 複数画像がある場合、最初をクエリ、残りをギャラリー
                        query_image = images[0]
                        gallery_images = images[1:]

                        # クエリに追加
                        self.reid_evaluator.add_query(
                            query_image[1]['features'],
                            true_id,
                            query_image[1]['camera_id']
                        )
                        query_count += 1

                        # ギャラリーに追加
                        for gallery_image in gallery_images:
                            self.reid_evaluator.add_gallery(
                                gallery_image[1]['features'],
                                true_id,
                                gallery_image[1]['camera_id']
                            )
                            gallery_count += 1
                    else:
                        # 単一画像の場合はギャラリーに追加
                        self.reid_evaluator.add_gallery(
                            images[0][1]['features'],
                            true_id,
                            images[0][1]['camera_id']
                        )
                        gallery_count += 1

            if query_count == 0 or gallery_count == 0:
                self.logger.warning("評価に十分なデータがありません（クエリまたはギャラリーが空）")
                return None

            # 評価実行
            results = self.reid_evaluator.evaluate()

            # 結果を保存
            self.reid_evaluator.save_evaluation_results(results, self.output_dir)

            # CMC曲線を保存
            cmc_path = Path(self.output_dir) / "cmc_curve.png"
            self.reid_evaluator.plot_cmc_curve(
                results,
                str(cmc_path),
                f"CMC Curve - {self.reid_backend.upper()}"
            )

            # 結果を表示
            self.reid_evaluator.print_evaluation_summary(results)

            self.logger.info(f"評価完了: クエリ{query_count}件, ギャラリー{gallery_count}件")

            return results

        except Exception as e:
            self.logger.error(f"評価実行エラー: {e}")
            return None

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

            # 評価結果があれば追加
            if hasattr(summary, 'evaluation_results') and summary.evaluation_results:
                eval_results = summary.evaluation_results
                summary_dict['evaluation_results'] = {
                    'mAP': float(eval_results['mAP']),
                    'rank1': float(eval_results['rank1']),
                    'rank_k_accuracy': self.reid_evaluator.compute_rank_k_accuracy(eval_results) if self.reid_evaluator else {}
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
    parser = argparse.ArgumentParser(
        description='複数人物画像直接処理アプリケーション',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python src/person_image_app.py --input_dir ./person_images --output_dir ./output --reid_backend clip
  python src/person_image_app.py --input_dir ./person_images --output_dir ./output --reid_backend trans_reid
  python src/person_image_app.py --input_dir ./person_images --output_dir ./output --reid_backend la_transformer
  python src/person_image_app.py --input_dir ./person_images --output_dir ./output --reid_backend clip --disable_evaluation

サポートされる画像形式:
  JPEG, PNG, BMP

ReIDバックエンド:
  clip          - CLIP-ReID (デフォルト)
  trans_reid    - TransReID
  la_transformer - LA-Transformer

評価機能:
  デフォルトで有効。Rank-k Accuracy、mAP、CMC曲線を計算・出力します。
  ファイル名から人物IDとカメラIDを自動抽出します。
  例: person_001_cam1.jpg -> 人物ID: 1, カメラID: 1

ディレクトリ構造:
  1. query/gallery構造（推奨）:
     input_dir/
     ├── query/     # クエリ画像
     └── gallery/   # ギャラリー画像

  2. 通常構造:
     input_dir/     # 全ての画像（自動分割）

注意:
  このアプリケーションは事前に切り出された人物画像を直接処理します。
  人物検出は行わず、ReID特徴抽出とID割り当てのみを実行します。
        """
    )

    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='入力人物画像ファイルが格納されているディレクトリパス'
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

    parser.add_argument(
        '--enable_evaluation',
        action='store_true',
        default=True,
        help='ReID評価機能を有効にする (デフォルト: True)'
    )

    parser.add_argument(
        '--disable_evaluation',
        action='store_true',
        help='ReID評価機能を無効にする'
    )

    try:
        args = parser.parse_args()

        # ログレベルを更新
        if args.log_level != 'INFO':
            logging.getLogger().setLevel(getattr(logging, args.log_level))

        # 評価機能の設定
        if args.disable_evaluation:
            args.enable_evaluation = False

        # 引数検証
        if not os.path.exists(args.input_dir):
            print(f"エラー: 入力ディレクトリが存在しません: {args.input_dir}")
            return 1

        if not os.path.isdir(args.input_dir):
            print(f"エラー: 指定されたパスはディレクトリではありません: {args.input_dir}")
            return 1

        # 出力ディレクトリの親ディレクトリが存在するかチェック
        output_parent = Path(args.output_dir).parent
        if not output_parent.exists():
            print(f"エラー: 出力ディレクトリの親ディレクトリが存在しません: {output_parent}")
            return 1

        print(f"=== 人物画像処理アプリケーション ===")
        print(f"入力ディレクトリ: {args.input_dir}")
        print(f"出力ディレクトリ: {args.output_dir}")
        print(f"ReIDバックエンド: {args.reid_backend}")
        print(f"ログレベル: {args.log_level}")
        print()

        # アプリケーション実行
        app = PersonImageReIDApp(
            reid_backend=args.reid_backend,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            enable_evaluation=args.enable_evaluation
        )

        # 人物画像処理実行
        summary = app.process_person_images()

        # 結果表示
        print(f"\n=== 処理完了 ===")
        print(f"総ファイル数: {summary.total_files}")
        print(f"成功: {summary.successful_files}")
        print(f"失敗: {summary.failed_files}")
        print(f"検出された人物総数: {summary.total_persons_detected}")
        print(f"ユニーク人物数: {summary.unique_person_count}")
        print(f"使用ReIDバックエンド: {summary.reid_backend}")
        print(f"総処理時間: {summary.total_processing_time:.2f}秒")
        print(f"処理結果は以下に保存されました: {args.output_dir}")

        # 評価結果の表示
        if hasattr(summary, 'evaluation_results') and summary.evaluation_results:
            eval_results = summary.evaluation_results
            print(f"\n=== ReID評価結果 ===")
            print(f"mAP: {eval_results['mAP']:.1%}")
            print(f"Rank-1: {eval_results['rank1']:.1%}")
            print(f"評価結果とCMC曲線が保存されました")

        if summary.failed_files > 0:
            print(f"\n失敗したファイル:")
            for result in summary.results:
                if not result.success:
                    print(f"  - {result.file_path}: {result.error_message}")
            return 2  # 部分的な失敗

        return 0  # 成功

    except KeyboardInterrupt:
        print("\n処理が中断されました")
        return 130

    except Exception as e:
        print(f"予期しないエラーが発生しました: {e}")
        logging.getLogger(__name__).exception("予期しないエラー")
        return 1


if __name__ == "__main__":
    exit(main())
