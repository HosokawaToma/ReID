"""動画ファイル処理アプリケーション"""
import os
import cv2
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from config import VIDEO_APP_CONFIG
from managers.data_manager import DataManager
from managers.yolo_model_manager import YoloModelManager
from managers.reid_model_manager import ReIDModelManager
from managers.post_processing_manager import PostProcessingManager


# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class VideoReIDApp:
    """動画ファイル処理アプリケーション"""

    def __init__(self, video_dir_str: str, reid_backend: str):
        """初期化

        Args:
            video_dir_str: 動画ディレクトリパス
            reid_backend: 使用するReIDバックエンド ('clip', 'trans_reid', 'la_transformer')
        """
        self.reid_backend = reid_backend
        self.video_dir_path = Path(video_dir_str)
        self.input_dir_path = self.video_dir_path / "input"
        self.output_dir_path = self.video_dir_path / "output"

        # ログ設定
        self.logger = logging.getLogger(__name__)

        # 人物追跡用の状態管理
        self.person_tracker = {}  # {track_id: {'features': [], 'last_seen': frame_count}}
        self.next_person_id = 1
        self.feature_threshold = 0.7  # 類似度閾値

        # ディレクトリの検証と作成
        self._validate_directories()

        # コンポーネント初期化
        self._initialize_components()

    def _validate_directories(self) -> None:
        """ディレクトリの存在確認と作成"""
        if not self.video_dir_path.exists():
            raise FileNotFoundError(f"動画ディレクトリが存在しません: {self.video_dir_path}")

        if not self.input_dir_path.exists():
            raise FileNotFoundError(f"入力ディレクトリが存在しません: {self.input_dir_path}")

        # 出力ディレクトリの作成
        self.output_dir_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            f"ディレクトリ検証完了 - 入力: {self.input_dir_path}, 出力: {self.output_dir_path}")

    def _initialize_components(self) -> None:
        """AIモデルとコンポーネントの初期化"""
        self.logger.info("AIモデルの初期化を開始...")

        # データ管理の初期化
        self.data_manager = DataManager()

        # YOLO管理の初期化
        self.yolo_manager = YoloModelManager()

        # ReID管理の初期化
        self.reid_manager = ReIDModelManager(backend=self.reid_backend)

        # 後処理管理の初期化
        self.post_processing_manager = PostProcessingManager()

        self.logger.info(f"AIモデル初期化完了 - ReIDバックエンド: {self.reid_backend}")

    def process_videos(self) -> Dict[str, Any]:
        """複数動画ファイルを処理

        Returns:
            処理結果のサマリー辞書
        """
        start_time = time.time()

        try:
            # 動画ファイルを検出
            video_files = list(self.input_dir_path.glob("*.mp4"))

            if not video_files:
                self.logger.warning(
                    f"処理対象の動画ファイルが見つかりませんでした: {self.input_dir_path}")
                return self._create_empty_summary()

            self.logger.info(f"処理対象動画ファイル数: {len(video_files)}")

            # 各動画ファイルを処理
            for i, video_path in enumerate(video_files):
                try:
                    self.logger.info(
                        f"処理開始 ({i+1}/{len(video_files)}): {video_path.name}")
                    self._process_single_video(video_path, i)
                    self.logger.info(
                        f"処理完了 ({i+1}/{len(video_files)}): {video_path.name}")

                except Exception as e:
                    self.logger.error(f"動画処理エラー ({video_path.name}): {e}")

            processing_time = time.time() - start_time
            self.logger.info(f"全動画処理完了 - 処理時間: {processing_time:.2f}秒")

            return self._create_summary(processing_time, len(video_files))

        except Exception as e:
            self.logger.error(f"動画処理中に予期しないエラーが発生しました: {e}")
            return self._create_empty_summary()

    def _process_single_video(self, video_path: Path, video_id: int) -> None:
        """単一動画ファイルを処理

        Args:
            video_path: 動画ファイルパス
            video_id: 動画ID
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"動画ファイルを開けませんでした: {video_path}")

        try:
            # 出力動画の設定
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            output_path = self.output_dir_path / \
                f"processed_{video_path.stem}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(output_path), fourcc, fps, (width, height))

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    # フレーム処理
                    processed_frame, frame_persons = self._process_frame(
                        frame, frame_count, video_id)
                    out.write(processed_frame)
                    frame_count += 1

                    # 進捗ログ（一定間隔で）
                    if frame_count % 100 == 0:
                        self.logger.info(f"フレーム処理進捗: {frame_count}")

                except Exception as e:
                    self.logger.warning(f"フレーム {frame_count} の処理中にエラー: {e}")
                    # エラーが発生したフレームは元のフレームを使用
                    out.write(frame)
                    frame_count += 1

            self.logger.info(f"動画処理完了: {video_path.name} - {frame_count}フレーム")

        finally:
            cap.release()
            if 'out' in locals():
                out.release()

    def _process_frame(self, frame: np.ndarray, frame_number: int, video_id: int) -> Tuple[np.ndarray, List[int]]:
        """単一フレームを処理

        Args:
            frame: 入力フレーム
            frame_number: フレーム番号
            video_id: 動画ID

        Returns:
            (処理済みフレーム, 検出された人物IDのリスト)
        """
        try:
            # YOLO推論で人物検出・追跡
            person_detections = self.yolo_manager._track_persons(frame)

            if not person_detections:
                self.logger.debug(f"フレーム {frame_number}: 人物が検出されませんでした")
                return frame, []

            self.logger.debug(
                f"フレーム {frame_number}: {len(person_detections)}人の人物を検出")
            detected_person_ids = []
            processed_frame = frame.copy()

            # 各検出された人物を処理
            for i, (bounding_box, person_crop) in enumerate(person_detections):
                try:
                    self.logger.debug(f"フレーム {frame_number}, 人物 {i+1}: 特徴抽出開始")

                    # ReID特徴抽出
                    feat = self.reid_manager.extract_features(
                        person_crop, camera_id=video_id)

                    # 人物ID割り当て（改善版）
                    person_id = self._assign_person_id(
                        feat, video_id, frame_number)

                    self.logger.debug(
                        f"フレーム {frame_number}, 人物 {i+1}: ID {person_id} を割り当て")

                    # バウンディングボックスとIDを描画
                    processed_frame = self._draw_detection(
                        processed_frame, bounding_box, person_id)
                    detected_person_ids.append(person_id)

                except Exception as e:
                    self.logger.warning(
                        f"人物処理エラー (フレーム {frame_number}, 人物 {i+1}): {e}")
                    continue

            return processed_frame, detected_person_ids

        except Exception as e:
            self.logger.error(f"フレーム処理エラー (フレーム {frame_number}): {e}")
            return frame, []

    def _assign_person_id(self, feat: np.ndarray, video_id: int, frame_number: int) -> int:
        """人物IDを割り当て（改善版）

        Args:
            feat: 特徴量
            video_id: 動画ID
            frame_number: フレーム番号

        Returns:
            人物ID
        """
        try:
            # 既存の人物との類似度を計算
            best_match_id = None
            best_similarity = 0.0

            for track_id, track_info in self.person_tracker.items():
                if not track_info['features']:
                    continue

                # 最新の特徴量との類似度を計算
                latest_feat = track_info['features'][-1]
                similarity = self._calculate_similarity(feat, latest_feat)

                if similarity > best_similarity and similarity > self.feature_threshold:
                    best_similarity = similarity
                    best_match_id = track_id

            # 新しい人物として割り当て
            if best_match_id is None:
                person_id = self.next_person_id
                self.next_person_id += 1

                # 新しい追跡エントリを作成
                self.person_tracker[person_id] = {
                    'features': [feat.cpu().numpy()],
                    'last_seen': frame_number
                }

                self.logger.debug(f"新しい人物ID {person_id} を作成")
            else:
                person_id = best_match_id
                # 既存の追跡エントリを更新
                self.person_tracker[person_id]['features'].append(
                    feat.cpu().numpy())
                self.person_tracker[person_id]['last_seen'] = frame_number

                # 特徴量履歴を制限（最新10個まで）
                if len(self.person_tracker[person_id]['features']) > 10:
                    self.person_tracker[person_id]['features'] = self.person_tracker[person_id]['features'][-10:]

                self.logger.debug(
                    f"既存の人物ID {person_id} を更新 (類似度: {best_similarity:.3f})")

            return person_id

        except Exception as e:
            self.logger.error(f"人物ID割り当てエラー: {e}")
            # エラー時は新しいIDを割り当て
            person_id = self.next_person_id
            self.next_person_id += 1
            return person_id

    def _calculate_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """2つの特徴量間の類似度を計算

        Args:
            feat1: 特徴量1
            feat2: 特徴量2

        Returns:
            類似度（0-1の範囲）
        """
        try:
            # コサイン類似度を計算
            dot_product = np.dot(feat1.flatten(), feat2.flatten())
            norm1 = np.linalg.norm(feat1.flatten())
            norm2 = np.linalg.norm(feat2.flatten())

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return max(0.0, min(1.0, similarity))  # 0-1の範囲に制限

        except Exception as e:
            self.logger.error(f"類似度計算エラー: {e}")
            return 0.0

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

        # 色を選択（IDに基づいて一貫性のある色を生成）
        color = self._get_color_for_id(person_id)

        # バウンディングボックスを描画
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # IDラベルを描画
        label = f"ID: {person_id}"
        label_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

        # ラベル背景を描画
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                      (x1 + label_size[0], y1), color, -1)

        # ラベルテキストを描画
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def _get_color_for_id(self, person_id: int) -> Tuple[int, int, int]:
        """人物IDに基づいて一貫性のある色を生成

        Args:
            person_id: 人物ID

        Returns:
            BGR色タプル
        """
        # 色のパレット（BGR形式）
        colors = [
            (0, 255, 0),    # 緑
            (255, 0, 0),    # 青
            (0, 0, 255),    # 赤
            (255, 255, 0),  # シアン
            (255, 0, 255),  # マゼンタ
            (0, 255, 255),  # 黄色
            (128, 0, 128),  # 紫
            (0, 128, 128),  # 茶色
            (128, 128, 0),  # オリーブ
            (255, 165, 0),  # オレンジ
        ]

        return colors[person_id % len(colors)]

    def _create_empty_summary(self) -> Dict[str, Any]:
        """空の処理サマリーを作成"""
        return {
            'total_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'total_processing_time': 0.0,
            'reid_backend': self.reid_backend
        }

    def _create_summary(self, processing_time: float, total_files: int) -> Dict[str, Any]:
        """処理サマリーを作成"""
        return {
            'total_files': total_files,
            'successful_files': total_files,  # 簡易版
            'failed_files': 0,  # 簡易版
            'total_processing_time': processing_time,
            'reid_backend': self.reid_backend
        }


def main():
    """メイン関数 - コマンドライン実行用"""

    app = VideoReIDApp(video_dir_str=VIDEO_APP_CONFIG.Directories.input_dir_str,
                       reid_backend=VIDEO_APP_CONFIG.Default.reid_backend)
    app.process_videos()


if __name__ == "__main__":
    main()
