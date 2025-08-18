"""動画ファイル処理アプリケーション"""
import cv2
import logging
import time
from pathlib import Path
from typing import List, Tuple
import numpy as np
from dataclasses import dataclass
from managers.data_manager import DataManager
from managers.yolo_model_manager import YoloModelManager
from managers.reid_model_manager import ReIDModelManager
from managers.post_processing_manager import PostProcessingManager
from managers.pre_processing_manager import PreProcessingManager


@dataclass
class VideoAppConfig:
    class REID_BACKEND:
        name: str = "clip"

    class Directories:
        input_dir_str: str = "./resources/videos/input"
        output_dir_str: str = "./resources/videos/output"

    class PreProcessing:
        clahe: bool = False
        retinex: bool = False


VIDEO_APP_CONFIG = VideoAppConfig()


class VideoReIDApp:
    """動画ファイル処理アプリケーション"""

    def __init__(
        self,
        reid_backend: str = VIDEO_APP_CONFIG.REID_BACKEND.name,
        input_dir_str: str = VIDEO_APP_CONFIG.Directories.input_dir_str,
        output_dir_str: str = VIDEO_APP_CONFIG.Directories.output_dir_str,
        clahe: bool = VIDEO_APP_CONFIG.PreProcessing.clahe,
        retinex: bool = VIDEO_APP_CONFIG.PreProcessing.retinex
    ):
        """初期化

        Args:
            reid_backend: 使用するReIDバックエンド ('clip', 'trans_reid', 'la_transformer')
        """
        self._setup_logging()
        self.reid_backend = reid_backend
        self.input_dir_str = input_dir_str
        self.output_dir_str = output_dir_str
        self.clahe = clahe
        self.retinex = retinex
    def _setup_logging(self) -> None:
        """ログ設定の初期化"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _set_directories(self) -> None:
        """ディレクトリの設定"""
        self.input_dir_path = Path(self.input_dir_str)
        self.output_dir_path = Path(self.output_dir_str)

    def _validate_directories(self) -> None:
        """ディレクトリの存在確認と作成"""
        if not self.input_dir_path.exists():
            raise FileNotFoundError(f"入力ディレクトリが存在しません: {self.input_dir_path}")

        if not self.input_dir_path.is_dir():
            raise NotADirectoryError(
                f"指定されたパスはディレクトリではありません: {self.input_dir_path}")

        if not self.output_dir_path.exists():
            self.output_dir_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            f"ディレクトリ検証完了 - 入力: {self.input_dir_path}, 出力: {self.output_dir_path}")

    def _set_components(self) -> None:
        """コンポーネントの初期化"""
        self.logger.info("コンポーネントの初期化を開始します...")

        self.data_manager = DataManager()

        self.yolo_manager = YoloModelManager()

        self.reid_manager = ReIDModelManager(backend=self.reid_backend)

        self.pre_processing_manager = PreProcessingManager()

        self.post_processing_manager = PostProcessingManager()

        self.logger.info(f"全てのコンポーネントの初期化が完了しました")

    def _initialize_process(self) -> None:
        """アプリケーションの初期化"""
        self.logger.info("アプリケーションの初期化を開始します...")
        self._set_directories()
        self._validate_directories()
        self._set_components()
        self.logger.info("アプリケーションの初期化が完了しました")

    def _main_process(self) -> None:
        """複数動画ファイルを処理

        Returns:
            処理結果のサマリー辞書
        """
        self.logger.info("メイン処理を開始します...")

        start_time = time.time()

        try:
            video_files = list(self.input_dir_path.glob("*.mp4"))

            if not video_files:
                self.logger.warning(
                    f"処理対象の動画ファイルが見つかりませんでした: {self.input_dir_path}")

            self.logger.info(f"処理対象動画ファイル数: {len(video_files)}")

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

        except Exception as e:
            self.logger.error(f"動画処理中に予期しないエラーが発生しました: {e}")

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
                    processed_frame = self._process_frame(
                        frame, frame_count, video_id)
                    out.write(processed_frame)
                    frame_count += 1

                    if frame_count % 100 == 0:
                        self.logger.info(f"フレーム処理進捗: {frame_count}")

                except Exception as e:
                    self.logger.warning(f"フレーム {frame_count} の処理中にエラー: {e}")
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
            person_detections = self.yolo_manager.extract_person_crop_from_box(frame)

            if not person_detections:
                self.logger.debug(f"フレーム {frame_number}: 人物が検出されませんでした")
                return frame, []

            self.logger.debug(
                f"フレーム {frame_number}: {len(person_detections)}人の人物を検出")
            processed_frame = frame.copy()

            for i, person_detection in enumerate(person_detections):
                try:
                    self.logger.debug(f"フレーム {frame_number}, 人物 {i+1}: 特徴抽出開始")

                    if self.clahe:
                        person_detection.person_crop = self.pre_processing_manager.clahe(person_detection.person_crop)
                    if self.retinex:
                        person_detection.person_crop = self.pre_processing_manager.retinex(person_detection.person_crop)

                    feat = self.reid_manager.extract_features(
                        person_detection.person_crop, camera_id=video_id)

                    person_id = self.post_processing_manager.assign_person_id(
                        feat, self.data_manager.gallery_feats, self.data_manager.gallery_person_ids)

                    self.logger.debug(
                        f"フレーム {frame_number}, 人物 {i+1}: ID {person_id} を割り当て")

                    processed_frame = self._draw_detection(
                        processed_frame, person_detection.bounding_box, person_id)
                    self.data_manager.add_gallery(person_id, video_id, 0, feat)

                except Exception as e:
                    self.logger.warning(
                        f"人物処理エラー (フレーム {frame_number}, 人物 {i+1}): {e}")
                    continue

            return processed_frame

        except Exception as e:
            self.logger.error(f"フレーム処理エラー (フレーム {frame_number}): {e}")
            return frame, []

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

        color = self._get_color_for_id(person_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"ID: {person_id}"
        label_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                      (x1 + label_size[0], y1), color, -1)

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
        colors = [
            (0, 0, 139),      # 濃い赤
            (139, 0, 0),      # 濃い青
            (0, 100, 0),      # ダークグリーン
            (128, 0, 128),    # 紫
            (0, 128, 128),    # 濃いシアン
            (128, 128, 0),    # オリーブ
            (255, 69, 0),     # オレンジレッド
            (72, 61, 139),    # ダークブルー
            (0, 0, 128),      # ネイビーブルー
            (85, 107, 47),    # ダークオリーブグリーン
            (139, 69, 19),    # サドルブラウン
            (0, 139, 139),    # ダークシアン
            (46, 139, 87),    # シーグリーン
            (160, 32, 240),   # パープル
            (0, 191, 255),    # ディープスカイブルー
            (255, 140, 0),    # ダークオレンジ
            (0, 128, 0),      # グリーン
            (0, 0, 205),      # ミディアムブルー
            (34, 139, 34),    # フォレストグリーン
            (255, 20, 147),   # ディープピンク
            (25, 25, 112),    # ミッドナイトブルー
            (128, 0, 0),      # マルーン
            (0, 255, 127),    # スプリンググリーン
            (255, 0, 127),    # ローズ
            (70, 130, 180),   # スチールブルー
            (0, 206, 209),    # ダークターコイズ
            (199, 21, 133),   # ミディアムバイオレットレッド
            (255, 0, 255),    # マゼンタ
            (0, 191, 255),    # ディープスカイブルー
            (139, 0, 139),    # ダークマゼンタ
        ]

        return colors[person_id % len(colors)]

    def run(self) -> None:
        """アプリケーションの実行"""
        self._initialize_process()
        self._main_process()


def main():
    """メイン関数 - コマンドライン実行用"""

    app = VideoReIDApp()
    app.run()


if __name__ == "__main__":
    import cProfile
    import pstats
    with cProfile.Profile() as pr:
        main()
    stats = pstats.Stats(pr)
    stats.sort_stats("cumtime").print_stats(30)  # 上位30件を表示
