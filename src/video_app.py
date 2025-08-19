"""動画ファイル処理アプリケーション"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2
from typing import Tuple
from processors.logger import LoggerProcessor
from processors.directory.videos import VideosDirectoryProcessor
from processors.yolo import YoloProcessor
from processors.reid.clip import ClipReIDProcessor
from processors.post.assign_person_id import AssignPersonIdPostProcessor

@dataclass
class Config:
    SIMILARITY_THRESHOLD = 0.5


CONFIG = Config()


class VideoReIDApp:
    """動画ファイル処理アプリケーション"""

    def __init__(self):
        self.logger = LoggerProcessor.setup_logging()
        self.videos_directory_processor = VideosDirectoryProcessor()
        self.yolo_processor = YoloProcessor()
        self.clip_reid_processor = ClipReIDProcessor()
        self.assign_person_id_processor = AssignPersonIdPostProcessor(
            similarity_threshold=CONFIG.SIMILARITY_THRESHOLD
        )

    def run(self) -> None:
        """アプリケーションの実行"""
        self.logger.info("アプリケーションの実行を開始します...")

        self._process_directory()

        self._process_videos()

        self.logger.info("アプリケーションの実行が完了しました")

    def _process_directory(self) -> None:
        """ディレクトリの処理"""
        self.logger.info("ディレクトリの処理を開始します...")

        self.logger.info("必要なディレクトリの確認を開始します...")
        if not self.videos_directory_processor.validate_directories():
            self.logger.error("必要なディレクトリが存在しません")
            return
        self.logger.info("必要なディレクトリの確認が完了しました")

        self.logger.info("出力ディレクトリの作成を開始します...")
        self.videos_directory_processor.create_output_directory()
        self.logger.info("出力ディレクトリの作成が完了しました")

        self.logger.info("ディレクトリの処理が完了しました")

    def _process_videos(self) -> None:
        """動画ファイルの処理"""
        self.logger.info("動画ファイルの処理を開始します...")
        for video_file_path in self.videos_directory_processor.get_video_file_paths():
            self._process_video(video_file_path)
        self.logger.info("すべての動画ファイルの処理が完了しました")

    def _process_video(self, video_file_path: Path) -> None:
        """動画ファイルの処理"""
        self.logger.info(f"動画ファイルを処理します: {video_file_path}")
        video_capture = cv2.VideoCapture(str(video_file_path))
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter(
            str(self.videos_directory_processor.output_dir_path / video_file_path.name),
            cv2.VideoWriter_fourcc(*"mp4v"),
            frame_rate,
            (frame_width, frame_height)
        )

        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
            self._process_video_frame(frame, video_writer)

        video_capture.release()
        self.logger.info(f"動画ファイルの処理が完了しました: {video_file_path}")

    def _process_video_frame(self, frame: np.ndarray, video_writer: cv2.VideoWriter) -> None:
        """動画フレームの処理"""
        person_detections = self.yolo_processor.extract_person_detections(frame)
        for person_detection in person_detections:
            feature = self.clip_reid_processor.extract_feature(person_detection.person_crop)
            person_id = self.assign_person_id_processor.assign_person_id(feature)
            frame = self._draw_detection(frame, person_detection.bounding_box, person_id)

        video_writer.write(frame)

    def _draw_detection(self, frame: np.ndarray, bounding_box: np.ndarray, person_id: int) -> np.ndarray:
        color = self._get_color_for_id(person_id)
        frame = cv2.rectangle(
            frame, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), color, 2)
        frame = cv2.putText(
            frame, str(person_id), (bounding_box[0], bounding_box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def _get_color_for_id(self, person_id: int) -> Tuple[int, int, int]:
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

def main():
    app = VideoReIDApp()
    app.run()

if __name__ == "__main__":
    import cProfile
    import pstats
    with cProfile.Profile() as pr:
        main()
    stats = pstats.Stats(pr)
    stats.sort_stats("cumtime").print_stats(30)
