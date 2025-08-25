"""動画ファイル処理アプリケーション"""
from dataclasses import dataclass
from pathlib import Path
import cv2
import torch
from processors.logger import LoggerProcessor
from processors.directory.videos import VideosDirectoryProcessor
from processors.pre.clahe import ClahePreProcessor
from processors.pre.retinex import RetinexPreProcessor
from processors.pre.homorphic_filter import HomorphicFilterProcessor
from processors.pre.logarithmic_transform import LogarithmicTransformProcessor
from processors.pre.ace import AcePreProcessor
from processors.pre.wavelet import WaveletPreProcessor
from processors.pre.msrcr import MsrcrPreProcessor
from processors.pre.ganma import GanmaPreProcessor
from processors.pre.tone_curve import ToneCurvePreProcessor


class VideoReIDApp:
    """動画ファイル処理アプリケーション"""

    def __init__(self):
        self.logger = LoggerProcessor.setup_logging()
        self.videos_directory_processor = VideosDirectoryProcessor()
        self.device = "cuda"
        self.clahe_processor = ClahePreProcessor(self.device)
        self.retinex_processor = RetinexPreProcessor(self.device)
        self.homorphic_filter_processor = HomorphicFilterProcessor(self.device)
        self.logarithmic_transform_processor = LogarithmicTransformProcessor(
            self.device)
        self.ace_processor = AcePreProcessor(self.device)
        self.msrcr_processor = MsrcrPreProcessor(self.device)
        self.wavelet_processor = WaveletPreProcessor(self.device, brightness_uniformity=True, edge_enhancement=False, brightness_factor=1.3, preserve_contrast=True)
        self.ganma_processor = GanmaPreProcessor()
        self.tone_curve_processor = ToneCurvePreProcessor()

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

        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
            if frame is None:
                continue
            frame_number = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if frame_number == 1200:
                output_file_dir_path = self.videos_directory_processor.get_output_dir_path()
                cv2.imwrite(str(output_file_dir_path / f"original_{frame_number}.jpg"), frame)
                clahe_frame = self.clahe_processor.process(frame)
                cv2.imwrite(str(output_file_dir_path / f"clahe_{frame_number}.jpg"), clahe_frame)
                retinex_frame = self.retinex_processor.process(frame)
                cv2.imwrite(str(output_file_dir_path / f"retinex_{frame_number}.jpg"), retinex_frame)
                homorphic_filter_frame = self.homorphic_filter_processor.process(frame)
                cv2.imwrite(str(output_file_dir_path / f"homorphic_filter_{frame_number}.jpg"), homorphic_filter_frame)
                logarithmic_transform_frame = self.logarithmic_transform_processor.process(frame)
                cv2.imwrite(str(output_file_dir_path / f"logarithmic_transform_{frame_number}.jpg"), logarithmic_transform_frame)
                ace_frame = self.ace_processor.process(frame)
                cv2.imwrite(str(output_file_dir_path / f"ace_{frame_number}.jpg"), ace_frame)
                wavelet_frame = self.wavelet_processor.process(frame)
                cv2.imwrite(str(output_file_dir_path / f"wavelet_{frame_number}.jpg"), wavelet_frame)
                ganma_frame = self.ganma_processor.process(frame)
                cv2.imwrite(str(output_file_dir_path / f"ganma_{frame_number}.jpg"), ganma_frame)
                tone_curve_frame = self.tone_curve_processor.process(frame)
                cv2.imwrite(str(output_file_dir_path / f"tone_curve_{frame_number}.jpg"), tone_curve_frame)
                # msrcr_frame = self.msrcr_processor.process(frame)
                # cv2.imwrite(str(output_file_dir_path /f"msrcr_{frame_number}.jpg"), msrcr_frame)
                break

        video_capture.release()
        self.logger.info(f"動画ファイルの処理が完了しました: {video_file_path}")


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
