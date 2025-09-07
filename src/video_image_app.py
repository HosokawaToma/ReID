"""動画ファイル処理アプリケーション"""
from pathlib import Path
import cv2
from processors.logger import LoggerProcessor
from processors.directory.videos import VideosDirectoryProcessor
from processors.pre.ganma import GanmaPreProcessor


class VideoReIDApp:
    """動画ファイル処理アプリケーション"""

    def __init__(self):
        self.logger = LoggerProcessor.setup_logging()
        self.videos_directory_processor = VideosDirectoryProcessor()
        self.device = "cuda"
        self.ganma_processor = GanmaPreProcessor()

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
        video_file_paths = self.videos_directory_processor.get_video_file_paths()
        target_video_file_path = video_file_paths[4]
        self._process_video(target_video_file_path)
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
                ganma_frame = self.ganma_processor.process(frame)
                cv2.imwrite(str(output_file_dir_path / f"ganma_{frame_number}.jpg"), ganma_frame)
                cv2.imwrite(str(output_file_dir_path / f"original_{frame_number}.jpg"), frame)

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
