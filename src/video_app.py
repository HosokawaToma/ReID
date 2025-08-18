"""動画ファイル処理アプリケーション"""
import cv2
import logging
import time
from pathlib import Path
from typing import List, Tuple
import numpy as np
from dataclasses import dataclass
from processors.logger import LoggerProcessor
from processors.directory.videos import VideosDirectoryProcessor
from processors.videos.reid_frame import VideosReIDFrameProcessor
from processors.videos.encode import VideosEncodeProcessor
from processors.yolo.extract_person_detections import YoloExtractPersonDetectionsProcessor
from processors.reid.clip import ClipReIDProcessor
from processors.post.assign_person_id import AssignPersonIdPostProcessor

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
        self
    ):
        """初期化"""
        self.logger = LoggerProcessor().setup_logging()
        self.videos_directory_processor = VideosDirectoryProcessor()
        self.videos_directory_processor.validate_directories()
        self.videos_directory_processor.create_output_directory()
        self.clip_reid_processor = ClipReIDProcessor()
        self.yolo_extract_person_detections_processor = YoloExtractPersonDetectionsProcessor()
        self.assign_person_id_processor = AssignPersonIdPostProcessor(
            similarity_threshold=0.5
        )
        self.videos_reid_frame_processor = VideosReIDFrameProcessor(
            extract_person_detections=self.yolo_extract_person_detections_processor.extract_person_detections,
            extract_features=self.clip_reid_processor.extract_feat,
            assign_person_id=self.assign_person_id_processor.assign_person_id
        )
        self.videos_encode_processor = VideosEncodeProcessor(
            process_frame=self.videos_reid_frame_processor.process_frame
        )

    def run(self) -> None:
        """アプリケーションの実行"""
        self.logger.info("アプリケーションの実行を開始します...")

        self.logger.info("動画ファイルを処理します...")
        self.videos_encode_processor.encode_videos()
        self.logger.info("動画ファイルの処理が完了しました")

        self.logger.info("アプリケーションの実行を完了します...")

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
