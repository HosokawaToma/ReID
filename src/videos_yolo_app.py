import glob
import os
import cv2
from dataclasses import dataclass
from managers.yolo_model_manager import YoloModelManager

@dataclass
class VideosYoloAppConfig:
    class Directories:
        input_dir_str: str = "resources/videos/input"
        output_dir_str: str = "resources/videos/output"


class VideosYoloApp:
    def __init__(self):
        self.yolo_model_manager = YoloModelManager()
        self.input_dir_str = VideosYoloAppConfig.Directories.input_dir_str
        self.output_dir_str = VideosYoloAppConfig.Directories.output_dir_str

    def _get_video_paths(self):
        video_paths = glob.glob(os.path.join(self.input_dir_str, "*.mp4"))
        return video_paths

    def _process_video(self, video_path, video_id):
        cap = cv2.VideoCapture(video_path)
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            person_crop_list = self.yolo_model_manager.extract_person_crop_from_box(frame)
            for _, person_crop, track_id in person_crop_list:
                file_name = f"{video_id}_{i}.jpg"
                output_dir = os.path.join(self.output_dir_str, f"{track_id}")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                file_path = os.path.join(output_dir, file_name)
                cv2.imwrite(file_path, person_crop)
                i += 1
        cap.release()

    def run(self):
        video_paths = self._get_video_paths()
        for i, video_path in enumerate(video_paths):
            self._process_video(video_path, i)

if __name__ == "__main__":
    videos_yolo_app = VideosYoloApp()
    videos_yolo_app.run()
