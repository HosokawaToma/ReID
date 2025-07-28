import glob
import os
import cv2
from managers.yolo_model_manager import YoloModelManager
from config import VIDEOS_YOLO_APP_CONFIG


class VideosYoloApp:
    def __init__(self):
        self.yolo_model_manager = YoloModelManager()
        self.input_dir_str = VIDEOS_YOLO_APP_CONFIG.Directories.input_dir_str
        self.output_dir_str = VIDEOS_YOLO_APP_CONFIG.Directories.output_dir_str

    def _get_video_paths(self):
        video_paths = glob.glob(os.path.join(self.input_dir_str, "*.mp4"))
        return video_paths

    def _process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            person_crop_list = self.yolo_model_manager.extract_person_crop_from_box(frame)
            for person_crop in person_crop_list:
                cv2.imwrite(os.path.join(self.output_dir_str, f"{video_path.split('/')[-1].split('.')[0]}_{i}.jpg"), person_crop)
                i += 1
        cap.release()

    def run(self):
        video_paths = self._get_video_paths()
        for video_path in video_paths:
            self._process_video(video_path)

if __name__ == "__main__":
    videos_yolo_app = VideosYoloApp()
    videos_yolo_app.run()
