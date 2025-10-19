from application.video.loader import ApplicationVideoLoader
from application.video.writer import ApplicationVideoWriter
from application.yolo.detection import YoloDetection
from application.yolo.detection.crops import ApplicationYoloDetectionCrops
from application.yolo.segmentation import ApplicationYoloSegmentation
from application.yolo.segmentation.isolation import ApplicationYoloSegmentationIsolation
from application.yolo.pose import ApplicationYoloPose
from application.yolo.pose.verification import ApplicationYoloPoseVerification
from application.yolo.segmentation.verification import ApplicationYoloSegmentationVerification
from application.reid import ApplicationReID
from application.video.frame.drawer import ApplicationVideoFrameDrawer
from application.video.frame.correction.gamma import ApplicationVideoFrameCorrectionGamma


class ApplicationVideoReID:
    def __init__(self, input_video_file_paths: list[str], output_video_file_paths: list[str]):
        self.input_video_file_paths = input_video_file_paths
        self.output_video_file_paths = output_video_file_paths
        self.application_video_frame_correction_gamma = ApplicationVideoFrameCorrectionGamma()
        self.yolo_detection = YoloDetection()
        self.yolo_detection_crops = ApplicationYoloDetectionCrops()
        self.yolo_segmentation = ApplicationYoloSegmentation()
        self.yolo_segmentation_isolation = ApplicationYoloSegmentationIsolation()
        self.yolo_pose = ApplicationYoloPose()
        self.yolo_pose_verification = ApplicationYoloPoseVerification()
        self.yolo_segmentation_verification = ApplicationYoloSegmentationVerification()
        self.reid = ApplicationReID()
        self.application_video_frame_drawer = ApplicationVideoFrameDrawer()

    def run(self):
        for input_video_file_path, output_video_file_path in zip(self.input_video_file_paths, self.output_video_file_paths):
            video_loader = ApplicationVideoLoader(input_video_file_path)
            video_frame_rate = video_loader.get_frame_rate()
            video_frame_width = video_loader.get_frame_width()
            video_frame_height = video_loader.get_frame_height()
            video_writer = ApplicationVideoWriter(output_video_file_path, video_frame_rate, video_frame_width, video_frame_height)
            for frame in video_loader.load_video_frame():
                frame = self.application_video_frame_correction_gamma.correct(frame)
                detections = self.yolo_detection.extract(frame)
                for detection in detections:
                    crop_frame = self.yolo_detection_crops.crop(frame, detection)
                    masks = self.yolo_segmentation.extract(crop_frame)
                    if not self.yolo_segmentation_verification.verify(masks):
                        continue
                    keypoints = self.yolo_pose.extract(crop_frame)
                    if not self.yolo_pose_verification.verify(keypoints):
                        continue
                    feature = self.reid.extract_feature(crop_frame)
                    person_id = self.reid.assign_id(feature)
                    frame = self.application_video_frame_drawer.draw(frame, person_id, detection.x1, detection.y1, detection.x2, detection.y2)
                video_writer.write_video_frame(frame)
            print(f"Processed {input_video_file_path}")

def main():
    input_video_file_paths = [
        "resources/videos/input/video1.mp4",
        "resources/videos/input/video2.mp4",
        "resources/videos/input/video3.mp4",
        "resources/videos/input/video4.mp4",
        "resources/videos/input/video5.mp4",
    ]
    output_video_file_paths = [
        "resources/videos/output/video1_clip_reid_gamma.mp4",
        "resources/videos/output/video2_clip_reid_gamma.mp4",
        "resources/videos/output/video3_clip_reid_gamma.mp4",
        "resources/videos/output/video4_clip_reid_gamma.mp4",
        "resources/videos/output/video5_clip_reid_gamma.mp4",
    ]
    application_video_reid = ApplicationVideoReID(input_video_file_paths, output_video_file_paths)
    application_video_reid.run()


if __name__ == "__main__":
    main()
