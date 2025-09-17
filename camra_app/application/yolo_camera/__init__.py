from service.camera import ServiceCamera
from service.yolo import ServiceYolo
from service.yolo.crop_person_frame import ServiceYoloCropPersonFrame
from service.yolo.detections_verification import \
    ServiceYoloDetectionsVerification


class ApplicationYoloCamera:
    def __init__(self):
        self.camera = ServiceCamera()
        self.yolo = ServiceYolo()

    def __del__(self):
        self.camera.__del__()

    def run(self):
        for frame in self.camera.run():
            detections = self.yolo.extract_person_detections(frame)
            verified_detections = ServiceYoloDetectionsVerification.verify(detections)
            person_cropped_images = ServiceYoloCropPersonFrame.crops(frame, verified_detections)
            yield person_cropped_images
