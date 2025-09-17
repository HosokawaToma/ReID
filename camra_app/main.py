from datetime import datetime

from application.api import ApplicationApi
from application.yolo_camera import ApplicationYoloCamera
from entities.api.person_crop_images_metadata import \
    EntityApiPersonCropImagesMetadata


class CameraApp:
    def __init__(self):
        self.yolo_camera_application = ApplicationYoloCamera()
        self.api_application = ApplicationApi(server_ip="localhost:8000")

    def run(self):
        for person_cropped_images in self.yolo_camera_application.run():
            metadata = EntityApiPersonCropImagesMetadata(
                camera_id=0,
                view_id=0,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            )
            self.api_application.post_person_cropped_images(
                person_cropped_images,
                metadata=metadata)

if __name__ == "__main__":
    app = CameraApp()
    app.run()
