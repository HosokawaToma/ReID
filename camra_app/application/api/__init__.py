import numpy as np
import requests

from entities.api.person_crop_images_metadata import \
    EntityApiPersonCropImagesMetadata


class ApplicationApi:
    def __init__(self, server_ip: str):
        self.server_ip = server_ip
        self.person_cropped_images_url = "http://" + self.server_ip + "/identify_person"

    def post_person_cropped_images(self, person_cropped_images: list[np.ndarray], metadata: EntityApiPersonCropImagesMetadata):
        files = [
            ("images", (f"person_cropped_image_{i}.jpg", person_cropped_image, "image/jpeg"))
            for i, person_cropped_image in enumerate(person_cropped_images)
        ]
        data = {
            "camera_id": metadata.camera_id,
            "view_id": metadata.view_id,
            "timestamp": metadata.timestamp
        }
        requests.post(self.person_cropped_images_url, files=files, data=data)
