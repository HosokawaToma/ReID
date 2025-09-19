from PIL import Image


class EntityPersonCropImage:
    def __init__(self, image: Image, camera_id: int, view_id: int, timestamp: float):
        self.image = image
        self.camera_id = camera_id
        self.view_id = view_id
        self.timestamp = timestamp
