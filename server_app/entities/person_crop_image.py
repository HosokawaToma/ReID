from datetime import datetime
from PIL import Image
import uuid
from pathlib import Path


class EntityPersonCropImage:
    def __init__(self, image: Image, camera_id: int, view_id: int, timestamp: datetime, person_id: int = None, filepath: Path = None):
        self.id = uuid.uuid4()
        self.image = image
        self.camera_id = camera_id
        self.view_id = view_id
        self.timestamp = timestamp
        self.person_id = person_id
        self.filepath = filepath
