import io
from datetime import datetime

from entities.person_crop_image import EntityPersonCropImage
from fastapi import UploadFile
from PIL import Image


class PresentationIdentifyPersonParse:
    @staticmethod
    async def parse(
        images: list[UploadFile],
        camera_id: int,
        view_id: int,
        timestamp_str: str
    ) -> list[EntityPersonCropImage]:
        person_crop_images = []
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
        for image in images:
            image_bytes = await image.read()
            image_pil = Image.open(io.BytesIO(image_bytes))
            if image_pil.mode != "RGB":
                image_pil = image_pil.convert("RGB")
            person_crop_image = EntityPersonCropImage(
                image_pil, camera_id, view_id, timestamp)
            person_crop_images.append(person_crop_image)
        return person_crop_images
