import io

from entities.person_crop_image import EntityPersonCropImage
from fastapi import UploadFile
from PIL import Image
from presentation.identify_person.metadate import \
    PresentationIdentifyPersonMetadata


class PresentationIdentifyPersonParse:
    @staticmethod
    async def parse(
        images: list[UploadFile],
        metadata: PresentationIdentifyPersonMetadata
    ) -> list[EntityPersonCropImage]:
        person_crop_images = []
        for image in images:
            image_bytes = await image.read()
            image_pil = Image.open(io.BytesIO(image_bytes))
            person_crop_image = EntityPersonCropImage(
                image_pil, metadata.camera_id, metadata.view_id, metadata.timestamp)
            person_crop_images.append(person_crop_image)
        return person_crop_images
