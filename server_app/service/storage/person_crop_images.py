from pathlib import Path

import cv2

from entities.person_crop_image import EntityPersonCropImage


class ServiceStoragePersonCropImages:
    path = Path("results/identify_person")

    @staticmethod
    def save(person_crop_image: EntityPersonCropImage) -> Path:
        filename = f"{person_crop_image.id}_{person_crop_image.person_id}_{person_crop_image.camera_id}_{person_crop_image.view_id}.jpg"
        filepath = ServiceStoragePersonCropImages.path / filename
        cv2.imwrite(filepath, person_crop_image.image)
        return filepath
