import asyncio

from application.identify_person.background_process import \
    ApplicationIdentifyPersonBackgroundProcess
from entities.person_crop_image import EntityPersonCropImage
from service.storage.person_crop_images import ServiceStoragePersonCropImages


class ApplicationIdentifyPerson:
    @staticmethod
    async def process(person_crop_images: list[EntityPersonCropImage]) -> None:
        for person_crop_image in person_crop_images:
            asyncio.to_thread(ApplicationIdentifyPersonBackgroundProcess.add_task(person_crop_image))
