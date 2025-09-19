from application.identify_person.background_process import \
    ApplicationIdentifyPersonBackgroundProcess
from entities.person_crop_image import EntityPersonCropImage


class ApplicationIdentifyPerson:
    @staticmethod
    async def process(person_crop_images: list[EntityPersonCropImage]) -> None:
        for person_crop_image in person_crop_images:
            await ApplicationIdentifyPersonBackgroundProcess.add_task(person_crop_image)
