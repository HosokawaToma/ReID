import asyncio
from queue import Queue

from entities.person_crop_image import EntityPersonCropImage
from service.identify_person import ServiceIdentifyPerson


class ApplicationIdentifyPersonBackgroundProcess:
    running = False
    queue = Queue()
    tasks: list[asyncio.Task] = []

    @staticmethod
    async def start():
        ApplicationIdentifyPersonBackgroundProcess.running = True
        task = asyncio.create_task(ApplicationIdentifyPersonBackgroundProcess.process_queue())
        ApplicationIdentifyPersonBackgroundProcess.tasks.append(task)

    @staticmethod
    async def stop():
        ApplicationIdentifyPersonBackgroundProcess.running = False
        for task in ApplicationIdentifyPersonBackgroundProcess.tasks:
            task.cancel()
        ApplicationIdentifyPersonBackgroundProcess.tasks = []

    @staticmethod
    async def add_task(person_crop_image: EntityPersonCropImage):
        ApplicationIdentifyPersonBackgroundProcess.queue.put(person_crop_image)

    @staticmethod
    async def process_queue():
        while ApplicationIdentifyPersonBackgroundProcess.running:
            person_crop_image = ApplicationIdentifyPersonBackgroundProcess.queue.get()
            await ApplicationIdentifyPersonBackgroundProcess.process(person_crop_image)

    @staticmethod
    async def process(person_crop_image: EntityPersonCropImage):
        image = person_crop_image.image
        camera_id = person_crop_image.camera_id
        view_id = person_crop_image.view_id
        person_id = await ServiceIdentifyPerson.Identify(image, camera_id, view_id)
