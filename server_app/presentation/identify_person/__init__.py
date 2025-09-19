from fastapi import Depends, FastAPI, UploadFile

from application.identify_person import ApplicationIdentifyPerson
from application.identify_person.background_process import \
    ApplicationIdentifyPersonBackgroundProcess
from presentation.identify_person.metadate import \
    PresentationIdentifyPersonMetadata
from presentation.identify_person.parse import PresentationIdentifyPersonParse


class PresentationIdentifyPerson:
    @staticmethod
    def setup(fastapi_app: FastAPI):
        fastapi_app.add_api_route(
            "/identify_person", PresentationIdentifyPerson.endpoint, methods=["POST"])
        fastapi_app.add_event_handler("startup", PresentationIdentifyPerson.startup)
        fastapi_app.add_event_handler("shutdown", PresentationIdentifyPerson.shutdown)

    @staticmethod
    async def endpoint(
        images: list[UploadFile],
        metadata: PresentationIdentifyPersonMetadata = Depends()
    ):
        person_crop_images = await PresentationIdentifyPersonParse.parse(images, metadata)
        await ApplicationIdentifyPerson.process(person_crop_images)
        return {"message": "Success"}

    @staticmethod
    async def startup():
        ApplicationIdentifyPersonBackgroundProcess.start()

    @staticmethod
    async def shutdown():
        ApplicationIdentifyPersonBackgroundProcess.stop()
