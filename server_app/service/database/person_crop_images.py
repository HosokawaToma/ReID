from entities.person_crop_image import EntityPersonCropImage
from service.database.session import ServiceDatabaseSession
from migration.models.person_crop_images import PersonCropImages

class ServiceDatabasePersonCropImages:
    @staticmethod
    def insert(person_crop_image: EntityPersonCropImage):
        person_crop_image_model = PersonCropImages(
            id=person_crop_image.id,
            person_id=person_crop_image.person_id,
            camera_id=person_crop_image.camera_id,
            view_id=person_crop_image.view_id,
            image_path=person_crop_image.filepath,
            timestamp=person_crop_image.timestamp,
        )
        with ServiceDatabaseSession.get_session() as session:
            session.add(person_crop_image_model)
            session.commit()
