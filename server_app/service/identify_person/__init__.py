from PIL import Image
from service.identify_person.assign_id import ServiceIdentifyPersonAssignId
from service.identify_person.clip_reid_model import \
    ServiceIdentifyPersonClipReIDModel


class ServiceIdentifyPerson:
    clip_reid_model = ServiceIdentifyPersonClipReIDModel()
    assign_id = ServiceIdentifyPersonAssignId()

    def Identify(self, image: Image, camera_id: int = 0, view_id: int = 0) -> int:
        query_feature = self.clip_reid_model.extract_feature(image, camera_id, view_id)
        return self.assign_id.assign(query_feature)

