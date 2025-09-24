from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class PersonCropImages(Base):
    __tablename__ = "person_crop_images"

    id = Column(String, primary_key=True)
    person_id = Column(Integer)
    camera_id = Column(Integer)
    view_id = Column(Integer)
    image_path = Column(String)
    timestamp = Column(DateTime)

    def __repr__(self):
        return "<PersonCropImages(id={}, person_id={}, camera_id={}, view_id={}, image_path={}, timestamp={})>" \
            .format(self.id, self.person_id, self.camera_id, self.view_id, self.image_path, self.timestamp)
