from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


class ServiceDatabaseSession:
    _engine = create_engine("sqlite:///.db")
    _Session = sessionmaker(bind=_engine)

    @staticmethod
    def get_session():
        return ServiceDatabaseSession._Session()
