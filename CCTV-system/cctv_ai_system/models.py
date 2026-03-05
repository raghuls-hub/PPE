from sqlalchemy import Column, Integer, String
from database import Base


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    stream_url = Column(String)
    location = Column(String)
