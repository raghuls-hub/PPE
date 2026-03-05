from pydantic import BaseModel


class CameraCreate(BaseModel):
    name: str
    stream_url: str
    location: str


class CameraOut(BaseModel):
    id: int
    name: str
    stream_url: str
    location: str

    model_config = {"from_attributes": True}
