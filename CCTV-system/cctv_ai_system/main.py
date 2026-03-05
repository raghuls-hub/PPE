from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import models
import schemas
from database import engine, SessionLocal, Base

Base.metadata.create_all(bind=engine)

app = FastAPI(title="CCTV AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Serve dashboard static files ──────────────────────────────────────────────
app.mount("/dashboard", StaticFiles(directory="dashboard", html=True), name="dashboard")

@app.get("/")
def root():
    return FileResponse("dashboard/index.html")


# ── Camera CRUD ───────────────────────────────────────────────────────────────
@app.post("/cameras", response_model=schemas.CameraOut)
def add_camera(camera: schemas.CameraCreate, db: Session = Depends(get_db)):
    new_camera = models.Camera(
        name=camera.name,
        stream_url=camera.stream_url,
        location=camera.location,
    )
    db.add(new_camera)
    db.commit()
    db.refresh(new_camera)
    return new_camera


@app.get("/cameras", response_model=list[schemas.CameraOut])
def get_cameras(db: Session = Depends(get_db)):
    return db.query(models.Camera).all()


@app.delete("/cameras/{camera_id}")
def delete_camera(camera_id: int, db: Session = Depends(get_db)):
    camera = db.query(models.Camera).filter(models.Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    db.delete(camera)
    db.commit()
    return {"message": "Camera deleted"}
