"""
Configuration module for Smart PPE Detection & Face Recognition Attendance System
Loads from .env file if present, with sensible defaults.
"""

import os
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))


class Config:
    """Base configuration"""

    # ── MongoDB ──────────────────────────────────────────────────────────────
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    DB_NAME: str = os.getenv("DB_NAME", "ppe_attendance")

    # ── Paths ─────────────────────────────────────────────────────────────────
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH: str = os.path.join(BASE_DIR, "best.pt")
    FIRE_MODEL_PATH: str = os.path.join(BASE_DIR, "hf_firedetection.pt")
    FALL_MODEL_PATH: str = os.path.join(BASE_DIR, "fall_detection.pt")
    FACES_FOLDER: str = os.path.join(BASE_DIR, "faces")
    SNAPSHOTS_FOLDER: str = os.path.join(BASE_DIR, "snapshots")
    REPORTS_FOLDER: str = os.path.join(BASE_DIR, "reports")

    # ── Face Recognition ──────────────────────────────────────────────────────
    FACE_MATCH_THRESHOLD: float = float(os.getenv("FACE_MATCH_THRESHOLD", "120"))  # LBPH confidence (lower = better; 120 is more forgiving for lighting variation)
    FACE_SCALE_FACTOR: float = 1.1
    FACE_MIN_NEIGHBORS: int = 3      # Lowered from 5 → detects faces more readily
    FACE_MIN_SIZE: tuple = (40, 40)  # Smaller min size to catch faces farther from camera

    # ── PPE Detection ─────────────────────────────────────────────────────────
    PPE_CONFIDENCE_THRESHOLD: float = float(os.getenv("PPE_CONFIDENCE_THRESHOLD", "0.45"))
    PPE_IOU_THRESHOLD: float = float(os.getenv("PPE_IOU_THRESHOLD", "0.45"))

    # ── Fire Detection ────────────────────────────────────────────────────────
    FIRE_DETECTION_ENABLED: bool = os.getenv("FIRE_DETECTION_ENABLED", "true").lower() == "true"
    FIRE_CONFIDENCE_THRESHOLD: float = float(os.getenv("FIRE_CONFIDENCE_THRESHOLD", "0.40"))
    FIRE_IOU_THRESHOLD: float = float(os.getenv("FIRE_IOU_THRESHOLD", "0.45"))
    FIRE_ALERT_COOLDOWN_SEC: float = float(os.getenv("FIRE_ALERT_COOLDOWN_SEC", "5.0"))

    # ── Fall Detection ────────────────────────────────────────────────────────
    FALL_DETECTION_ENABLED: bool = os.getenv("FALL_DETECTION_ENABLED", "true").lower() == "true"
    FALL_CONFIDENCE_THRESHOLD: float = float(os.getenv("FALL_CONFIDENCE_THRESHOLD", "0.50"))
    FALL_IOU_THRESHOLD: float = float(os.getenv("FALL_IOU_THRESHOLD", "0.45"))
    FALL_FRAME_THRESHOLD: int = int(os.getenv("FALL_FRAME_THRESHOLD", "10"))    # consecutive frames
    FALL_ALERT_COOLDOWN_FRAMES: int = int(os.getenv("FALL_ALERT_COOLDOWN_FRAMES", "30"))  # frames alert stays visible

    # ── Attendance ─────────────────────────────────────────────────────────────
    ATTENDANCE_COOLDOWN_MINUTES: int = int(os.getenv("ATTENDANCE_COOLDOWN_MINUTES", "5"))

    # ── Camera ────────────────────────────────────────────────────────────────
    CAMERA_INDEX: int = int(os.getenv("CAMERA_INDEX", "0"))
    FRAME_WIDTH: int = int(os.getenv("FRAME_WIDTH", "1280"))
    FRAME_HEIGHT: int = int(os.getenv("FRAME_HEIGHT", "720"))

    # ── YOLO Classes (matches best.pt) ───────────────────────────────────────
    YOLO_CLASSES: list = [
        'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask',
        'NO-Safety Vest', 'Person', 'Safety Cone',
        'Safety Vest', 'machinery', 'vehicle'
    ]

    # For each required PPE, its "negative" counterpart that must NOT be detected
    PPE_NEGATIVE_MAP: dict = {
        "Hardhat": "NO-Hardhat",
        "Mask": "NO-Mask",
        "Safety Vest": "NO-Safety Vest",
    }

    # ── Audio (Bonus) ─────────────────────────────────────────────────────────
    SOUND_ENABLED: bool = os.getenv("SOUND_ENABLED", "true").lower() == "true"
    SOUND_FILE: str = os.path.join(BASE_DIR, "assets", "attend.wav")

    # ── Features ───────────────────────────────────────────────────────────────
    SAVE_SNAPSHOTS: bool = os.getenv("SAVE_SNAPSHOTS", "true").lower() == "true"
    PROCESS_EVERY_N_FRAMES: int = int(os.getenv("PROCESS_EVERY_N_FRAMES", "3"))  # For performance


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


_CONFIGS = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
}


def get_config(env: str = 'development') -> Config:
    """Return the configuration class for the given environment."""
    return _CONFIGS.get(env, DevelopmentConfig)()


def ensure_directories(cfg: Config) -> None:
    """Create required directories if they don't exist."""
    for folder in [cfg.FACES_FOLDER, cfg.SNAPSHOTS_FOLDER, cfg.REPORTS_FOLDER]:
        os.makedirs(folder, exist_ok=True)
