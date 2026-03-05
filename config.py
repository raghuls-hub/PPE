"""
config.py — Central Configuration for CCTV AI Admin Dashboard
=============================================================
All paths, thresholds, and settings in one place.
"""

import os
from dotenv import load_dotenv

# Load .env from this project root
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ── Base directories ──────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR       = BASE_DIR                                      # models sit here
FACES_DIR        = os.path.join(BASE_DIR, "faces")
SNAPSHOTS_DIR    = os.path.join(BASE_DIR, "snapshots")
REPORTS_DIR      = os.path.join(BASE_DIR, "reports")

for _d in [FACES_DIR, SNAPSHOTS_DIR, REPORTS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Model Paths ───────────────────────────────────────────────────────────────
PPE_MODEL_PATH   = os.path.join(MODELS_DIR, "PPE_detection.pt")
FALL_MODEL_PATH  = os.path.join(MODELS_DIR, "fall_detection.pt")
FIRE_MODEL_PATH  = os.path.join(MODELS_DIR, "fire_detection.pt")

# ── MongoDB ────────────────────────────────────────────────────────────────────
MONGODB_URI  = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME      = os.getenv("DB_NAME", "ppe_attendance")

# ── Detection Thresholds ──────────────────────────────────────────────────────
PPE_CONFIDENCE_THRESHOLD  = float(os.getenv("PPE_CONFIDENCE_THRESHOLD",  "0.45"))
PPE_IOU_THRESHOLD         = float(os.getenv("PPE_IOU_THRESHOLD",         "0.45"))
FIRE_CONFIDENCE_THRESHOLD = float(os.getenv("FIRE_CONFIDENCE_THRESHOLD", "0.40"))
FIRE_IOU_THRESHOLD        = float(os.getenv("FIRE_IOU_THRESHOLD",        "0.45"))
FALL_CONFIDENCE_THRESHOLD = float(os.getenv("FALL_CONFIDENCE_THRESHOLD", "0.50"))
FALL_IOU_THRESHOLD        = float(os.getenv("FALL_IOU_THRESHOLD",        "0.45"))

# ── Alert frame thresholds (consecutive frames before alert triggers) ─────────
PPE_VIOLATION_THRESHOLD = int(os.getenv("PPE_VIOLATION_THRESHOLD", "15"))
FIRE_FRAME_THRESHOLD    = int(os.getenv("FIRE_FRAME_THRESHOLD",    "5"))
FALL_FRAME_THRESHOLD    = int(os.getenv("FALL_FRAME_THRESHOLD",    "10"))
FALL_ALERT_COOLDOWN     = int(os.getenv("FALL_ALERT_COOLDOWN",     "30"))

# ── Face Recognition ──────────────────────────────────────────────────────────
FACE_MATCH_THRESHOLD  = float(os.getenv("FACE_MATCH_THRESHOLD", "120"))
FACE_SCALE_FACTOR     = 1.1
FACE_MIN_NEIGHBORS    = 3
FACE_MIN_SIZE         = (40, 40)

# ── Attendance ────────────────────────────────────────────────────────────────
ATTENDANCE_COOLDOWN_MINUTES = int(os.getenv("ATTENDANCE_COOLDOWN_MINUTES", "5"))
PROCESS_EVERY_N_FRAMES      = int(os.getenv("PROCESS_EVERY_N_FRAMES",      "3"))

# ── PPE class definitions ─────────────────────────────────────────────────────
AVAILABLE_PPE_OPTIONS = ["Hardhat", "Mask", "Safety Vest"]

PPE_NEGATIVE_MAP = {
    "Hardhat":     "NO-Hardhat",
    "Mask":        "NO-Mask",
    "Safety Vest": "NO-Safety Vest",
}

# Classes the YOLO PPE model outputs
YOLO_CLASSES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask",
    "NO-Safety Vest", "Person", "Safety Cone",
    "Safety Vest", "machinery", "vehicle",
]
