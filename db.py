"""
db.py — MongoDB Atlas Data Layer for CCTV AI Admin Dashboard.
All imports LOCAL — references local config.py only.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pymongo import MongoClient, DESCENDING
from pymongo.database import Database
from bson import ObjectId

import config as cfg

# ── Singleton connection ───────────────────────────────────────────────────────

_client: Optional[MongoClient] = None
_db:     Optional[Database]    = None


def get_db() -> Database:
    global _client, _db
    if _db is None:
        _client = MongoClient(cfg.MONGODB_URI, serverSelectionTimeoutMS=5000)
        _db      = _client[cfg.DB_NAME]
    return _db


# ── Cameras ────────────────────────────────────────────────────────────────────

def add_camera(name: str, stream_url: str, location: str, cam_type: str) -> str:
    doc = {"name": name, "stream_url": stream_url, "location": location,
           "type": cam_type, "created_at": datetime.utcnow()}
    return str(get_db().cameras.insert_one(doc).inserted_id)


def get_cameras(cam_type: Optional[str] = None) -> List[Dict]:
    q = {"type": cam_type} if cam_type else {}
    cams = list(get_db().cameras.find(q).sort("created_at", DESCENDING))
    for c in cams:
        c["_id"] = str(c["_id"])
    return cams


def delete_camera(camera_id: str) -> bool:
    res = get_db().cameras.delete_one({"_id": ObjectId(camera_id)})
    return res.deleted_count > 0


# ── Workers ────────────────────────────────────────────────────────────────────

def add_worker(emp_id: str, name: str, department: str) -> Optional[str]:
    db = get_db()
    if db.workers.find_one({"emp_id": emp_id}):
        return None
    doc = {
        "emp_id": emp_id, "name": name, "department": department,
        "face_encoding": None, "face_encodings_all": [],
        "face_image_path": None, "is_active": True,
        "created_at": datetime.utcnow(),
    }
    return str(db.workers.insert_one(doc).inserted_id)


def get_workers(active_only: bool = True) -> List[Dict]:
    q = {"is_active": True} if active_only else {}
    workers = list(get_db().workers.find(q).sort("name", 1))
    for w in workers:
        w["_id"] = str(w["_id"])
    return workers


def delete_worker(worker_id: str) -> bool:
    res = get_db().workers.update_one(
        {"_id": ObjectId(worker_id)},
        {"$set": {"is_active": False}},
    )
    return res.modified_count > 0


def save_worker_face(worker_id: str, face_encoding: list,
                     all_encodings: list, image_path: str) -> bool:
    res = get_db().workers.update_one(
        {"_id": ObjectId(worker_id)},
        {"$set": {
            "face_encoding":      face_encoding,
            "face_encodings_all": all_encodings,
            "face_image_path":    image_path,
            "updated_at":         datetime.utcnow(),
        }},
    )
    return res.modified_count > 0


def get_worker_encodings() -> List[Dict]:
    """Returns workers with a face encoding (for LBPH training)."""
    workers = list(get_db().workers.find(
        {"is_active": True, "face_encoding": {"$ne": None}},
        {"emp_id": 1, "name": 1, "department": 1,
         "face_encoding": 1, "face_encodings_all": 1},
    ))
    for w in workers:
        w["_id"] = str(w["_id"])
    return workers


# ── Attendance ─────────────────────────────────────────────────────────────────

def is_recently_marked(emp_id: str, cooldown_minutes: int = cfg.ATTENDANCE_COOLDOWN_MINUTES) -> bool:
    cutoff = datetime.utcnow() - timedelta(minutes=cooldown_minutes)
    return get_db().attendance.find_one({
        "emp_id": emp_id,
        "timestamp": {"$gte": cutoff},
    }) is not None


def mark_attendance(
    emp_id: str, name: str, ppe_ok: bool,
    detected_ppe: list, missing_ppe: list,
    camera_id: str = "", confidence: float = 0.0,
) -> Optional[str]:
    if is_recently_marked(emp_id):
        return None
    doc = {
        "emp_id":         emp_id,
        "employee_id":    emp_id,
        "name":           name,
        "timestamp":      datetime.utcnow(),
        "ppe_ok":         ppe_ok,
        "ppe_verified":   ppe_ok,
        "detected_ppe":   detected_ppe,
        "missing_ppe":    missing_ppe,
        "camera_id":      camera_id,
        "camera_source":  camera_id,
        "confidence":     round(confidence, 4),
        "confidence_score": round(confidence, 4),
    }
    return str(get_db().attendance.insert_one(doc).inserted_id)


def get_today_attendance() -> List[Dict]:
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    recs  = list(get_db().attendance.find(
        {"timestamp": {"$gte": today}}, {"_id": 0}
    ).sort("timestamp", DESCENDING))
    return recs


def get_attendance_by_date(date: datetime) -> List[Dict]:
    start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end   = start + timedelta(days=1)
    return list(get_db().attendance.find(
        {"timestamp": {"$gte": start, "$lt": end}}, {"_id": 0}
    ).sort("timestamp", DESCENDING))


# ── PPE Config ─────────────────────────────────────────────────────────────────

def get_required_ppe() -> List[str]:
    doc = get_db().ppe_config.find_one({}, sort=[("updated_at", DESCENDING)])
    return doc.get("required_ppe", []) if doc else ["Hardhat", "Safety Vest"]


def set_required_ppe(ppe_list: List[str], updated_by: str = "admin") -> bool:
    valid = [p for p in ppe_list if p in cfg.AVAILABLE_PPE_OPTIONS]
    try:
        get_db().ppe_config.replace_one(
            {},
            {"required_ppe": valid, "updated_at": datetime.utcnow(), "updated_by": updated_by},
            upsert=True,
        )
        return True
    except Exception as e:
        print(f"❌ PPE config save error: {e}")
        return False


# ── Alerts ─────────────────────────────────────────────────────────────────────

def push_alert(camera_id: str, alert_type: str, message: str) -> str:
    doc = {
        "camera_id":   camera_id,
        "alert_type":  alert_type,
        "message":     message,
        "timestamp":   datetime.utcnow(),
        "acknowledged": False,
    }
    return str(get_db().alerts.insert_one(doc).inserted_id)


def get_recent_alerts(limit: int = 50) -> List[Dict]:
    alerts = list(get_db().alerts.find({}).sort("timestamp", DESCENDING).limit(limit))
    for a in alerts:
        a["_id"] = str(a["_id"])
    return alerts


def acknowledge_alert(alert_id: str) -> bool:
    res = get_db().alerts.update_one(
        {"_id": ObjectId(alert_id)},
        {"$set": {"acknowledged": True}},
    )
    return res.modified_count > 0
