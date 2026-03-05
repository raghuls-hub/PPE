"""
Attendance Service — marks attendance with cooldown (MongoDB).
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pymongo.database import Database


class AttendanceService:
    """Handles attendance operations with MongoDB backend."""

    def __init__(self, db: Database, cooldown_minutes: int = 5):
        self.db = db
        self.cooldown_minutes = cooldown_minutes

    def is_recently_marked(self, employee_id: str) -> bool:
        cutoff = datetime.utcnow() - timedelta(minutes=self.cooldown_minutes)
        return self.db.attendance.find_one({
            "emp_id": employee_id,
            "timestamp": {"$gte": cutoff},
        }) is not None

    def mark_attendance(
        self,
        employee_id: str,
        name: str,
        ppe_verified: bool,
        detected_ppe: List[str],
        missing_ppe: List[str],
        camera_source: str = "Live",
        confidence_score: float = 0.0,
        snapshot_path: Optional[str] = None,
    ) -> Optional[str]:
        try:
            doc = {
                "emp_id":           employee_id,
                "employee_id":      employee_id,   # backward compat
                "name":             name,
                "timestamp":        datetime.utcnow(),
                "ppe_verified":     ppe_verified,
                "ppe_ok":           ppe_verified,
                "detected_ppe":     detected_ppe,
                "missing_ppe":      missing_ppe,
                "camera_source":    camera_source,
                "camera_id":        camera_source,
                "confidence_score": round(confidence_score, 4),
                "confidence":       round(confidence_score, 4),
                "snapshot_path":    snapshot_path,
            }
            result = self.db.attendance.insert_one(doc)
            return str(result.inserted_id)
        except Exception as e:
            print(f"❌ Attendance insert error: {e}")
            return None

    def get_daily_report(self, date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        if date is None:
            date = datetime.utcnow()
        start = datetime(date.year, date.month, date.day, 0, 0, 0)
        end   = start + timedelta(days=1)
        return list(self.db.attendance.find(
            {"timestamp": {"$gte": start, "$lt": end}},
            {"_id": 0}
        ).sort("timestamp", 1))

    def get_all_today(self) -> List[Dict[str, Any]]:
        return self.get_daily_report()
