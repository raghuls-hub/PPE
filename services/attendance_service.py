"""
Attendance Service — marks attendance with cooldown and generates daily reports.
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pymongo.database import Database


class AttendanceService:
    """Handles attendance operations with MongoDB backend."""

    def __init__(self, db: Database, cooldown_minutes: int = 5):
        self.db = db
        self.cooldown_minutes = cooldown_minutes

    # ──────────────────────────── Cooldown Check ──────────────────────────────

    def is_recently_marked(self, employee_id: str) -> bool:
        """
        Returns True if attendance was marked within the cooldown window.
        Prevents duplicate marking within the configured interval.
        """
        cutoff = datetime.utcnow() - timedelta(minutes=self.cooldown_minutes)
        record = self.db.attendance.find_one({
            "employee_id": employee_id,
            "timestamp": {"$gte": cutoff},
        })
        return record is not None

    # ──────────────────────────── Mark Attendance ─────────────────────────────

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
        """
        Insert an attendance record into MongoDB.

        Returns:
            Inserted document ID as string, or None on failure.
        """
        try:
            doc = {
                "employee_id": employee_id,
                "name": name,
                "timestamp": datetime.utcnow(),
                "ppe_verified": ppe_verified,
                "detected_ppe": detected_ppe,
                "missing_ppe": missing_ppe,
                "camera_source": camera_source,
                "confidence_score": round(confidence_score, 4),
                "snapshot_path": snapshot_path,
            }
            result = self.db.attendance.insert_one(doc)
            return str(result.inserted_id)
        except Exception as e:
            print(f"❌ Attendance insert error: {e}")
            return None

    # ──────────────────────────── Reporting ───────────────────────────────────

    def get_daily_report(self, date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Retrieve all attendance records for a given date (default: today UTC).
        """
        if date is None:
            date = datetime.utcnow()

        start = datetime(date.year, date.month, date.day, 0, 0, 0)
        end = start + timedelta(days=1)

        records = list(self.db.attendance.find(
            {"timestamp": {"$gte": start, "$lt": end}},
            {"_id": 0}
        ).sort("timestamp", 1))
        return records

    def get_all_today(self) -> List[Dict[str, Any]]:
        """Convenience wrapper for today's report."""
        return self.get_daily_report()
