"""
Report Generator — produces daily attendance CSV reports.
"""

import csv
import os
from datetime import datetime
from typing import List, Dict, Any, Optional


class ReportGenerator:
    """Generates daily attendance reports in CSV format."""

    def __init__(self, reports_folder: str):
        self.reports_folder = reports_folder
        os.makedirs(reports_folder, exist_ok=True)

    def generate_daily_csv(
        self,
        records: List[Dict[str, Any]],
        date: Optional[datetime] = None,
    ) -> str:
        """
        Write attendance records to a dated CSV file.

        Args:
            records:  List of attendance dicts from MongoDB.
            date:     Date for the report (default: today).

        Returns:
            Absolute path to the generated CSV file.
        """
        if date is None:
            date = datetime.utcnow()

        filename = f"attendance_{date.strftime('%Y-%m-%d')}.csv"
        filepath = os.path.join(self.reports_folder, filename)

        fieldnames = [
            "employee_id", "name", "timestamp", "ppe_verified",
            "detected_ppe", "missing_ppe", "camera_source", "confidence_score",
        ]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in records:
                writer.writerow({
                    "employee_id": rec.get("employee_id", ""),
                    "name": rec.get("name", ""),
                    "timestamp": rec.get("timestamp", ""),
                    "ppe_verified": rec.get("ppe_verified", False),
                    "detected_ppe": ", ".join(rec.get("detected_ppe", [])),
                    "missing_ppe": ", ".join(rec.get("missing_ppe", [])),
                    "camera_source": rec.get("camera_source", ""),
                    "confidence_score": rec.get("confidence_score", 0.0),
                })

        print(f"📄 Report saved: {filepath}  ({len(records)} records)")
        return filepath

