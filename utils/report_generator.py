"""
Report Generator — produces daily attendance CSV reports.
"""

import csv
import os
from datetime import datetime
from typing import List, Dict, Any, Optional


class ReportGenerator:
    """Generates daily attendance CSV reports."""

    def __init__(self, reports_folder: str):
        self.reports_folder = reports_folder
        os.makedirs(reports_folder, exist_ok=True)

    def generate_daily_csv(
        self,
        records: List[Dict[str, Any]],
        date: Optional[datetime] = None,
    ) -> str:
        if date is None:
            date = datetime.utcnow()
        filename = f"attendance_{date.strftime('%Y-%m-%d')}.csv"
        filepath = os.path.join(self.reports_folder, filename)

        fieldnames = [
            "emp_id", "name", "timestamp", "ppe_ok",
            "detected_ppe", "missing_ppe", "camera_id", "confidence",
        ]
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in records:
                writer.writerow({
                    "emp_id":        rec.get("emp_id", rec.get("employee_id", "")),
                    "name":          rec.get("name", ""),
                    "timestamp":     rec.get("timestamp", ""),
                    "ppe_ok":        rec.get("ppe_ok", rec.get("ppe_verified", False)),
                    "detected_ppe":  ", ".join(rec.get("detected_ppe", [])),
                    "missing_ppe":   ", ".join(rec.get("missing_ppe", [])),
                    "camera_id":     rec.get("camera_id", rec.get("camera_source", "")),
                    "confidence":    rec.get("confidence", rec.get("confidence_score", 0.0)),
                })
        print(f"📄 Report saved: {filepath}  ({len(records)} records)")
        return filepath
