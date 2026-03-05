"""
Admin Service — manages PPE configuration in MongoDB.
"""

from datetime import datetime
from typing import List
from pymongo.database import Database

from config import AVAILABLE_PPE_OPTIONS


class AdminService:
    """Manages admin-configurable PPE requirements."""

    COLLECTION = "ppe_config"

    def __init__(self, db: Database):
        self.db = db

    def get_required_ppe(self) -> List[str]:
        doc = self.db[self.COLLECTION].find_one({}, sort=[("updated_at", -1)])
        if doc:
            return doc.get("required_ppe", [])
        return ["Hardhat", "Safety Vest"]

    def set_required_ppe(self, ppe_list: List[str], updated_by: str = "admin") -> bool:
        valid = [p for p in ppe_list if p in AVAILABLE_PPE_OPTIONS]
        try:
            self.db[self.COLLECTION].replace_one(
                {},
                {"required_ppe": valid, "updated_at": datetime.utcnow(), "updated_by": updated_by},
                upsert=True,
            )
            return True
        except Exception as e:
            print(f"❌ Error saving PPE config: {e}")
            return False

    def get_ppe_options(self) -> List[str]:
        return list(AVAILABLE_PPE_OPTIONS)
