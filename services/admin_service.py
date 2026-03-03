"""
Admin Service — manages PPE configuration stored in MongoDB.
"""

from datetime import datetime
from typing import List, Optional
from pymongo.database import Database

from services.ppe_service import AVAILABLE_PPE_OPTIONS


class AdminService:
    """Manages admin-configurable PPE requirements in MongoDB."""

    COLLECTION = "ppe_config"

    def __init__(self, db: Database):
        self.db = db

    # ──────────────────────────── Get Config ──────────────────────────────────

    def get_required_ppe(self) -> List[str]:
        """
        Retrieve current required PPE list from DB.
        Returns empty list if not configured (means no PPE required).
        """
        doc = self.db[self.COLLECTION].find_one({}, sort=[("updated_at", -1)])
        if doc:
            return doc.get("required_ppe", [])
        # Default: require Hardhat + Safety Vest
        return ["Hardhat", "Safety Vest"]

    # ──────────────────────────── Set Config ──────────────────────────────────

    def set_required_ppe(self, ppe_list: List[str], updated_by: str = "admin") -> bool:
        """
        Upsert the PPE configuration in MongoDB.

        Args:
            ppe_list:   List of required PPE class names.
            updated_by: Identifier of who made the change.

        Returns:
            True on success, False on failure.
        """
        # Validate entries
        valid = [p for p in ppe_list if p in AVAILABLE_PPE_OPTIONS]
        invalid = [p for p in ppe_list if p not in AVAILABLE_PPE_OPTIONS]
        if invalid:
            print(f"⚠️  Ignored invalid PPE items: {invalid}")

        try:
            doc = {
                "required_ppe": valid,
                "updated_at": datetime.utcnow(),
                "updated_by": updated_by,
            }
            self.db[self.COLLECTION].replace_one({}, doc, upsert=True)
            print(f"✅ PPE config saved: {valid}")
            return True
        except Exception as e:
            print(f"❌ Error saving PPE config: {e}")
            return False

    # ──────────────────────────── Helpers ────────────────────────────────────

    def get_ppe_options(self) -> List[str]:
        """Return all configurable PPE items."""
        return list(AVAILABLE_PPE_OPTIONS)

    def display_current_config(self) -> None:
        """Print the current PPE configuration."""
        required = self.get_required_ppe()
        print("\n" + "="*50)
        print(" CURRENT PPE REQUIREMENTS")
        print("="*50)
        if not required:
            print("  ⚠️  No PPE required (attendance always marked)")
        else:
            for item in required:
                print(f"  ✅ {item}")
        print("="*50)
