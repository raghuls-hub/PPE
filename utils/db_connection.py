"""
MongoDB connection manager (singleton pattern).
"""

from pymongo import MongoClient
from pymongo.database import Database
from typing import Optional


class DBConnection:
    """Singleton MongoDB connection manager."""

    _client: Optional[MongoClient] = None
    _db: Optional[Database] = None

    def connect(self, uri: str, db_name: str) -> Database:
        """
        Connect to MongoDB and return the database object.
        Reuses existing connection if already established.
        """
        if self._client is None:
            try:
                self._client = MongoClient(uri, serverSelectionTimeoutMS=5000)
                # Verify connection
                self._client.admin.command('ping')
                print(f"✅ Connected to MongoDB: {uri}")
            except Exception as e:
                print(f"❌ MongoDB connection failed: {e}")
                raise

        self._db = self._client[db_name]
        self._ensure_indexes()
        return self._db

    def _ensure_indexes(self) -> None:
        """Create indexes for efficient queries."""
        if self._db is None:
            return
        try:
            # Attendance: fast lookup by employee + timestamp
            self._db.attendance.create_index(
                [("employee_id", 1), ("timestamp", -1)]
            )
            # Employees: unique employee_id
            self._db.employees.create_index(
                [("employee_id", 1)], unique=True
            )
            # PPE Config: single doc
            self._db.ppe_config.create_index(
                [("updated_at", -1)]
            )
        except Exception as e:
            print(f"⚠️  Index creation warning: {e}")

    def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            print("🔌 MongoDB disconnected.")

    @property
    def db(self) -> Optional[Database]:
        return self._db


# Global singleton instance
db_connection = DBConnection()
