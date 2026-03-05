"""
MongoDB connection manager (singleton pattern).
"""

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.database import Database
from typing import Optional


class DBConnection:
    """Singleton MongoDB connection manager."""

    _client: Optional[MongoClient] = None
    _db:     Optional[Database]    = None

    def connect(self, uri: str, db_name: str) -> Database:
        if self._client is None:
            try:
                self._client = MongoClient(uri, serverSelectionTimeoutMS=5000)
                self._client.admin.command("ping")
                print(f"✅ Connected to MongoDB: {uri}")
            except Exception as e:
                print(f"❌ MongoDB connection failed: {e}")
                raise
        self._db = self._client[db_name]
        self._ensure_indexes()
        return self._db

    def _ensure_indexes(self) -> None:
        if self._db is None:
            return
        try:
            self._db.attendance.create_index([("emp_id", ASCENDING), ("timestamp", DESCENDING)])
            self._db.workers.create_index([("emp_id", ASCENDING)], unique=True, background=True)
            self._db.ppe_config.create_index([("updated_at", DESCENDING)])
            self._db.cameras.create_index([("type", ASCENDING)])
        except Exception as e:
            print(f"⚠️  Index creation: {e}")

    def disconnect(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
            self._db = None

    @property
    def db(self) -> Optional[Database]:
        return self._db


db_connection = DBConnection()
