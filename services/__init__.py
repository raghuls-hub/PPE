"""
services/__init__.py
"""
from services.ppe_service    import PPEService,  PPEDetection
from services.fire_service   import FireService,  FireDetection
from services.fall_service   import FallService,  FallDetection
from services.face_service   import FaceService
from services.attendance_service import AttendanceService
from services.admin_service  import AdminService

__all__ = [
    "PPEService",  "PPEDetection",
    "FireService", "FireDetection",
    "FallService", "FallDetection",
    "FaceService",
    "AttendanceService",
    "AdminService",
]
