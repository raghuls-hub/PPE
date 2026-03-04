"""Services package for PPE Attendance System."""
from services.fire_service import FireService, FireDetection
from services.fall_service import FallService, FallDetection

__all__ = [
    "FireService", "FireDetection",
    "FallService", "FallDetection",
]
