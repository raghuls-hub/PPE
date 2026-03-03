"""
Live Face Recognition + PPE Detection + Attendance System
=========================================================
Core application class. Integrates:
  - LBPH Face Recognition (FaceService)
  - YOLO PPE Detection (PPEService)
  - Attendance Marking with cooldown (AttendanceService)
  - Admin-configurable PPE requirements (AdminService)
  - Live webcam and video file modes
  - Employee registration via live webcam
"""

import cv2
import numpy as np
import os
import sys
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, get_config, ensure_directories
from utils.db_connection import db_connection
from utils.sound_notification import play_attendance_sound
from utils.report_generator import ReportGenerator
from services.face_service import FaceService
from services.ppe_service import PPEService, PPEDetection
from services.attendance_service import AttendanceService
from services.admin_service import AdminService


# ── Color constants (BGR) ────────────────────────────────────────────────────
COLOR_GREEN  = (0, 220, 0)      # Recognized + PPE OK
COLOR_RED    = (0, 0, 220)      # Recognized + PPE FAILED
COLOR_YELLOW = (0, 200, 255)    # Unknown person
COLOR_GRAY   = (140, 140, 140)  # Encoding failed
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Number of face samples captured during registration (more = better LBPH accuracy)
NUM_REGISTRATION_SAMPLES = 5


class LiveFaceRecognition:
    """
    Main application class for Smart PPE Detection & Face Recognition Attendance.
    
    Supports:
      - Live webcam feed  (mode='live')
      - Video file input  (mode='video')
      - Employee registration via webcam
    """

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or get_config('development')
        ensure_directories(self.cfg)

        # ── Database ──────────────────────────────────────────────────────────
        self.db = db_connection.connect(self.cfg.MONGODB_URI, self.cfg.DB_NAME)

        # ── Services ──────────────────────────────────────────────────────────
        self.face_service = FaceService()
        self.ppe_service = PPEService(
            model_path=self.cfg.MODEL_PATH,
            confidence_threshold=self.cfg.PPE_CONFIDENCE_THRESHOLD,
            iou_threshold=self.cfg.PPE_IOU_THRESHOLD,
        )
        self.attendance_service = AttendanceService(
            self.db, self.cfg.ATTENDANCE_COOLDOWN_MINUTES
        )
        self.admin_service = AdminService(self.db)
        self.report_generator = ReportGenerator(self.cfg.REPORTS_FOLDER)

        # ── Load known employee faces ─────────────────────────────────────────
        self._load_employees()

        # ── State ─────────────────────────────────────────────────────────────
        self._frame_counter: int = 0
        self._last_ppe_results: List[PPEDetection] = []

        print("✅ Smart PPE Attendance System initialized")
        print(f"   Model  : {self.cfg.MODEL_PATH}")
        print(f"   DB     : {self.cfg.MONGODB_URI}/{self.cfg.DB_NAME}")
        print(f"   Cooldown: {self.cfg.ATTENDANCE_COOLDOWN_MINUTES} min")

    # ──────────────────────────── Data Loading ────────────────────────────────

    def _load_employees(self) -> None:
        """Load all active employee face encodings from MongoDB."""
        try:
            employees = list(self.db.employees.find({'is_active': True}))
            employee_data = [
                {
                    'employee_id': emp['employee_id'],
                    'name': emp['name'],
                    'department': emp.get('department', ''),
                    'face_encoding': emp['face_encoding'],
                    'face_encodings_all': emp.get('face_encodings_all'),  # Multi-sample (new registrations)
                }
                for emp in employees
                if emp.get('face_encoding')
            ]
            self.face_service.load_known_faces(employee_data)
        except Exception as e:
            print(f"❌ Error loading employees: {e}")

    # ──────────────────────────── PPE Delegation ──────────────────────────────

    def detect_ppe(self, frame: np.ndarray) -> List[PPEDetection]:
        """Run YOLO PPE detection on a frame."""
        return self.ppe_service.detect_ppe(frame)

    def verify_ppe(
        self, detected_classes: List[str], required_ppe: List[str]
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Validate PPE compliance.
        Returns (is_compliant, missing_list, present_list).
        """
        return self.ppe_service.verify_ppe(detected_classes, required_ppe)

    # ──────────────────────────── Attendance ─────────────────────────────────

    def mark_attendance(
        self,
        employee_id: str,
        name: str,
        ppe_verified: bool,
        detected_ppe: List[str],
        missing_ppe: List[str],
        frame: Optional[np.ndarray] = None,
        source: str = "Live",
        confidence: float = 0.0,
    ) -> bool:
        """
        Mark attendance if:
          1. PPE is compliant
          2. Not within cooldown window

        Returns True if attendance was successfully marked.
        """
        if not ppe_verified:
            return False

        if self.attendance_service.is_recently_marked(employee_id):
            return False  # Cooldown active — silent skip

        # Save snapshot
        snapshot_path: Optional[str] = None
        if self.cfg.SAVE_SNAPSHOTS and frame is not None:
            snap_name = f"{employee_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            snapshot_path = os.path.join(self.cfg.SNAPSHOTS_FOLDER, snap_name)
            cv2.imwrite(snapshot_path, frame)

        # Insert record
        record_id = self.attendance_service.mark_attendance(
            employee_id=employee_id,
            name=name,
            ppe_verified=ppe_verified,
            detected_ppe=detected_ppe,
            missing_ppe=missing_ppe,
            camera_source=source,
            confidence_score=confidence,
            snapshot_path=snapshot_path,
        )

        if record_id:
            print(f"✅ ATTENDANCE MARKED → {name} ({employee_id}) | PPE: {detected_ppe}")
            if self.cfg.SOUND_ENABLED:
                play_attendance_sound(self.cfg.SOUND_FILE)
            return True
        return False

    # ──────────────────────────── Frame Processing ────────────────────────────

    def process_frame(
        self,
        frame: np.ndarray,
        source: str = "Live",
    ) -> np.ndarray:
        """
        Full pipeline for a single frame:
          1. PPE Detection (YOLO) — runs every N frames for performance
          2. Face Detection
          3. Per-face: extract encoding → identify → verify PPE → mark attendance
          4. Draw overlays

        Returns:
            Annotated frame.
        """
        self._frame_counter += 1

        # ── Step 1: PPE Detection (every N frames) ───────────────────────────
        if self._frame_counter % self.cfg.PROCESS_EVERY_N_FRAMES == 0:
            self._last_ppe_results = self.detect_ppe(frame)

        ppe_detections = self._last_ppe_results
        detected_classes = self.ppe_service.get_detected_class_names(ppe_detections)
        required_ppe = self.admin_service.get_required_ppe()

        # Draw only admin-selected PPE bounding boxes (+ their NO-* counterparts)
        self.ppe_service.draw_ppe_boxes(frame, ppe_detections, required_ppe=required_ppe)

        # ── Step 2: Face Detection ────────────────────────────────────────────
        faces = self.face_service.detect_faces(
            frame,
            scale_factor=self.cfg.FACE_SCALE_FACTOR,
            min_neighbors=self.cfg.FACE_MIN_NEIGHBORS,
            min_size=self.cfg.FACE_MIN_SIZE,
        )

        num_known = 0
        num_unknown = 0

        # ── Step 3: Process each face ─────────────────────────────────────────
        for face_loc in faces:
            x, y, w, h = face_loc

            # Extract face ROI for recognition
            face_roi = self.face_service.extract_face_roi(frame, face_loc)

            if face_roi is None:
                # Gray box: can't process
                cv2.rectangle(frame, (x, y), (x+w, y+h), COLOR_GRAY, 2)
                continue

            # ── Step 3a: Identify employee ────────────────────────────────────
            result = self.face_service.identify_employee(
                face_roi,
                confidence_threshold=self.cfg.FACE_MATCH_THRESHOLD,
            )

            if result is None:
                # Unknown person — print raw LBPH distance for debugging
                if self.face_service.recognizer_trained:
                    try:
                        _lbl, _conf = self.face_service.recognizer.predict(face_roi)
                        print(f"[DEBUG] Face detected but UNKNOWN — LBPH distance={_conf:.1f} (threshold={self.cfg.FACE_MATCH_THRESHOLD})")
                    except Exception:
                        pass
                num_unknown += 1
                self._draw_face_overlay(
                    frame, x, y, w, h,
                    name="UNKNOWN",
                    status_text="UNKNOWN PERSON",
                    color=COLOR_YELLOW,
                    ppe_ok=False,
                    missing=[],
                    attendance_marked=False,
                    confidence=0.0,
                )
                continue

            # ── Step 3b: PPE validation ───────────────────────────────────────
            is_compliant, missing_ppe, present_ppe = self.verify_ppe(
                detected_classes, required_ppe
            )

            emp_id = result['employee_id']
            emp_name = result['name']
            face_confidence = 1.0 - (result['raw_confidence'] / 100.0)  # Normalize to 0-1

            # ── Step 3c: Mark Attendance ──────────────────────────────────────
            attendance_marked = False
            if is_compliant:
                num_known += 1
                attendance_marked = self.mark_attendance(
                    employee_id=emp_id,
                    name=emp_name,
                    ppe_verified=True,
                    detected_ppe=present_ppe,
                    missing_ppe=[],
                    frame=frame,
                    source=source,
                    confidence=face_confidence,
                )
            else:
                num_known += 1

            # ── Step 3d: Draw face overlay ────────────────────────────────────
            color = COLOR_GREEN if is_compliant else COLOR_RED
            status = "VERIFIED" if is_compliant else "FAILED"
            self._draw_face_overlay(
                frame, x, y, w, h,
                name=emp_name,
                status_text=status,
                color=color,
                ppe_ok=is_compliant,
                missing=missing_ppe,
                attendance_marked=attendance_marked,
                confidence=face_confidence,
                employee_id=emp_id,
            )

        return frame, len(faces), num_known, num_unknown

    # ──────────────────────────── Drawing Helpers ─────────────────────────────

    def _draw_face_overlay(
        self,
        frame: np.ndarray,
        x: int, y: int, w: int, h: int,
        name: str,
        status_text: str,
        color: Tuple[int, int, int],
        ppe_ok: bool,
        missing: List[str],
        attendance_marked: bool,
        confidence: float,
        employee_id: str = "",
    ) -> None:
        """Draw face bounding box and information overlay."""

        # ── Bounding box ─────────────────────────────────────────────────────
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # ── Info lines ───────────────────────────────────────────────────────
        lines = [
            f"Employee: {name}",
            f"PPE Status: {status_text}",
            f"Missing: {', '.join(missing) if missing else 'None'}",
            f"Attendance: {'MARKED' if attendance_marked else ('NOT MARKED' if ppe_ok else 'PPE FAILED')}",
        ]
        if employee_id:
            lines.insert(1, f"ID: {employee_id}")

        line_height = 22
        panel_h = line_height * len(lines) + 12
        panel_y = max(0, y - panel_h - 4)

        # Background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, panel_y), (x + 280, panel_y + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (x, panel_y), (x + 280, panel_y + panel_h), color, 1)

        for i, line in enumerate(lines):
            text_y = panel_y + 16 + i * line_height
            txt_color = COLOR_GREEN if ppe_ok else (COLOR_RED if i == 1 else COLOR_WHITE)
            if i == 0 or (i == 1 and not employee_id):
                txt_color = COLOR_WHITE
            cv2.putText(frame, line, (x + 5, text_y), FONT, 0.48, txt_color, 1, cv2.LINE_AA)

    def _draw_info_panel(
        self,
        frame: np.ndarray,
        fps: float,
        num_faces: int,
        num_known: int,
        num_unknown: int,
        required_ppe: List[str],
    ) -> None:
        """Draw top HUD panel with stats."""
        h, w = frame.shape[:2]

        # Semi-transparent header bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 75), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Stats line
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28), FONT, 0.7, COLOR_WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Faces: {num_faces}", (130, 28), FONT, 0.7, COLOR_WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Known: {num_known}", (270, 28), FONT, 0.7, COLOR_GREEN, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Unknown: {num_unknown}", (400, 28), FONT, 0.7, COLOR_YELLOW, 2, cv2.LINE_AA)

        # Required PPE line
        ppe_str = "Required PPE: " + (", ".join(required_ppe) if required_ppe else "None")
        cv2.putText(frame, ppe_str, (10, 58), FONT, 0.55, (180, 220, 255), 1, cv2.LINE_AA)

        # Controls
        controls = "q:Quit  r:Reload  n:Register  s:Screenshot  e:Export Report"
        cv2.putText(frame, controls, (10, h - 10), FONT, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    # ──────────────────────────── Run Modes ──────────────────────────────────

    def run_live(self) -> None:
        """Start live webcam recognition loop."""
        cap = cv2.VideoCapture(self.cfg.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.FRAME_HEIGHT)

        if not cap.isOpened():
            print("❌ Could not open camera.")
            return

        self._recognition_loop(cap, source="Live")
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Camera released. Goodbye!")

    def run_video(self, video_path: str) -> None:
        """Process a video file frame-by-frame."""
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return

        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"🎬 Video: {video_path}  ({total_frames} frames @ {fps_in:.1f} fps)")

        self._recognition_loop(cap, source="Video")
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Video processing complete.")

    def _recognition_loop(self, cap: cv2.VideoCapture, source: str = "Live") -> None:
        """
        Shared loop for live and video modes.
        Handles frame capture, processing, overlay, and key controls.
        """
        WINDOW_TITLE = "Smart PPE Attendance System"
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

        fps = 0.0
        frame_count = 0
        tick_start = cv2.getTickCount()

        while True:
            ret, frame = cap.read()
            if not ret:
                if source == "Live":
                    print("❌ Failed to capture frame.")
                else:
                    print("🏁 End of video.")
                break

            # Mirror for live camera only
            if source == "Live":
                frame = cv2.flip(frame, 1)

            # ── Process & annotate ────────────────────────────────────────────
            required_ppe = self.admin_service.get_required_ppe()
            frame, num_faces, num_known, num_unknown = self.process_frame(frame, source)

            # ── FPS calculation ───────────────────────────────────────────────
            frame_count += 1
            if frame_count >= 20:
                end_tick = cv2.getTickCount()
                fps = 20 / ((end_tick - tick_start) / cv2.getTickFrequency())
                tick_start = end_tick
                frame_count = 0

            # ── Draw HUD ──────────────────────────────────────────────────────
            self._draw_info_panel(frame, fps, num_faces, num_known, num_unknown, required_ppe)

            cv2.imshow(WINDOW_TITLE, frame)

            # ── Key Handling ──────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("👋 Quitting...")
                break

            elif key == ord('r'):
                print("🔄 Reloading employees...")
                self._load_employees()
                print(f"✅ Loaded {len(self.face_service.known_employees)} employee(s)")

            elif key == ord('n') and source == "Live":
                cv2.destroyWindow(WINDOW_TITLE)
                self.register_employee(cap)
                cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

            elif key == ord('s'):
                snap = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(snap, frame)
                print(f"📸 Screenshot saved: {snap}")

            elif key == ord('e'):
                records = self.attendance_service.get_all_today()
                if records:
                    path = self.report_generator.generate_daily_csv(records)
                    print(f"📄 Report exported: {path}")
                else:
                    print("📭 No attendance records today.")

    # ──────────────────────────── Registration ────────────────────────────────

    def register_employee(self, cap: cv2.VideoCapture) -> bool:
        """
        Live employee registration from webcam.
        Captures multiple face samples, extracts encodings, saves to MongoDB.
        """
        print("\n" + "="*60)
        print("📝 LIVE EMPLOYEE REGISTRATION")
        print("="*60)

        employee_id = input("Enter Employee ID (e.g. EMP-001): ").strip()
        if not employee_id:
            print("❌ Registration cancelled — no ID.")
            return False

        if self.db.employees.find_one({'employee_id': employee_id}):
            print(f"❌ Employee {employee_id} already exists.")
            return False

        name = input("Enter Full Name: ").strip()
        if not name:
            print("❌ Registration cancelled — no name.")
            return False

        department = input("Enter Department: ").strip() or "General"

        print(f"\n👤 {employee_id} | {name} | {department}")
        print(f"📸 Position face in camera and press SPACE to capture {NUM_REGISTRATION_SAMPLES} samples.")
        print("   Move your head slightly between captures for better recognition.")
        print("   ESC to cancel.")

        collected_rois: List[np.ndarray] = []
        collected_encodings: List[list] = []
        first_frame_saved: Optional[np.ndarray] = None

        while len(collected_rois) < NUM_REGISTRATION_SAMPLES:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            faces = self.face_service.detect_faces(frame)

            remaining = NUM_REGISTRATION_SAMPLES - len(collected_rois)
            if len(faces) == 1:
                x, y, w, h = faces[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), COLOR_GREEN, 3)
                cv2.putText(
                    frame,
                    f"Face detected — Press SPACE ({len(collected_rois)}/{NUM_REGISTRATION_SAMPLES} captured)",
                    (30, 50), FONT, 0.7, COLOR_GREEN, 2, cv2.LINE_AA,
                )
            elif len(faces) == 0:
                cv2.putText(frame, "No face detected",
                            (30, 50), FONT, 1.0, COLOR_RED, 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, f"{len(faces)} faces — show only ONE",
                            (30, 50), FONT, 0.9, COLOR_RED, 2, cv2.LINE_AA)
                for fx, fy, fw, fh in faces:
                    cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), COLOR_RED, 2)

            # Progress bar
            bar_w = int(300 * len(collected_rois) / NUM_REGISTRATION_SAMPLES)
            cv2.rectangle(frame, (30, 70), (330, 90), (50, 50, 50), -1)
            cv2.rectangle(frame, (30, 70), (30 + bar_w, 90), COLOR_GREEN, -1)
            cv2.putText(frame, f"{len(collected_rois)}/{NUM_REGISTRATION_SAMPLES}",
                        (340, 87), FONT, 0.6, COLOR_WHITE, 1, cv2.LINE_AA)

            cv2.imshow("Employee Registration", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 32 and len(faces) == 1:  # SPACE
                roi = self.face_service.extract_face_roi(frame, faces[0])
                enc = self.face_service.extract_face_encoding(frame, faces[0])
                if roi is not None and enc is not None:
                    collected_rois.append(roi)
                    collected_encodings.append(enc.tolist())
                    if first_frame_saved is None:
                        first_frame_saved = frame.copy()
                    print(f"✅ Sample {len(collected_rois)}/{NUM_REGISTRATION_SAMPLES} captured!")
                    # Brief visual flash
                    flash = frame.copy()
                    cv2.rectangle(flash, (0, 0), (frame.shape[1], frame.shape[0]), COLOR_GREEN, 10)
                    cv2.imshow("Employee Registration", flash)
                    cv2.waitKey(300)
                else:
                    print("❌ Failed to extract encoding. Try again.")

            elif key == 27:  # ESC
                print("❌ Registration cancelled.")
                cv2.destroyWindow("Employee Registration")
                return False

        cv2.destroyWindow("Employee Registration")

        if len(collected_rois) < NUM_REGISTRATION_SAMPLES:
            print(f"❌ Only {len(collected_rois)} samples captured, registration aborted.")
            return False

        print(f"✅ All {NUM_REGISTRATION_SAMPLES} samples captured!")

        # Save best face image (first capture)
        face_filename = f"{employee_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        face_path = os.path.join(self.cfg.FACES_FOLDER, face_filename)
        if first_frame_saved is not None:
            cv2.imwrite(face_path, first_frame_saved)

        # Insert to MongoDB — store ALL encodings
        try:
            doc = {
                'employee_id': employee_id,
                'name': name,
                'department': department,
                'face_image_path': face_path,
                'face_encoding': collected_encodings[0],          # Primary encoding (backward compat)
                'face_encodings_all': collected_encodings,         # All samples for LBPH training
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow(),
                'is_active': True,
            }
            result = self.db.employees.insert_one(doc)
            print(f"✅ Employee saved! MongoDB ID: {result.inserted_id}")
            print("🔄 Reloading face encodings...")
            self._load_employees()
            print(f"✅ Recognizing {len(self.face_service.known_employees)} employee(s)")
            return True
        except Exception as e:
            print(f"❌ Database error: {e}")
            return False
