"""
Attendance Engine — PPE + Face Recognition attendance thread.
Runs on a single designated attendance camera.
All imports are LOCAL to this project.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

import config as cfg
import db
from utils.video_utils import convert_gdrive_url

from services.face_service import FaceService
from services.ppe_service  import PPEService

NUM_SAMPLES = 5

# ── PPE Service singleton ──────────────────────────────────────────────────────
_ppe_lock    = threading.Lock()
_ppe_service: Optional[PPEService] = None


def _get_ppe_service() -> Optional[PPEService]:
    global _ppe_service
    with _ppe_lock:
        if _ppe_service is None:
            try:
                _ppe_service = PPEService(cfg.PPE_MODEL_PATH)
            except Exception as e:
                print(f"⚠️  PPE service failed: {e}")
    return _ppe_service


# ── Attendance Thread ──────────────────────────────────────────────────────────

class AttendanceThread(threading.Thread):
    """Face recognition + PPE-gated attendance on ONE camera."""

    def __init__(self, camera: Dict):
        super().__init__(daemon=True, name=f"Attendance-{camera.get('name','cam')}")
        self.camera   = camera
        self.cam_id   = camera["_id"]
        self.cam_name = camera.get("name", "Attendance Camera")
        self.stream   = camera["stream_url"]

        self._stop_evt = threading.Event()
        self._lock     = threading.Lock()
        self.state: Dict[str, Any] = {
            "frame":       None,
            "last_marked": None,
            "status":      "starting",
            "error":       None,
            "fps":         0.0,
        }
        self._face_service: Optional[FaceService] = None
        self._ppe_service:  Optional[PPEService]  = None

    def stop(self):
        self._stop_evt.set()

    def reload_faces(self):
        if self._face_service:
            self._face_service.load_known_faces(db.get_worker_encodings())

    def get_state(self) -> Dict:
        with self._lock:
            return dict(self.state)

    def _upd(self, **kw):
        with self._lock:
            self.state.update(kw)

    def run(self):
        self._face_service = FaceService()
        self._face_service.load_known_faces(db.get_worker_encodings())
        self._ppe_service = _get_ppe_service()

        src = self.stream
        if isinstance(src, str) and "drive.google.com" in src:
            src = convert_gdrive_url(src)

        try:
            src = int(src)
        except (ValueError, TypeError):
            pass

        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            self._upd(status="error", error=f"Cannot open: {self.stream}")
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._upd(status="running")
        frame_n = 0
        t_fps   = time.time()
        fps_cnt = 0
        _last_ppe_dets = []

        while not self._stop_evt.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(1)
                continue

            frame = cv2.flip(frame, 1)
            frame_n += 1
            fps_cnt  += 1
            h, w = frame.shape[:2]

            # ── PPE detection ─────────────────────────────────────────────────
            if frame_n % cfg.PROCESS_EVERY_N_FRAMES == 0 and self._ppe_service:
                _last_ppe_dets = self._ppe_service.detect_ppe(frame)

            required_ppe     = db.get_required_ppe()
            detected_classes = (self._ppe_service.get_detected_class_names(_last_ppe_dets)
                                if self._ppe_service else [])

            if self._ppe_service:
                self._ppe_service.draw_ppe_boxes(frame, _last_ppe_dets, required_ppe)

            is_compliant, missing_ppe, present_ppe = (
                self._ppe_service.verify_ppe(detected_classes, required_ppe)
                if self._ppe_service else (True, [], [])
            )

            # ── Face detection ────────────────────────────────────────────────
            faces = self._face_service.detect_faces(
                frame,
                scale_factor=cfg.FACE_SCALE_FACTOR,
                min_neighbors=cfg.FACE_MIN_NEIGHBORS,
                min_size=cfg.FACE_MIN_SIZE,
            )

            for face_loc in faces:
                x, y, fw, fh = face_loc
                face_roi = self._face_service.extract_face_roi(frame, face_loc)
                if face_roi is None:
                    cv2.rectangle(frame, (x, y), (x+fw, y+fh), (140, 140, 140), 2)
                    continue

                result = self._face_service.identify_employee(
                    face_roi, confidence_threshold=cfg.FACE_MATCH_THRESHOLD
                )

                if result is None:
                    cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 200, 255), 2)
                    cv2.putText(frame, "UNKNOWN", (x, y - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
                    continue

                emp_id = result.get("emp_id") or result.get("employee_id", "")
                name   = result["name"]
                conf   = 1.0 - result["raw_confidence"] / 100.0
                color  = (0, 220, 0) if is_compliant else (0, 0, 220)

                cv2.rectangle(frame, (x, y), (x+fw, y+fh), color, 2)
                lines = [name, f"PPE: {'OK' if is_compliant else 'FAIL'}"]
                if not is_compliant and missing_ppe:
                    lines.append(f"Missing: {', '.join(missing_ppe)}")
                for i, ln in enumerate(lines):
                    cv2.putText(frame, ln, (x, y - 8 - i * 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

                # ── Mark attendance ───────────────────────────────────────────
                if is_compliant:
                    rec_id = db.mark_attendance(
                        emp_id=emp_id, name=name,
                        ppe_ok=True, detected_ppe=present_ppe,
                        missing_ppe=[], camera_id=self.cam_id, confidence=conf,
                    )
                    if rec_id:
                        self._upd(last_marked={
                            "name": name, "emp_id": emp_id,
                            "time": datetime.utcnow().strftime("%H:%M:%S"),
                        })
                        ov = frame.copy()
                        cv2.rectangle(ov, (0, 0), (w, h), (0, 180, 0), -1)
                        cv2.addWeighted(ov, 0.15, frame, 0.85, 0, frame)
                        cv2.putText(frame, f"ATTENDANCE MARKED — {name}", (10, h - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 80), 2)

            # ── PPE HUD bar ───────────────────────────────────────────────────
            if required_ppe:
                bar_txt = "PPE OK" if is_compliant else f"MISSING: {', '.join(missing_ppe)}"
                bar_col = (0, 220, 0) if is_compliant else (0, 0, 220)
                cv2.rectangle(frame, (0, 0), (w, 30), (20, 20, 20), -1)
                cv2.putText(frame, bar_txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, bar_col, 1)

            # ── FPS ───────────────────────────────────────────────────────────
            elapsed = time.time() - t_fps
            if elapsed >= 2.0:
                fps = fps_cnt / elapsed
                t_fps   = time.time()
                fps_cnt = 0
            else:
                fps = self.state.get("fps", 0.0)

            cv2.putText(frame, f"FPS:{fps:.1f}", (w - 80, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            # Lower quality to 40 for speed
            _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
            self._upd(frame=jpg.tobytes(), fps=fps, status="running")

        cap.release()
        self._upd(status="stopped")


# ── Face Capture Helper ────────────────────────────────────────────────────────

def capture_face_samples(stream_url: str, progress_cb=None) -> Optional[List]:
    """
    Opens camera, captures NUM_SAMPLES face encodings (auto-capture every 0.4s).
    progress_cb(i, total, frame_bytes) after each capture.
    Returns list of flat float32 arrays, or None on failure.
    """
    face_svc = FaceService()

    src = stream_url
    try:
        src = int(src)
    except (ValueError, TypeError):
        pass

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return None

    collected: List[list] = []
    timeout = time.time() + 60

    while len(collected) < NUM_SAMPLES and time.time() < timeout:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        faces = face_svc.detect_faces(frame)

        display = frame.copy()
        h, w    = display.shape[:2]
        cv2.putText(display,
                    f"Align face — {len(collected)}/{NUM_SAMPLES} captured",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
        bar = int(w * len(collected) / NUM_SAMPLES)
        cv2.rectangle(display, (0, h - 10), (bar, h), (0, 200, 0), -1)

        if len(faces) == 1:
            x, y, fw, fh = faces[0]
            cv2.rectangle(display, (x, y), (x+fw, y+fh), (0, 220, 0), 2)
            enc = face_svc.extract_face_encoding(frame, faces[0])
            if enc is not None:
                collected.append(enc.tolist())
                _, jpg = cv2.imencode(".jpg", display)
                if progress_cb:
                    progress_cb(len(collected), NUM_SAMPLES, jpg.tobytes())
                time.sleep(0.4)

    cap.release()
    return collected if len(collected) == NUM_SAMPLES else None
