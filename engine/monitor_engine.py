"""
Monitor Engine — per-camera background detection threads for Live Monitoring tab.
Each CameraMonitorThread runs PPE + Fire + Fall detection on one camera stream.
Models are shared singletons across threads to save memory.
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

FALL_CLASSES = {"fall", "fallen"}
FIRE_CLASSES = {"fire"}

# ── Shared Model Singletons (loaded once, shared across all threads) ───────────

_model_lock    = threading.Lock()
_ppe_service   = None
_fire_service  = None
_fall_service  = None
_models_loaded = False


def _load_models():
    global _ppe_service, _fire_service, _fall_service, _models_loaded
    with _model_lock:
        if _models_loaded:
            return
        from services.ppe_service  import PPEService
        from services.fire_service import FireService
        from services.fall_service import FallService

        try:
            _ppe_service = PPEService(cfg.PPE_MODEL_PATH)
            print("✅ Shared PPE model loaded")
        except Exception as e:
            print(f"⚠️  PPE model failed: {e}")

        try:
            _fire_service = FireService(cfg.FIRE_MODEL_PATH)
            print("✅ Shared Fire model loaded")
        except Exception as e:
            print(f"⚠️  Fire model failed: {e}")

        try:
            _fall_service = FallService(cfg.FALL_MODEL_PATH)
            print("✅ Shared Fall model loaded")
        except Exception as e:
            print(f"⚠️  Fall model failed: {e}")

        _models_loaded = True


# ── Per-Camera Monitor Thread ─────────────────────────────────────────────────

class CameraMonitorThread(threading.Thread):
    """Runs PPE + Fire + Fall detection on one camera stream."""

    def __init__(self, camera: Dict):
        super().__init__(daemon=True, name=f"Monitor-{camera.get('name','cam')}")
        self.camera   = camera
        self.cam_id   = camera["_id"]
        self.cam_name = camera.get("name", "Camera")
        self.stream   = camera["stream_url"]

        self._stop_evt = threading.Event()
        self._lock     = threading.Lock()
        self.state: Dict[str, Any] = {
            "frame":     None,
            "ppe_alert": False,
            "fire_alert": False,
            "fall_alert": False,
            "ppe_violation_frames": 0,
            "fire_frames":  0,
            "fall_frames":  0,
            "last_alert_time": None,
            "fps":       0.0,
            "status":    "starting",
            "error":     None,
        }

    def stop(self):
        self._stop_evt.set()

    def get_state(self) -> Dict:
        with self._lock:
            return dict(self.state)

    def _upd(self, **kw):
        with self._lock:
            self.state.update(kw)

    def run(self):
        _load_models()

        src = self.stream
        if isinstance(src, str) and "drive.google.com" in src:
            src = convert_gdrive_url(src)

        try:
            src = int(src)
        except (ValueError, TypeError):
            pass

        print(f"[ENGINE] Thread starting for {self.cam_name} with source: {src}")
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"[ENGINE] Failed to open VideoCapture for {self.stream}")
            self._upd(status="error", error=f"Cannot open: {self.stream}")
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"[ENGINE] VideoCapture opened successfully for {self.cam_name}")

        self._upd(status="running")
        frame_n   = 0
        t_fps     = time.time()
        fps_cnt   = 0
        ppe_viol  = 0
        fire_f    = 0
        fall_f    = 0

        # Per-thread fall state tracker
        from services.fall_service import FallService as _FallCls
        _fall_state_obj = None
        if _fall_service is not None:
            _fall_state_obj = _FallCls.__new__(_FallCls)
            _fall_state_obj._consecutive_fall_frames = 0
            _fall_state_obj._alert_active_frames     = 0
            _fall_state_obj._alert_blink_counter     = 0
            _fall_state_obj._total_falls_logged      = 0
            _fall_state_obj.fall_frame_threshold     = cfg.FALL_FRAME_THRESHOLD
            _fall_state_obj.alert_cooldown_frames    = cfg.FALL_ALERT_COOLDOWN

        while not self._stop_evt.is_set():
            ret, frame = cap.read()
            if not ret:
                print(f"[ENGINE] Warning: cap.read() failed for {self.cam_name}. Retrying in 1s...")
                time.sleep(1)
                cap.release()
                cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue

            frame_n += 1
            fps_cnt += 1

            if frame_n % cfg.PROCESS_EVERY_N_FRAMES == 0:
                required_ppe = db.get_required_ppe()
                h, w = frame.shape[:2]

                # ── PPE Detection ────────────────────────────────────────────
                ppe_ok = True
                if _ppe_service:
                    with _model_lock:
                        dets = _ppe_service.detect_ppe(frame)
                    _ppe_service.draw_ppe_boxes(frame, dets, required_ppe)
                    detected_cls = _ppe_service.get_detected_class_names(dets)
                    ppe_ok, missing, _ = _ppe_service.verify_ppe(detected_cls, required_ppe)

                ppe_viol = 0 if ppe_ok else ppe_viol + 1

                # ── Fire Detection ───────────────────────────────────────────
                fire_det = False
                if _fire_service:
                    with _model_lock:
                        fire_dets = _fire_service.detect_fire(frame)
                    _fire_service.draw_fire_boxes(frame, fire_dets)
                    fire_det = _fire_service.has_fire(fire_dets)
                fire_f = fire_f + 1 if fire_det else 0

                # ── Fall Detection ───────────────────────────────────────────
                fall_alert = False
                if _fall_service and _fall_state_obj:
                    with _model_lock:
                        fall_dets = _fall_service.detect_fall(frame)
                    _fall_service.draw_fall_boxes(frame, fall_dets)
                    fall_alert = _fall_state_obj.update_fall_state(fall_dets)
                    if fall_alert:
                        _fall_service.draw_fall_alert(frame, True)

                # ── Alert states ─────────────────────────────────────────────
                ppe_alert  = ppe_viol >= cfg.PPE_VIOLATION_THRESHOLD
                fire_alert = fire_f   >= cfg.FIRE_FRAME_THRESHOLD

                # ── Overlays ─────────────────────────────────────────────────
                if ppe_alert:
                    cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 220), 8)
                    cv2.putText(frame, "PPE VIOLATION", (10, h - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if fire_alert:
                    ov = frame.copy()
                    cv2.rectangle(ov, (0, h - 50), (w, h), (0, 0, 180), -1)
                    cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)
                    cv2.rectangle(frame, (0, 0), (w, h), (0, 30, 255), 8)
                    cv2.putText(frame, "FIRE DETECTED — EVACUATE", (10, h - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    # DB alert push (rate-limited 30s)
                    last = self.state.get("last_alert_time")
                    if last is None or (datetime.utcnow() - last).seconds > 30:
                        db.push_alert(self.cam_id, "fire", f"Fire on {self.cam_name}")
                        self._upd(last_alert_time=datetime.utcnow())

                if fall_alert:
                    last = self.state.get("last_alert_time")
                    if last is None or (datetime.utcnow() - last).seconds > 30:
                        db.push_alert(self.cam_id, "fall", f"Fall on {self.cam_name}")
                        self._upd(last_alert_time=datetime.utcnow())

                # ── HUD ───────────────────────────────────────────────────────
                elapsed = time.time() - t_fps
                if elapsed >= 2.0:
                    fps = fps_cnt / elapsed
                    t_fps   = time.time()
                    fps_cnt = 0
                else:
                    fps = self.state.get("fps", 0.0)

                cv2.putText(frame, f"{self.cam_name}  FPS:{fps:.1f}", (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                # Lower quality to 40 for speed
                _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
                self._upd(
                    frame=jpg.tobytes(),
                    ppe_alert=ppe_alert,
                    fire_alert=fire_alert,
                    fall_alert=fall_alert,
                    ppe_violation_frames=ppe_viol,
                    fire_frames=fire_f,
                    fall_frames=fall_f,
                    fps=fps,
                    status="running",
                )

        cap.release()
        self._upd(status="stopped")


# ── Engine Manager ─────────────────────────────────────────────────────────────

class MonitorEngine:
    """Pool of CameraMonitorThreads stored in st.session_state."""

    def __init__(self):
        self._threads: Dict[str, CameraMonitorThread] = {}

    def start_camera(self, camera: Dict) -> None:
        cid = camera["_id"]
        if cid in self._threads and self._threads[cid].is_alive():
            return
        t = CameraMonitorThread(camera)
        t.start()
        self._threads[cid] = t

    def stop_camera(self, cam_id: str) -> None:
        t = self._threads.pop(cam_id, None)
        if t:
            t.stop()

    def stop_all(self) -> None:
        for t in list(self._threads.values()):
            t.stop()
        self._threads.clear()

    def get_state(self, cam_id: str) -> Optional[Dict]:
        t = self._threads.get(cam_id)
        return t.get_state() if t else None

    def is_running(self, cam_id: str) -> bool:
        t = self._threads.get(cam_id)
        return t is not None and t.is_alive()

    def running_ids(self) -> List[str]:
        return [cid for cid, t in self._threads.items() if t.is_alive()]
