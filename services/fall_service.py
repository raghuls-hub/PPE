"""
Fall Detection Service using YOLO (Ultralytics).
Detects human falls using a consecutive-frame threshold to reduce false positives.
"""

import cv2
import numpy as np
from typing import List, Tuple
from ultralytics import YOLO

from config import FALL_CONFIDENCE_THRESHOLD, FALL_IOU_THRESHOLD, FALL_FRAME_THRESHOLD, FALL_ALERT_COOLDOWN

FALL_CLASSES   = {"fall", "Fall", "FALL", "fallen", "Fallen"}
NORMAL_CLASSES = {"person", "Person", "stand", "Stand", "walk", "Walk", "standing", "sitting"}


class FallDetection:
    """Represents a single fall detection result."""
    def __init__(self, class_name: str, confidence: float, bbox: Tuple[int, int, int, int]):
        self.class_name = class_name
        self.confidence = confidence
        self.bbox       = bbox
        self.is_fall    = class_name.lower() in {c.lower() for c in FALL_CLASSES}

    def __repr__(self):
        return f"FallDetection({self.class_name}, conf={self.confidence:.2f})"


class FallService:
    """YOLO-based fall detection with consecutive-frame alerting."""

    COLOR_FALL   = (0,   0,   255)
    COLOR_NORMAL = (0,   220,  0)
    COLOR_ALERT  = (0,   0,   180)
    FONT         = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = FALL_CONFIDENCE_THRESHOLD,
        iou_threshold: float        = FALL_IOU_THRESHOLD,
        fall_frame_threshold: int   = FALL_FRAME_THRESHOLD,
        alert_cooldown_frames: int  = FALL_ALERT_COOLDOWN,
    ):
        self.confidence_threshold  = confidence_threshold
        self.iou_threshold         = iou_threshold
        self.fall_frame_threshold  = fall_frame_threshold
        self.alert_cooldown_frames = alert_cooldown_frames

        self._consecutive_fall_frames = 0
        self._alert_active_frames     = 0
        self._alert_blink_counter     = 0
        self._total_falls_logged      = 0

        print(f"🚨 Loading Fall model: {model_path}")
        self.model = YOLO(model_path)
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, verbose=False)
        print("✅ Fall model loaded")

    # ── Detection ──────────────────────────────────────────────────────────────

    def detect_fall(self, frame: np.ndarray) -> List[FallDetection]:
        try:
            results = self.model.predict(
                frame, conf=self.confidence_threshold,
                iou=self.iou_threshold, verbose=False,
            )
            detections: List[FallDetection] = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls  = self.model.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    detections.append(FallDetection(cls, conf, (x1, y1, x2, y2)))
            detections.sort(key=lambda d: d.confidence, reverse=True)
            return detections
        except Exception as e:
            print(f"⚠️  Fall detection error: {e}")
            return []

    def update_fall_state(self, detections: List[FallDetection]) -> bool:
        """Returns True when a sustained fall alert should be shown."""
        has_fall = any(d.is_fall for d in detections)
        if has_fall:
            self._consecutive_fall_frames += 1
        else:
            self._consecutive_fall_frames = 0

        if self._consecutive_fall_frames >= self.fall_frame_threshold:
            if self._alert_active_frames == 0:
                self._total_falls_logged += 1
                print(f"🚨 FALL DETECTED! event #{self._total_falls_logged}")
            self._alert_active_frames = self.alert_cooldown_frames

        if self._alert_active_frames > 0:
            self._alert_active_frames -= 1
            self._alert_blink_counter += 1
            return True

        self._alert_blink_counter = 0
        return False

    @property
    def is_alert_active(self) -> bool:
        return self._alert_active_frames > 0

    @property
    def consecutive_fall_frames(self) -> int:
        return self._consecutive_fall_frames

    # ── Drawing ────────────────────────────────────────────────────────────────

    def draw_fall_boxes(self, frame: np.ndarray, detections: List[FallDetection]) -> None:
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color    = self.COLOR_FALL if det.is_fall else self.COLOR_NORMAL
            thick    = 3 if det.is_fall else 2
            label    = f"{det.class_name} {det.confidence:.0%}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            corner   = min(30, (x2 - x1) // 4, (y2 - y1) // 4)
            for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
                cv2.line(frame, (cx, cy), (cx + dx * corner, cy), color, thick)
                cv2.line(frame, (cx, cy), (cx, cy + dy * corner), color, thick)
            (tw, th), _ = cv2.getTextSize(label, self.FONT, 0.6, 1)
            lx, ly = x1, max(th + 12, y1)
            cv2.rectangle(frame, (lx, ly - th - 10), (lx + tw + 6, ly), color, -1)
            cv2.putText(frame, label, (lx + 3, ly - 3), self.FONT, 0.6, (255, 255, 255), 1)

    def draw_fall_alert(self, frame: np.ndarray, alert_active: bool) -> None:
        if not alert_active:
            return
        h, w = frame.shape[:2]
        bar_h = 52
        if (self._alert_blink_counter // 15) % 2 == 0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - bar_h), (w, h), self.COLOR_ALERT, -1)
            cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
            alert_text = "FALL DETECTED — ALERT SECURITY / MEDICAL"
            (tw, th), _ = cv2.getTextSize(alert_text, self.FONT, 0.8, 2)
            cv2.putText(frame, alert_text, (max(10, (w - tw) // 2), h - bar_h + th + 10),
                        self.FONT, 0.8, (255, 255, 255), 2)
        border = frame.copy()
        cv2.rectangle(border, (0, 0), (w, h), (0, 0, 200), 8)
        cv2.addWeighted(border, 0.6, frame, 0.4, 0, frame)

    def annotate_frame(self, frame: np.ndarray, detections: List[FallDetection], alert_active: bool) -> None:
        self.draw_fall_boxes(frame, detections)
        self.draw_fall_alert(frame, alert_active)
