"""
Fire Detection Service using YOLO (Ultralytics).
Detects fire and smoke in video frames and triggers visual alerts.
Model: hf_firedetection.pt
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Optional
from ultralytics import YOLO


# Detection classes the fire model outputs (fire and smoke variants)
FIRE_CLASSES  = {"fire", "Fire", "FIRE"}
SMOKE_CLASSES = {"smoke", "Smoke", "SMOKE"}


class FireDetection:
    """Represents a single fire/smoke detection result."""

    def __init__(self, class_name: str, confidence: float, bbox: Tuple[int, int, int, int]):
        self.class_name  = class_name
        self.confidence  = confidence
        self.bbox        = bbox          # (x1, y1, x2, y2)
        self.is_fire     = class_name in FIRE_CLASSES
        self.is_smoke    = class_name in SMOKE_CLASSES

    def __repr__(self):
        return f"FireDetection({self.class_name}, conf={self.confidence:.2f})"


class FireService:
    """
    YOLO-based fire & smoke detection service.
    The model is loaded once at startup; detection runs every N frames
    to keep CPU/GPU usage manageable alongside the PPE & face pipelines.
    """

    # Visual constants
    COLOR_FIRE  = (0,   60, 255)   # BGR — vivid red-orange
    COLOR_SMOKE = (140, 140, 140)  # BGR — gray
    ALERT_BG    = (0,   0,  180)   # BGR — dark red alert bar
    FONT        = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.40,
        iou_threshold: float        = 0.45,
        alert_cooldown_sec: float   = 5.0,
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold        = iou_threshold
        self.alert_cooldown_sec   = alert_cooldown_sec

        self._last_alert_time: float = 0.0   # last time an alert was shown
        self._fire_frame_count: int  = 0     # frames with active detection

        print(f"🔥 Loading Fire Detection model from: {model_path}")
        try:
            self.model = YOLO(model_path)
            # Warm up
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, verbose=False)
            print("✅ Fire detection model loaded and warmed up")
        except Exception as e:
            print(f"❌ Failed to load fire detection model: {e}")
            raise

    # ─────────────────────────────── Detection ────────────────────────────────

    def detect_fire(self, frame: np.ndarray) -> List[FireDetection]:
        """
        Run YOLO inference on a frame and return all fire/smoke detections.

        Args:
            frame: BGR OpenCV frame.

        Returns:
            List of FireDetection objects sorted by confidence (descending).
        """
        try:
            results = self.model.predict(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )

            detections: List[FireDetection] = []
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    class_id   = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    detections.append(FireDetection(class_name, confidence, (x1, y1, x2, y2)))

            detections.sort(key=lambda d: d.confidence, reverse=True)
            return detections

        except Exception as e:
            print(f"⚠️  Fire detection error: {e}")
            return []

    def has_fire(self, detections: List[FireDetection]) -> bool:
        """Returns True if any fire (not just smoke) was detected."""
        return any(d.is_fire for d in detections)

    def has_smoke(self, detections: List[FireDetection]) -> bool:
        """Returns True if any smoke was detected."""
        return any(d.is_smoke for d in detections)

    # ─────────────────────────── Overlay Drawing ──────────────────────────────

    def draw_fire_boxes(
        self,
        frame: np.ndarray,
        detections: List[FireDetection],
    ) -> None:
        """
        Draw bounding boxes around detected fire/smoke regions.
        Fire → vivid red-orange outline; Smoke → gray outline.

        Args:
            frame:      Frame to annotate (in-place).
            detections: List of FireDetection objects.
        """
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color  = self.COLOR_FIRE if det.is_fire else self.COLOR_SMOKE
            label  = f"{det.class_name} {det.confidence:.0%}"

            # Thicker box for fire detections
            thickness = 3 if det.is_fire else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Label background
            (tw, th), _ = cv2.getTextSize(label, self.FONT, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                        self.FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_fire_alert(
        self,
        frame: np.ndarray,
        detections: List[FireDetection],
    ) -> None:
        """
        Draw a full-width alert banner at the bottom of the frame when
        fire or smoke is detected.  Pulses (visible every ~0.5 s) for
        better attention during recordings.

        Args:
            frame:      Frame to annotate (in-place).
            detections: List of current FireDetection objects.
        """
        if not detections:
            self._fire_frame_count = 0
            return

        self._fire_frame_count += 1
        fire_present  = self.has_fire(detections)
        smoke_present = self.has_smoke(detections)

        # Decide alert text
        if fire_present and smoke_present:
            alert_text  = "⚠ FIRE & SMOKE DETECTED — EVACUATE IMMEDIATELY"
            alert_color = (0, 30, 220)   # deep red
        elif fire_present:
            alert_text  = "⚠ FIRE DETECTED — ALERT SECURITY"
            alert_color = (0, 60, 255)
        else:
            alert_text  = "⚠ SMOKE DETECTED — CHECK AREA"
            alert_color = (60, 120, 200)

        h, w = frame.shape[:2]
        bar_h = 48

        # Pulsing: show every 2nd second (based on frame count)
        # This creates a visible blinking effect even in recorded video
        if (self._fire_frame_count // 15) % 2 == 0:
            # Draw alert bar
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - bar_h), (w, h), alert_color, -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

            # Alert text
            (tw, th), _ = cv2.getTextSize(alert_text, self.FONT, 0.75, 2)
            tx = max(10, (w - tw) // 2)
            ty = h - bar_h + th + 10
            cv2.putText(frame, alert_text, (tx, ty),
                        self.FONT, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        # Also overlay a red tinted border around the whole frame
        if fire_present:
            border = frame.copy()
            cv2.rectangle(border, (0, 0), (w, h), (0, 0, 200), 8)
            cv2.addWeighted(border, 0.6, frame, 0.4, 0, frame)

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[FireDetection],
    ) -> None:
        """
        Convenience method: draw both bounding boxes and alert banner.

        Args:
            frame:      Frame to annotate (in-place).
            detections: List of FireDetection objects.
        """
        self.draw_fire_boxes(frame, detections)
        self.draw_fire_alert(frame, detections)

    # ────────────────────────── Console Logging ───────────────────────────────

    def log_detections(self, detections: List[FireDetection], source: str = "") -> None:
        """Print a concise console log when fire/smoke is detected."""
        if not detections:
            return

        now = time.time()
        if now - self._last_alert_time < self.alert_cooldown_sec:
            return   # Throttle console spam

        self._last_alert_time = now
        items = ", ".join(f"{d.class_name}({d.confidence:.0%})" for d in detections)
        tag   = f"[{source}] " if source else ""
        print(f"🔥 {tag}FIRE/SMOKE DETECTED: {items}")
