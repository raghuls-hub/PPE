"""
Fall Detection Service using YOLO (Ultralytics).
Detects human falls in video frames using a consecutive-frame threshold
to reduce false positives, and renders visual alerts on the annotated frame.
Model: fall_detection.pt
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from ultralytics import YOLO


# Class names the fall model is expected to output
FALL_CLASSES   = {"fall", "Fall", "FALL", "fallen", "Fallen", "FALLEN"}
NORMAL_CLASSES = {"person", "Person", "stand", "Stand", "walk", "Walk",
                  "standing", "Standing", "sitting", "Sitting"}


class FallDetection:
    """Represents a single fall/normal detection result."""

    def __init__(self, class_name: str, confidence: float, bbox: Tuple[int, int, int, int]):
        self.class_name = class_name
        self.confidence = confidence
        self.bbox       = bbox          # (x1, y1, x2, y2)
        self.is_fall    = class_name.lower() in {c.lower() for c in FALL_CLASSES}

    def __repr__(self):
        return f"FallDetection({self.class_name}, conf={self.confidence:.2f})"


class FallService:
    """
    YOLO-based fall detection service.

    Key behaviour:
      - A fall alert is only raised after FALL_FRAME_THRESHOLD *consecutive*
        frames contain a fall detection, preventing single-frame false positives.
      - The consecutive counter resets as soon as a frame has NO fall detections.
      - All detected persons (fall + normal) are annotated with bounding boxes.
      - A full-width alert banner + red border are drawn when the threshold is met.
    """

    # Visual constants (BGR)
    COLOR_FALL   = (0,   0,   255)   # Red  — fall
    COLOR_NORMAL = (0,   220,  0)    # Green — standing / walking
    COLOR_ALERT  = (0,   0,   180)   # Dark red alert bar
    FONT         = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.50,
        iou_threshold: float        = 0.45,
        fall_frame_threshold: int   = 10,
        alert_cooldown_frames: int  = 30,
    ):
        """
        Args:
            model_path:            Path to fall_detection.pt.
            confidence_threshold:  Min YOLO confidence to consider a detection.
            iou_threshold:         NMS IoU threshold.
            fall_frame_threshold:  Consecutive frames required before alerting
                                   (matches sample code's fall_threshold = 10).
            alert_cooldown_frames: Minimum frames the alert banner stays visible
                                   after the consecutive threshold is met.
        """
        self.confidence_threshold  = confidence_threshold
        self.iou_threshold         = iou_threshold
        self.fall_frame_threshold  = fall_frame_threshold
        self.alert_cooldown_frames = alert_cooldown_frames

        # Internal state
        self._consecutive_fall_frames: int = 0   # running count of consecutive fall frames
        self._alert_active_frames: int     = 0   # frames remaining in the alert cooldown
        self._alert_blink_counter: int     = 0   # drives the pulsing animation
        self._total_falls_logged: int      = 0   # cumulative fall events (for console)

        print(f"🚨 Loading Fall Detection model from: {model_path}")
        try:
            self.model = YOLO(model_path)
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, verbose=False)
            print("✅ Fall detection model loaded and warmed up")
        except Exception as e:
            print(f"❌ Failed to load fall detection model: {e}")
            raise

    # ─────────────────────────────── Detection ────────────────────────────────

    def detect_fall(self, frame: np.ndarray) -> List[FallDetection]:
        """
        Run YOLO inference on a single frame.

        Args:
            frame: BGR OpenCV frame.

        Returns:
            List of FallDetection objects sorted by confidence (descending).
        """
        try:
            results = self.model.predict(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )

            detections: List[FallDetection] = []
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    class_id   = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    detections.append(FallDetection(class_name, confidence, (x1, y1, x2, y2)))

            detections.sort(key=lambda d: d.confidence, reverse=True)
            return detections

        except Exception as e:
            print(f"⚠️  Fall detection error: {e}")
            return []

    def update_fall_state(self, detections: List[FallDetection]) -> bool:
        """
        Update the consecutive-frame fall counter and cooldown.

        Returns:
            True if a sustained fall alert should currently be shown.
        """
        has_fall = any(d.is_fall for d in detections)

        if has_fall:
            self._consecutive_fall_frames += 1
        else:
            self._consecutive_fall_frames = 0

        # Trigger alert when threshold is reached
        if self._consecutive_fall_frames >= self.fall_frame_threshold:
            if self._alert_active_frames == 0:
                # New fall event
                self._total_falls_logged += 1
                print(
                    f"🚨 FALL DETECTED! (event #{self._total_falls_logged}, "
                    f"{self._consecutive_fall_frames} consecutive frames)"
                )
            self._alert_active_frames = self.alert_cooldown_frames

        # Count down the cooldown
        if self._alert_active_frames > 0:
            self._alert_active_frames -= 1
            self._alert_blink_counter += 1
            return True

        self._alert_blink_counter = 0
        return False

    @property
    def is_alert_active(self) -> bool:
        """True when the fall alert is currently in its cooldown period."""
        return self._alert_active_frames > 0

    @property
    def consecutive_fall_frames(self) -> int:
        return self._consecutive_fall_frames

    @property
    def total_falls_logged(self) -> int:
        """Cumulative number of distinct fall events that crossed the threshold."""
        return self._total_falls_logged

    # ─────────────────────────── Overlay Drawing ──────────────────────────────

    def draw_fall_boxes(
        self,
        frame: np.ndarray,
        detections: List[FallDetection],
    ) -> None:
        """
        Draw bounding boxes for all detections.
          - Fall   → red box (higher confidence → thicker border)
          - Normal → green box

        Args:
            frame:      Frame to annotate (in-place).
            detections: List of FallDetection objects.
        """
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color     = self.COLOR_FALL if det.is_fall else self.COLOR_NORMAL
            thickness = 3 if det.is_fall else 2
            label     = f"{det.class_name} {det.confidence:.0%}"

            # Draw corner-style box (matching sample code's cvzone.cornerRect style)
            # Full rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            # Corner accents
            corner_len = min(30, (x2 - x1) // 4, (y2 - y1) // 4)
            for cx, cy, dx, dy in [
                (x1, y1, 1, 1), (x2, y1, -1, 1),
                (x1, y2, 1, -1), (x2, y2, -1, -1),
            ]:
                cv2.line(frame, (cx, cy), (cx + dx * corner_len, cy), color, thickness)
                cv2.line(frame, (cx, cy), (cx, cy + dy * corner_len), color, thickness)

            # Label background + text
            (tw, th), _ = cv2.getTextSize(label, self.FONT, 0.6, 1)
            lx, ly = x1, max(th + 12, y1)
            cv2.rectangle(frame, (lx, ly - th - 10), (lx + tw + 6, ly), color, -1)
            cv2.putText(frame, label, (lx + 3, ly - 3),
                        self.FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_fall_alert(
        self,
        frame: np.ndarray,
        alert_active: bool,
    ) -> None:
        """
        Draw the full-width FALL DETECTED alert banner and red border.

        Args:
            frame:        Frame to annotate (in-place).
            alert_active: Whether the alert should be shown (from update_fall_state).
        """
        if not alert_active:
            return

        h, w = frame.shape[:2]
        bar_h = 52

        # Pulsing: alternate every ~15 frames
        if (self._alert_blink_counter // 15) % 2 == 0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - bar_h), (w, h), self.COLOR_ALERT, -1)
            cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

            alert_text = "FALL DETECTED — ALERT SECURITY / MEDICAL"
            (tw, th), _ = cv2.getTextSize(alert_text, self.FONT, 0.8, 2)
            tx = max(10, (w - tw) // 2)
            ty = h - bar_h + th + 10
            cv2.putText(frame, alert_text, (tx, ty),
                        self.FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Red border around the whole frame
        border = frame.copy()
        cv2.rectangle(border, (0, 0), (w, h), (0, 0, 200), 8)
        cv2.addWeighted(border, 0.6, frame, 0.4, 0, frame)

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[FallDetection],
        alert_active: bool,
    ) -> None:
        """
        Convenience: draw boxes + alert banner in one call.

        Args:
            frame:        Frame to annotate (in-place).
            detections:   Current FallDetection list.
            alert_active: Return value of update_fall_state().
        """
        self.draw_fall_boxes(frame, detections)
        self.draw_fall_alert(frame, alert_active)
