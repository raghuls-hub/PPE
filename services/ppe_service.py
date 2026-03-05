"""
PPE Detection Service using YOLO (Ultralytics).
Detects PPE items from camera frames and validates compliance.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from ultralytics import YOLO

from config import (
    PPE_CONFIDENCE_THRESHOLD, PPE_IOU_THRESHOLD,
    AVAILABLE_PPE_OPTIONS, PPE_NEGATIVE_MAP,
)


class PPEDetection:
    """Represents a single PPE detection result."""
    def __init__(self, class_name: str, confidence: float, bbox: Tuple[int, int, int, int]):
        self.class_name = class_name
        self.confidence = confidence
        self.bbox       = bbox          # (x1, y1, x2, y2)

    def __repr__(self):
        return f"PPEDetection({self.class_name}, conf={self.confidence:.2f})"


class PPEService:
    """YOLO-based PPE detection service — model loaded once at startup."""

    POSITIVE_COLOR = (0, 200, 0)     # Green — PPE present
    NEGATIVE_COLOR = (0, 80, 255)    # Orange-red — PPE missing
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = PPE_CONFIDENCE_THRESHOLD,
        iou_threshold: float        = PPE_IOU_THRESHOLD,
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold        = iou_threshold

        print(f"🔧 Loading PPE model: {model_path}")
        self.model = YOLO(model_path)
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, verbose=False)
        print("✅ PPE model loaded and warmed up")

    # ── Detection ──────────────────────────────────────────────────────────────

    def detect_ppe(self, frame: np.ndarray) -> List[PPEDetection]:
        try:
            results = self.model.predict(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )
            detections: List[PPEDetection] = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls  = self.model.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    detections.append(PPEDetection(cls, conf, (x1, y1, x2, y2)))
            detections.sort(key=lambda d: d.confidence, reverse=True)
            return detections
        except Exception as e:
            print(f"⚠️  PPE detection error: {e}")
            return []

    # ── Validation ─────────────────────────────────────────────────────────────

    def verify_ppe(
        self,
        detected_classes: List[str],
        required_ppe: List[str],
    ) -> Tuple[bool, List[str], List[str]]:
        """Returns (is_compliant, missing_list, present_list)."""
        if not required_ppe:
            return True, [], list(detected_classes)

        missing: List[str] = []
        present: List[str] = []

        for item in required_ppe:
            pos = item in detected_classes
            neg = PPE_NEGATIVE_MAP.get(item, "")
            neg_detected = neg in detected_classes if neg else False
            if pos and not neg_detected:
                present.append(item)
            else:
                missing.append(item)

        return len(missing) == 0, missing, present

    def get_detected_class_names(self, detections: List[PPEDetection]) -> List[str]:
        return list({d.class_name for d in detections})

    # ── Drawing ────────────────────────────────────────────────────────────────

    def draw_ppe_boxes(
        self,
        frame: np.ndarray,
        detections: List[PPEDetection],
        required_ppe: Optional[List[str]] = None,
    ) -> None:
        """Draw bounding boxes for required PPE + their negative counterparts."""
        if required_ppe:
            allowed: set = set()
            for item in required_ppe:
                allowed.add(item)
                neg = PPE_NEGATIVE_MAP.get(item)
                if neg:
                    allowed.add(neg)
        else:
            allowed = set(AVAILABLE_PPE_OPTIONS) | set(PPE_NEGATIVE_MAP.values())

        for det in detections:
            if det.class_name not in allowed:
                continue
            x1, y1, x2, y2 = det.bbox
            label = f"{det.class_name} {det.confidence:.0%}"
            color = self.NEGATIVE_COLOR if det.class_name.startswith("NO-") else self.POSITIVE_COLOR
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, self.FONT, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4), self.FONT, 0.5, (255, 255, 255), 1)
