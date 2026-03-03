"""
PPE Detection Service using YOLO (Ultralytics).
Detects PPE items from camera frames and validates compliance.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from ultralytics import YOLO

# Maps each required PPE to its "negative" counterpart that must NOT be present
PPE_NEGATIVE_MAP: Dict[str, str] = {
    "Hardhat": "NO-Hardhat",
    "Mask": "NO-Mask",
    "Safety Vest": "NO-Safety Vest",
}

# Displayable PPE options (positive classes only)
AVAILABLE_PPE_OPTIONS: List[str] = ["Hardhat", "Mask", "Safety Vest"]


class PPEDetection:
    """Represents a single PPE detection result."""
    def __init__(self, class_name: str, confidence: float, bbox: Tuple[int, int, int, int]):
        self.class_name = class_name      # e.g. "Hardhat", "NO-Hardhat"
        self.confidence = confidence       # 0.0–1.0
        self.bbox = bbox                   # (x1, y1, x2, y2)

    def __repr__(self):
        return f"PPEDetection({self.class_name}, conf={self.confidence:.2f})"


class PPEService:
    """
    YOLO-based PPE detection service.
    The model (best.pt) is loaded once at startup to maximize FPS.
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.45,
                 iou_threshold: float = 0.45):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        print(f"🔧 Loading YOLO model from: {model_path}")
        try:
            self.model = YOLO(model_path)
            # Warm up the model with a dummy frame
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, verbose=False)
            print("✅ YOLO PPE model loaded and warmed up")
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            raise

    # ─────────────────────────────── Detection ────────────────────────────────

    def detect_ppe(self, frame: np.ndarray) -> List[PPEDetection]:
        """
        Run YOLO inference on a frame and return all PPE detections.

        Args:
            frame: BGR OpenCV frame.

        Returns:
            List of PPEDetection objects sorted by confidence (descending).
        """
        try:
            results = self.model.predict(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )

            detections: List[PPEDetection] = []
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    detections.append(PPEDetection(class_name, confidence, (x1, y1, x2, y2)))

            detections.sort(key=lambda d: d.confidence, reverse=True)
            return detections

        except Exception as e:
            print(f"⚠️  PPE detection error: {e}")
            return []

    # ─────────────────────────────── Validation ───────────────────────────────

    def verify_ppe(
        self,
        detected_classes: List[str],
        required_ppe: List[str],
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Validate that all required PPE items are present and their negative
        counterparts are absent.

        Logic:
          For each required item (e.g. "Hardhat"):
            ✅ "Hardhat" must be in detected_classes
            ❌ "NO-Hardhat" must NOT be in detected_classes

        Args:
            detected_classes: List of class names detected by YOLO.
            required_ppe:     Admin-configured required PPE list.

        Returns:
            (is_compliant, missing_ppe_list, present_ppe_list)
        """
        if not required_ppe:
            # No PPE required → always compliant
            return True, [], list(detected_classes)

        missing: List[str] = []
        present: List[str] = []

        for ppe_item in required_ppe:
            positive_detected = ppe_item in detected_classes
            negative_class = PPE_NEGATIVE_MAP.get(ppe_item)
            negative_detected = (negative_class is not None and
                                 negative_class in detected_classes)

            if positive_detected and not negative_detected:
                present.append(ppe_item)
            else:
                missing.append(ppe_item)

        is_compliant = len(missing) == 0
        return is_compliant, missing, present

    def get_detected_class_names(self, detections: List[PPEDetection]) -> List[str]:
        """Extract unique class names from a list of PPEDetection objects."""
        return list({d.class_name for d in detections})

    # ───────────────────────────── Overlay Drawing ───────────────────────────

    def draw_ppe_boxes(
        self,
        frame: np.ndarray,
        detections: List[PPEDetection],
        required_ppe: Optional[List[str]] = None,
    ) -> None:
        """
        Draw bounding boxes only for admin-selected PPE classes and their
        negative counterparts. Unrelated YOLO classes (Person, machinery,
        Safety Cone, vehicle, etc.) are suppressed.

        Args:
            frame:        Frame to draw on (modified in-place).
            detections:   List of PPEDetection objects.
            required_ppe: Admin-configured required items. If None or empty,
                          falls back to drawing all known PPE classes.
        """
        POSITIVE_COLOR = (0, 200, 0)    # Green — PPE present
        NEGATIVE_COLOR = (0, 80, 255)   # Orange-red — PPE missing/violated
        FONT = cv2.FONT_HERSHEY_SIMPLEX

        # Build the set of class names we want to show:
        # selected positive classes + their negative counterparts
        if required_ppe:
            allowed_classes: set = set()
            for item in required_ppe:
                allowed_classes.add(item)                          # e.g. "Hardhat"
                neg = PPE_NEGATIVE_MAP.get(item)
                if neg:
                    allowed_classes.add(neg)                       # e.g. "NO-Hardhat"
        else:
            # No admin selection — show all known PPE and NO-PPE classes
            allowed_classes = set(AVAILABLE_PPE_OPTIONS) | set(PPE_NEGATIVE_MAP.values())

        for det in detections:
            # Skip classes not relevant to admin's selection
            if det.class_name not in allowed_classes:
                continue

            x1, y1, x2, y2 = det.bbox
            label = f"{det.class_name} {det.confidence:.0%}"
            color = NEGATIVE_COLOR if det.class_name.startswith("NO-") else POSITIVE_COLOR

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4), FONT, 0.5, (255, 255, 255), 1)
