"""
Face Recognition Service using OpenCV LBPH.
Handles face detection, encoding extraction, and employee identification.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict, Any

from config import FACE_SCALE_FACTOR, FACE_MIN_NEIGHBORS, FACE_MIN_SIZE


class FaceService:
    """LBPH-based face recognition — detects and identifies employees."""

    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar Cascade.")

        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=2, neighbors=8, grid_x=8, grid_y=8, threshold=100.0
        )
        self.recognizer_trained = False
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_labels:    List[int]        = []
        self.known_employees:      List[Dict]       = []
        print("✅ FaceService initialized (LBPH)")

    # ── Detection ──────────────────────────────────────────────────────────────

    def detect_faces(
        self,
        frame: np.ndarray,
        scale_factor: float = FACE_SCALE_FACTOR,
        min_neighbors: int  = FACE_MIN_NEIGHBORS,
        min_size: tuple     = FACE_MIN_SIZE,
    ) -> List[Tuple[int, int, int, int]]:
        h_orig, w_orig = frame.shape[:2]
        MAX_W = 640
        if w_orig > MAX_W:
            scale = MAX_W / w_orig
            small = cv2.resize(frame, (int(w_orig * scale), int(h_orig * scale)))
        else:
            scale = 1.0
            small = frame

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 5, 50, 50)
        gray = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=scale_factor,
            minNeighbors=min_neighbors, minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(faces) == 0:
            return []
        if scale != 1.0:
            inv = 1.0 / scale
            return [(int(x*inv), int(y*inv), int(w*inv), int(h*inv)) for x, y, w, h in faces]
        return [tuple(f) for f in faces]  # type: ignore

    # ── Encoding ───────────────────────────────────────────────────────────────

    def extract_face_encoding(
        self,
        frame: np.ndarray,
        face_location: Tuple[int, int, int, int],
        target_size: Tuple[int, int] = (100, 100),
    ) -> Optional[np.ndarray]:
        try:
            x, y, w, h = face_location
            pad = int(min(w, h) * 0.15)
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
            gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi      = gray[y1:y2, x1:x2]
            if roi.size == 0:
                return None
            resized  = cv2.resize(roi, target_size, interpolation=cv2.INTER_CUBIC)
            filtered = cv2.bilateralFilter(resized, 5, 50, 50)
            normed   = cv2.equalizeHist(filtered)
            return normed.flatten().astype(np.float32)
        except Exception as e:
            print(f"⚠️  Face encoding error: {e}")
            return None

    def extract_face_roi(
        self,
        frame: np.ndarray,
        face_location: Tuple[int, int, int, int],
        target_size: Tuple[int, int] = (100, 100),
    ) -> Optional[np.ndarray]:
        try:
            x, y, w, h = face_location
            pad = int(min(w, h) * 0.15)
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
            gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi     = gray[y1:y2, x1:x2]
            if roi.size == 0:
                return None
            resized  = cv2.resize(roi, target_size, interpolation=cv2.INTER_CUBIC)
            filtered = cv2.bilateralFilter(resized, 5, 50, 50)
            return cv2.equalizeHist(filtered)
        except Exception as e:
            print(f"⚠️  ROI extraction error: {e}")
            return None

    # ── Training ───────────────────────────────────────────────────────────────

    def load_known_faces(self, employee_data: List[Dict[str, Any]]) -> None:
        """Load and train LBPH recognizer from employee face encodings."""
        self.known_face_encodings = []
        self.known_face_labels    = []
        self.known_employees      = []
        self.recognizer_trained   = False

        valid = [e for e in employee_data if e.get("face_encoding")]
        if not valid:
            print("⚠️  No employee face encodings found.")
            return

        faces_for_train: List[np.ndarray] = []
        labels_for_train: List[int]       = []
        total_samples = 0

        for idx, emp in enumerate(valid):
            all_encs = emp.get("face_encodings_all") or [emp["face_encoding"]]
            added = 0
            for raw in all_encs:
                try:
                    enc = np.array(raw, dtype=np.float32)
                    if enc.ndim == 1 and enc.shape[0] == 10000:
                        enc_2d = enc.reshape(100, 100).astype(np.uint8)
                    elif enc.ndim == 2:
                        enc_2d = enc.astype(np.uint8)
                    else:
                        side = int(np.sqrt(enc.shape[0]))
                        enc_2d = enc[:side*side].reshape(side, side).astype(np.uint8)
                    faces_for_train.append(enc_2d)
                    labels_for_train.append(idx)
                    added += 1
                except Exception as e:
                    print(f"⚠️  Encoding error for {emp.get('name','?')}: {e}")
            if added > 0:
                self.known_employees.append(emp)
                total_samples += added
                print(f"   ✓ {emp.get('name','?'):20s} ({emp.get('emp_id','?')}) — {added} sample(s)")

        if not faces_for_train:
            print("❌ No valid encodings to train on.")
            return

        try:
            self.recognizer.train(faces_for_train, np.array(labels_for_train, dtype=np.int32))
            self.recognizer_trained = True
            print(f"✅ LBPH trained: {len(valid)} employee(s), {total_samples} samples")
        except Exception as e:
            print(f"❌ Training failed: {e}")

    # ── Recognition ────────────────────────────────────────────────────────────

    def identify_employee(
        self,
        face_roi_2d: np.ndarray,
        confidence_threshold: float = 120.0,
    ) -> Optional[Dict[str, Any]]:
        if not self.recognizer_trained or not self.known_employees:
            return None
        try:
            label, confidence = self.recognizer.predict(face_roi_2d)
            if confidence <= confidence_threshold:
                emp = dict(self.known_employees[label])
                emp["raw_confidence"]    = float(confidence)
                emp["confidence_score"]  = max(0.0, 1.0 - confidence / 100.0)
                emp["employee_id"]       = emp.get("emp_id", emp.get("employee_id", ""))
                return emp
            return None
        except Exception as e:
            print(f"⚠️  Recognition error: {e}")
            return None
