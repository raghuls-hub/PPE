"""
Face Recognition Service using OpenCV LBPH (Local Binary Pattern Histograms).
Handles face detection, encoding extraction, and employee identification.

IMPROVEMENTS:
- Better face detection parameters
- Consistent encoding normalization
- Stricter confidence thresholds
- Multi-sample training optimization
- Better debugging output
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict, Any


class FaceService:
    """
    LBPH-based face recognition service.
    Detects faces with Haar Cascade and identifies employees.
    """

    def __init__(self):
        # Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            raise RuntimeError("❌ Failed to load Haar Cascade for face detection.")

        # LBPH recognizer with optimized parameters
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=2,        # Increased from default 1 for better feature capture
            neighbors=8,     # Default, good balance
            grid_x=8,        # Default
            grid_y=8,        # Default
            threshold=100.0  # Will be overridden in identify_employee
        )
        self.recognizer_trained = False

        # Known faces (in-memory)
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_labels: List[int] = []
        self.known_employees: List[Dict[str, Any]] = []  # Metadata per label index

        print("✅ FaceService initialized (LBPH with optimized parameters)")

    # ───────────────────────────────── Detection ──────────────────────────────

    def detect_faces(
        self,
        frame: np.ndarray,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,  # INCREASED from 3 for better quality
        min_size: Tuple[int, int] = (60, 60),  # INCREASED from (40,40)
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a frame using Haar Cascade.
        Internally downscales large frames so detection runs fast on HD feeds.

        Args:
            scale_factor: How much to reduce image size at each scale (1.1 = 10% reduction)
            min_neighbors: How many neighbors each rectangle should have (higher = stricter)
            min_size: Minimum face size in pixels

        Returns:
            List of (x, y, w, h) tuples for each detected face (original coords).
        """
        h_orig, w_orig = frame.shape[:2]

        # Downscale for speed — Haar on 1280×720 is ~4× slower than on 640×360
        MAX_DETECT_WIDTH = 640
        if w_orig > MAX_DETECT_WIDTH:
            scale = MAX_DETECT_WIDTH / w_orig
            small = cv2.resize(frame, (int(w_orig * scale), int(h_orig * scale)))
        else:
            scale = 1.0
            small = frame

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter for noise reduction while preserving edges
        gray = cv2.bilateralFilter(gray, 5, 50, 50)
        
        # Histogram equalization for better contrast
        gray = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE  # More accurate but slightly slower
        )

        if len(faces) == 0:
            return []

        # Scale bounding boxes back to original resolution
        if scale != 1.0:
            inv = 1.0 / scale
            return [(int(x * inv), int(y * inv), int(w * inv), int(h * inv))
                    for x, y, w, h in faces]
        return [tuple(f) for f in faces]  # type: ignore

    # ───────────────────────────────── Encoding ───────────────────────────────

    def extract_face_encoding(
        self,
        frame: np.ndarray,
        face_location: Tuple[int, int, int, int],
        target_size: Tuple[int, int] = (100, 100),
    ) -> Optional[np.ndarray]:
        """
        Extract a normalized grayscale face ROI for LBPH matching.
        
        IMPROVED: Better preprocessing pipeline for consistent encodings.

        Returns:
            Flattened float32 numpy array, or None on failure.
        """
        try:
            x, y, w, h = face_location
            
            # Add moderate padding for better context
            pad = int(min(w, h) * 0.15)  # Increased from 0.1
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_roi = gray[y1:y2, x1:x2]

            if face_roi.size == 0:
                return None

            # Resize to standard size
            face_resized = cv2.resize(face_roi, target_size, interpolation=cv2.INTER_CUBIC)
            
            # Apply bilateral filter to reduce noise while preserving edges
            face_filtered = cv2.bilateralFilter(face_resized, 5, 50, 50)
            
            # Histogram equalization for consistent lighting
            face_normalized = cv2.equalizeHist(face_filtered)
            
            return face_normalized.flatten().astype(np.float32)

        except Exception as e:
            print(f"⚠️  Face encoding error: {e}")
            return None

    def extract_face_roi(
        self,
        frame: np.ndarray,
        face_location: Tuple[int, int, int, int],
        target_size: Tuple[int, int] = (100, 100),
    ) -> Optional[np.ndarray]:
        """
        Extract grayscale face ROI (2D) for LBPH training/prediction.
        
        IMPROVED: Consistent preprocessing matching extract_face_encoding.
        """
        try:
            x, y, w, h = face_location
            
            # Use same padding as encoding for consistency
            pad = int(min(w, h) * 0.15)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_roi = gray[y1:y2, x1:x2]
            
            if face_roi.size == 0:
                return None
                
            face_resized = cv2.resize(face_roi, target_size, interpolation=cv2.INTER_CUBIC)
            
            # Apply bilateral filter
            face_filtered = cv2.bilateralFilter(face_resized, 5, 50, 50)
            
            # Histogram equalization
            return cv2.equalizeHist(face_filtered)
            
        except Exception as e:
            print(f"⚠️  ROI extraction error: {e}")
            return None

    # ───────────────────────────────── Training ───────────────────────────────

    def load_known_faces(self, employee_data: List[Dict[str, Any]]) -> None:
        """
        Load and train LBPH recognizer from employee face encodings stored in DB.

        IMPROVED: Better validation and error handling.

        Each entry: {employee_id, name, department, face_encoding (list of floats)}
        """
        self.known_face_encodings = []
        self.known_face_labels = []
        self.known_employees = []
        self.recognizer_trained = False

        valid_employees = [e for e in employee_data if e.get('face_encoding')]
        if not valid_employees:
            print("⚠️  No employee face encodings found.")
            return

        faces_for_training: List[np.ndarray] = []
        labels_for_training: List[int] = []
        
        total_samples = 0

        for idx, emp in enumerate(valid_employees):
            # Use all multi-sample encodings if available, else use single encoding
            all_encs = emp.get('face_encodings_all') or [emp['face_encoding']]

            added = 0
            for raw_enc in all_encs:
                try:
                    enc = np.array(raw_enc, dtype=np.float32)
                    
                    # Reshape flat encoding back to 2D (100x100)
                    if enc.ndim == 1 and enc.shape[0] == 10000:
                        enc_2d = enc.reshape(100, 100).astype(np.uint8)
                    elif enc.ndim == 2:
                        enc_2d = enc.astype(np.uint8)
                    else:
                        # Fallback: try to make a square
                        side = int(np.sqrt(enc.shape[0]))
                        if side * side != enc.shape[0]:
                            print(f"⚠️  Invalid encoding shape for {emp.get('name', 'Unknown')}: {enc.shape}")
                            continue
                        enc_2d = enc[:side*side].reshape(side, side).astype(np.uint8)

                    faces_for_training.append(enc_2d)
                    labels_for_training.append(idx)  # Same label for all samples of same person
                    added += 1
                    
                except Exception as e:
                    print(f"⚠️  Error processing encoding for {emp.get('name', 'Unknown')}: {e}")
                    continue

            if added > 0:
                self.known_employees.append(emp)
                total_samples += added
                print(f"   ✓ {emp.get('name', 'Unknown'):20s} ({emp.get('employee_id', '?'):10s}) - {added:2d} sample(s)")

        if len(faces_for_training) == 0:
            print("❌ No valid face encodings to train on!")
            return

        # Train LBPH recognizer
        try:
            self.recognizer.train(faces_for_training, np.array(labels_for_training, dtype=np.int32))
            self.known_face_labels = labels_for_training
            self.recognizer_trained = True
            print(f"✅ LBPH trained: {len(valid_employees)} employee(s), {total_samples} total samples")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            self.recognizer_trained = False

    # ───────────────────────────────── Recognition ────────────────────────────

    def identify_employee(
        self,
        face_roi_2d: np.ndarray,
        confidence_threshold: float = 50.0,  # STRICTER: Changed from 70.0
    ) -> Optional[Dict[str, Any]]:
        """
        Identify an employee from a 2D grayscale face ROI.

        IMPROVED: Stricter threshold and better confidence reporting.

        Args:
            face_roi_2d: 2D grayscale face image (100x100).
            confidence_threshold: Max LBPH distance (lower = stricter).
                                  Recommended: 40-60 for good accuracy.
                                  0-40: Excellent match
                                  40-60: Good match
                                  60-80: Acceptable
                                  80+: Poor/reject

        Returns:
            Employee dict with 'raw_confidence' and 'confidence_score' keys, or None if unrecognized.
        """
        if not self.recognizer_trained or not self.known_employees:
            return None

        try:
            label, confidence = self.recognizer.predict(face_roi_2d)
            
            # Debug output for threshold tuning
            if confidence <= confidence_threshold:
                emp = dict(self.known_employees[label])
                emp['raw_confidence'] = float(confidence)
                # Normalized confidence score (0-1, where 1 is best)
                emp['confidence_score'] = max(0.0, 1.0 - (confidence / 100.0))
                return emp
            else:
                # Log near-misses for debugging
                if confidence <= confidence_threshold * 1.5:
                    try:
                        near_emp = self.known_employees[label]
                        print(f"   [Near miss] {near_emp.get('name', '?')} - confidence={confidence:.1f} (threshold={confidence_threshold})")
                    except:
                        pass
                return None
                
        except Exception as e:
            print(f"⚠️  Recognition error: {e}")
            return None
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Return training statistics for debugging."""
        return {
            'trained': self.recognizer_trained,
            'num_employees': len(self.known_employees),
            'num_labels': len(set(self.known_face_labels)) if self.known_face_labels else 0,
            'total_samples': len(self.known_face_labels),
            'avg_samples_per_employee': (
                len(self.known_face_labels) / len(self.known_employees) 
                if self.known_employees else 0
            ),
        }