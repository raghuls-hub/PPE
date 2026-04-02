import sys
import os
from pathlib import Path

# Add root to sys.path
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.video_utils import get_available_cameras, safe_open_video_capture

def test():
    print("--- Camera Validation Test ---")
    available = get_available_cameras()
    print(f"Detected cameras: {available}")
    
    test_idx = 0
    print(f"\nAttempting to open camera index {test_idx} safely...")
    cap = safe_open_video_capture(test_idx)
    
    if cap:
        print(f"Success! Camera {test_idx} is opened.")
        cap.release()
    else:
        print(f"Safe Skip: Camera {test_idx} is NOT available or failed to open.")
        print("Notice: No FFMPEG index-out-of-range error should be visible above if on Linux.")

if __name__ == "__main__":
    test()
