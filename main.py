"""
main.py — Entry point for Smart PPE Detection & Face Recognition Attendance System

Usage:
    python main.py                           # Live camera mode (default)
    python main.py --mode live               # Live camera mode
    python main.py --mode video --source path/to/video.mp4
    python main.py --admin                   # Open admin panel
"""

import argparse
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║     SMART PPE DETECTION & FACE RECOGNITION ATTENDANCE        ║
║         Powered by YOLO v8 + OpenCV LBPH + MongoDB           ║
╚══════════════════════════════════════════════════════════════╝
"""

CONTROLS = """
⌨️  CONTROLS
  q         — Quit
  r         — Reload employees from database
  n         — Register new employee (live mode only)
  s         — Take screenshot
  e         — Export today's attendance report (CSV)
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smart PPE Detection & Face Recognition Attendance System"
    )
    parser.add_argument(
        "--mode",
        choices=["live", "video", "fall"],
        default="live",
        help="Operation mode: 'live' (webcam, PPE+face), 'video' (file, PPE+face+fire+fall), 'fall' (standalone fall detection on a video file)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to video file (required when --mode video or --mode fall)",
    )
    parser.add_argument(
        "--admin",
        action="store_true",
        help="Open the admin configuration panel",
    )
    return parser.parse_args()


def run_admin():
    """Launch the admin panel."""
    import admin_panel
    admin_panel.main()


def run_live():
    """Launch live webcam recognit — PPE + face only, no fire/fall."""
    from live_recognition import LiveFaceRecognition

    print(BANNER)
    print(CONTROLS)
    print("🎥 Starting LIVE mode (PPE + Face Recognition only)...")

    recognizer = LiveFaceRecognition(enable_fire=False, enable_fall=False)
    recognizer.run_live()


def run_video(source: str):
    """Process a video file — PPE + face + fire + fall detection."""
    from live_recognition import LiveFaceRecognition

    print(BANNER)
    print(f"🎬 Starting VIDEO mode: {source}")
    print("   PPE + Face + 🔥 Fire + 🚨 Fall detection enabled")

    recognizer = LiveFaceRecognition(enable_fire=True, enable_fall=True)
    recognizer.run_video(source)


def run_fall_video(source: str):
    """Run standalone fall-only detection on a video and save annotated output."""
    import os
    from types import SimpleNamespace
    from test_fall_detection import run

    print(BANNER)
    print(f"🚨 Starting FALL DETECTION mode: {source}")

    default_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Fall.pt")
    fall_args = SimpleNamespace(
        input=source,
        output=None,
        model=default_model,
        conf=0.50,
        iou=0.45,
        fall_frames=5,
        cooldown=30,
        skip=0,
        preview=True,
    )
    run(fall_args)


def main():
    args = parse_args()

    try:
        if args.admin:
            run_admin()
            return

        if args.mode == "video":
            if not args.source:
                print("❌ --source is required in video mode.")
                print("   Example: python main.py --mode video --source video.mp4")
                sys.exit(1)
            run_video(args.source)

        elif args.mode == "fall":
            if not args.source:
                print("❌ --source is required in fall mode.")
                print("   Example: python main.py --mode fall --source video.mp4")
                sys.exit(1)
            run_fall_video(args.source)

        else:  # live (default)
            run_live()

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user.")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
