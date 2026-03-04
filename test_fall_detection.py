"""
test_fall_detection.py
──────────────────────
Standalone fall detection tester.

Usage:
    python test_fall_detection.py --input <video_path> [options]

Examples:
    # Process a video and save annotated output
    python test_fall_detection.py --input sample.mp4

    # Custom output path and confidence
    python test_fall_detection.py --input sample.mp4 --output result.mp4 --conf 0.40

    # Also show a live preview window while processing
    python test_fall_detection.py --input sample.mp4 --preview

    # Use a different model
    python test_fall_detection.py --input sample.mp4 --model fall_detection.pt
"""

import argparse
import os
import sys
import time

import cv2

# ── Import the project's FallService ──────────────────────────────────────────
# Add project root to path so we can import `services`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from services.fall_service import FallService


# ─────────────────────────────── Helpers ──────────────────────────────────────

def build_output_path(input_path: str) -> str:
    """Derive a default output filename from the input path."""
    base, ext = os.path.splitext(input_path)
    return f"{base}_fall_detected{ext or '.mp4'}"


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def draw_stats_overlay(frame, frame_num: int, total_frames: int, fps: float,
                       fall_count: int, consecutive: int, threshold: int):
    """Draw a small HUD in the top-left corner with processing stats."""
    h, w = frame.shape[:2]

    lines = [
        f"Frame : {frame_num}/{total_frames}",
        f"FPS   : {fps:.1f}",
        f"Falls : {fall_count}",
        f"Consec: {consecutive}/{threshold}",
    ]

    pad = 8
    line_h = 22
    box_w = 190
    box_h = pad * 2 + line_h * len(lines)

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for i, line in enumerate(lines):
        y = 10 + pad + line_h * (i + 1) - 4
        cv2.putText(frame, line, (18, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 230, 200), 1, cv2.LINE_AA)


# ─────────────────────────────── Main ─────────────────────────────────────────

def run(args):
    # ── Validate input ────────────────────────────────────────────────────────
    if not os.path.isfile(args.input):
        print(f"❌  Input file not found: {args.input}")
        sys.exit(1)

    model_path = 'D:\Antigravity\DT - PPE\Fall.pt'
    if not os.path.isfile(model_path):
        print(f"❌  Model file not found: {model_path}")
        sys.exit(1)

    output_path = args.output or build_output_path(args.input)

    print("=" * 60)
    print("  Fall Detection — Standalone Tester")
    print("=" * 60)
    print(f"  Input        : {args.input}")
    print(f"  Output       : {output_path}")
    print(f"  Model        : {model_path}")
    print(f"  Confidence   : {args.conf}")
    print(f"  IoU          : {args.iou}")
    print(f"  Fall frames  : {args.fall_frames}  (consecutive to trigger alert)")
    print(f"  Cooldown     : {args.cooldown}  frames")
    print(f"  Skip N frames: {args.skip}")
    print(f"  Preview      : {'on' if args.preview else 'off'}")
    print("=" * 60)

    # ── Load fall service ─────────────────────────────────────────────────────
    service = FallService(
        model_path=model_path,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        fall_frame_threshold=args.fall_frames,
        alert_cooldown_frames=args.cooldown,
    )

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"❌  Cannot open video: {args.input}")
        sys.exit(1)

    src_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n📹 Video: {src_w}×{src_h} @ {src_fps:.1f} fps  |  {total_frames} frames")

    # ── Output writer ─────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, src_fps, (src_w, src_h))
    if not writer.isOpened():
        print(f"❌  Cannot create output file: {output_path}")
        cap.release()
        sys.exit(1)

    # ── Process frames ────────────────────────────────────────────────────────
    frame_idx       = 0
    t_start         = time.time()
    last_detections = []
    alert_active    = False

    print("\n🎬 Processing… (press Q in preview window to abort)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # ── Run detection every N frames for speed ────────────────────────────
        if (frame_idx - 1) % max(1, args.skip + 1) == 0:
            last_detections = service.detect_fall(frame)
        # Always call update_fall_state so the cooldown ticks down every frame
        alert_active = service.update_fall_state(last_detections)

        # ── Annotate ──────────────────────────────────────────────────────────
        service.annotate_frame(frame, last_detections, alert_active)

        # ── Stats overlay ─────────────────────────────────────────────────────
        elapsed  = time.time() - t_start
        proc_fps = frame_idx / elapsed if elapsed > 0 else 0
        draw_stats_overlay(
            frame, frame_idx, total_frames, proc_fps,
            service.total_falls_logged,
            service.consecutive_fall_frames,
            service.fall_frame_threshold
        )

        # ── Write to output ───────────────────────────────────────────────────
        writer.write(frame)

        # ── Optional live preview ─────────────────────────────────────────────
        if args.preview:
            cv2.imshow("Fall Detection — Preview (Q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⚠️  Aborted by user.")
                break

        # ── Progress printer ──────────────────────────────────────────────────
        if frame_idx % 30 == 0 or frame_idx == total_frames:
            pct = (frame_idx / total_frames * 100) if total_frames else 0
            eta = (total_frames - frame_idx) / proc_fps if proc_fps > 0 else 0
            print(
                f"  [{frame_idx:>6}/{total_frames}]  {pct:5.1f}%  "
                f"| {proc_fps:5.1f} fps  "
                f"| ETA {format_time(eta)}  "
                f"| Falls: {service.total_falls_logged}  "
                f"| Consec: {service.consecutive_fall_frames}/{service.fall_frame_threshold}",
                end="\r",
            )

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    writer.release()
    if args.preview:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"\n\n✅ Done in {format_time(elapsed)}")
    print(f"   Total frames processed : {frame_idx}")
    print(f"   Fall events detected   : {service.total_falls_logged}")
    print(f"   Output saved to        : {os.path.abspath(output_path)}")


# ─────────────────────────────── CLI ──────────────────────────────────────────

def parse_args():
    default_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Fall.pt")

    parser = argparse.ArgumentParser(
        description="Test the Fall Detection model on a video file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument("--input",  "-i", required=True,
                        help="Path to the input video file.")

    # Optional
    parser.add_argument("--output", "-o", default=None,
                        help="Path for the annotated output video. "
                             "Default: <input>_fall_detected.<ext>")
    parser.add_argument("--model",  "-m", default=default_model,
                        help=f"Path to the YOLO fall model .pt file. "
                             f"Default: {default_model}")
    parser.add_argument("--conf",   "-c", type=float, default=0.50,
                        help="Confidence threshold (0–1). Default: 0.50")
    parser.add_argument("--iou",          type=float, default=0.45,
                        help="IoU (NMS) threshold (0–1). Default: 0.45")
    parser.add_argument("--fall-frames",  type=int,   default=5,
                        help="Consecutive fall frames required to trigger alert. Default: 5")
    parser.add_argument("--cooldown",     type=int,   default=30,
                        help="Alert-banner cooldown in frames. Default: 30")
    parser.add_argument("--skip",   "-s", type=int,   default=0,
                        help="Process every (skip+1)th frame (0 = every frame). Default: 0")
    parser.add_argument("--preview", "-p", action="store_true",
                        help="Show a live preview window while processing.")

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
