"""
Demo Video Analyzer — demo_analyze.py

Runs a local video file through ONE detection model at a time.
Choose which model to run via the --model argument.

Usage:
    python demo_analyze.py path/to/video.mp4 --model ppe
    python demo_analyze.py path/to/video.mp4 --model fire
    python demo_analyze.py path/to/video.mp4 --model fall
    python demo_analyze.py path/to/video.mp4 --model fall --output result.mp4

Keys while running:
  Q or ESC -- quit
  SPACE     -- pause / resume
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg


# ── Performance knobs ─────────────────────────────────────────────────────────
INFER_WIDTH      = 416    # downscale width for inference (speeds up YOLO)
TARGET_FPS       = 15     # display throttle
EVERY_N          = 2      # run inference every N frames

# Demo-safe alert thresholds (stricter than live mode to avoid false positives)
FALL_CONF_DEFAULT  = 0.65
FALL_THRESH        = 15   # consecutive inference frames before fall alert
PPE_THRESH         = 20
FIRE_THRESH        = 8


# ── Model loaders ─────────────────────────────────────────────────────────────

def load_ppe():
    from services.ppe_service import PPEService
    print("Loading PPE model: " + cfg.PPE_MODEL_PATH)
    svc = PPEService(cfg.PPE_MODEL_PATH)
    return svc

def load_fire():
    from services.fire_service import FireService
    print("Loading Fire model: " + cfg.FIRE_MODEL_PATH)
    return FireService(cfg.FIRE_MODEL_PATH)

def load_fall(conf):
    from services.fall_service import FallService
    print("Loading Fall model (conf=%.2f): %s" % (conf, cfg.FALL_MODEL_PATH))
    svc = FallService(cfg.FALL_MODEL_PATH)
    svc.confidence_threshold = conf
    return svc


# ── HUD ───────────────────────────────────────────────────────────────────────

def draw_hud(frame, frame_n, fps, alert, alert_label, info_text, elapsed_sec):
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 32), (20, 20, 20), -1)
    mm, ss = divmod(int(elapsed_sec), 60)
    hh, mm = divmod(mm, 60)
    cv2.putText(frame,
                "Frame:%d  FPS:%.1f  %02d:%02d:%02d" % (frame_n, fps, hh, mm, ss),
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Alert badge (top-right)
    if alert_label:
        col = (0, 40, 220) if alert else (50, 50, 50)
        (tw, _), _ = cv2.getTextSize(alert_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x = w - tw - 24
        cv2.rectangle(frame, (x, 4), (w - 8, 28), col, -1 if alert else 1)
        cv2.putText(frame, alert_label, (x + 6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Alert border
    if alert:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 30, 220), 6)

    # Bottom info text
    if info_text:
        cv2.rectangle(frame, (0, h - 36), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, info_text, (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 1)

    return frame


# ── Scale boxes from inference size back to display size ──────────────────────

def scale_boxes(dets, sx, sy):
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        d.bbox = (int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))
    return dets


# ── Analysis runners ──────────────────────────────────────────────────────────

def run_ppe(cap, src_fps, width, height, infer_h, sx, sy, writer, display, frame_delay_ms, total_frames):
    svc = load_ppe()
    print("\nRunning PPE detection...\nPress SPACE to pause, Q/ESC to quit.\n")

    last_dets = []
    alert = False
    viol_count = 0
    missing_ppe = []
    frame_n = fps_cnt = 0
    fps = 0.0
    t_fps = t_start = time.time()
    paused = False
    frame = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\nEnd of video.")
                break

            frame_n += 1
            fps_cnt += 1

            if frame_n % EVERY_N == 0:
                small = cv2.resize(frame, (INFER_WIDTH, infer_h))
                last_dets = scale_boxes(svc.detect_ppe(small), sx, sy)
                detected  = svc.get_detected_class_names(last_dets)
                required  = list(cfg.AVAILABLE_PPE_OPTIONS)
                ok, missing_ppe, _ = svc.verify_ppe(detected, required)
                viol_count = 0 if ok else viol_count + 1
                new_alert = viol_count >= PPE_THRESH
                if new_alert and not alert:
                    print("\n  PPE VIOLATION at frame %d: missing %s" % (frame_n, missing_ppe))
                alert = new_alert

            svc.draw_ppe_boxes(frame, last_dets)

            now = time.time()
            if now - t_fps >= 1.0:
                fps = fps_cnt / (now - t_fps)
                t_fps = now; fps_cnt = 0

            info = ("MISSING: " + ", ".join(missing_ppe)) if alert and missing_ppe else ""
            frame = draw_hud(frame, frame_n, fps, alert, "PPE!", info, now - t_start)

            pct = frame_n / max(total_frames, 1) * 100
            print("\r  [%5.1f%%]  Frame %d/%d  |  Violations:%d  |  FPS:%.1f"
                  % (pct, frame_n, total_frames, viol_count, fps), end="", flush=True)

            if writer:
                writer.write(frame)

        if display and frame is not None:
            cv2.imshow("PPE Detection -- Demo", frame)
            key = cv2.waitKey(frame_delay_ms) & 0xFF
            if key in (ord("q"), 27):
                print("\n  Stopped by user."); break
            elif key == ord(" "):
                paused = not paused
                print("\n  " + ("Paused" if paused else "Resumed"))


def run_fire(cap, src_fps, width, height, infer_h, sx, sy, writer, display, frame_delay_ms, total_frames):
    from services.fire_service import FireService
    svc = load_fire()
    print("\nRunning Fire detection...\nPress SPACE to pause, Q/ESC to quit.\n")

    last_dets = []
    alert = False
    fire_count = 0
    frame_n = fps_cnt = 0
    fps = 0.0
    t_fps = t_start = time.time()
    paused = False
    frame = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\nEnd of video.")
                break

            frame_n += 1
            fps_cnt += 1

            if frame_n % EVERY_N == 0:
                small = cv2.resize(frame, (INFER_WIDTH, infer_h))
                last_dets = scale_boxes(svc.detect_fire(small), sx, sy)
                fire_det  = svc.has_fire(last_dets)
                fire_count = fire_count + 1 if fire_det else 0
                new_alert  = fire_count >= FIRE_THRESH
                if new_alert and not alert:
                    print("\n  FIRE ALERT at frame %d" % frame_n)
                alert = new_alert

            svc.draw_fire_boxes(frame, last_dets)

            now = time.time()
            if now - t_fps >= 1.0:
                fps = fps_cnt / (now - t_fps)
                t_fps = now; fps_cnt = 0

            info = "FIRE DETECTED -- EVACUATE" if alert else ""
            frame = draw_hud(frame, frame_n, fps, alert, "FIRE", info, now - t_start)

            pct = frame_n / max(total_frames, 1) * 100
            print("\r  [%5.1f%%]  Frame %d/%d  |  FireFrames:%d  |  FPS:%.1f"
                  % (pct, frame_n, total_frames, fire_count, fps), end="", flush=True)

            if writer:
                writer.write(frame)

        if display and frame is not None:
            cv2.imshow("Fire Detection -- Demo", frame)
            key = cv2.waitKey(frame_delay_ms) & 0xFF
            if key in (ord("q"), 27):
                print("\n  Stopped by user."); break
            elif key == ord(" "):
                paused = not paused
                print("\n  " + ("Paused" if paused else "Resumed"))


def run_fall(cap, src_fps, width, height, infer_h, sx, sy, writer, display, frame_delay_ms, total_frames, fall_conf, fall_thresh):
    svc = load_fall(fall_conf)
    print("\nRunning Fall detection (conf=%.2f, thresh=%d)..." % (fall_conf, fall_thresh))
    print("Press SPACE to pause, Q/ESC to quit.\n")

    last_dets  = []
    alert      = False
    consec     = 0
    frame_n = fps_cnt = 0
    fps = 0.0
    t_fps = t_start = time.time()
    paused = False
    frame = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\nEnd of video.")
                break

            frame_n += 1
            fps_cnt += 1

            if frame_n % EVERY_N == 0:
                small = cv2.resize(frame, (INFER_WIDTH, infer_h))
                last_dets = scale_boxes(svc.detect_fall(small), sx, sy)
                has_fall  = any(d.is_fall for d in last_dets)
                consec    = consec + 1 if has_fall else max(0, consec - 1)
                new_alert = consec >= fall_thresh
                if new_alert and not alert:
                    print("\n  FALL ALERT at frame %d" % frame_n)
                alert = new_alert

            svc.draw_fall_boxes(frame, last_dets)
            if alert:
                svc.draw_fall_alert(frame, True)

            now = time.time()
            if now - t_fps >= 1.0:
                fps = fps_cnt / (now - t_fps)
                t_fps = now; fps_cnt = 0

            info = "FALL DETECTED -- CALL FOR HELP" if alert else ""
            frame = draw_hud(frame, frame_n, fps, alert, "FALL", info, now - t_start)

            pct = frame_n / max(total_frames, 1) * 100
            print("\r  [%5.1f%%]  Frame %d/%d  |  Consec:%d  |  FPS:%.1f"
                  % (pct, frame_n, total_frames, consec, fps), end="", flush=True)

            if writer:
                writer.write(frame)

        if display and frame is not None:
            cv2.imshow("Fall Detection -- Demo", frame)
            key = cv2.waitKey(frame_delay_ms) & 0xFF
            if key in (ord("q"), 27):
                print("\n  Stopped by user."); break
            elif key == ord(" "):
                paused = not paused
                print("\n  " + ("Paused" if paused else "Resumed"))


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Demo video analyzer — run ONE detection model at a time",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Models:
  ppe   — PPE compliance detection (helmet, vest, mask)
  fire  — Fire & smoke detection
  fall  — Person fall detection

Examples:
  python demo_analyze.py video.mp4 --model ppe
  python demo_analyze.py video.mp4 --model fall --fall-conf 0.75 --output out.mp4
  python demo_analyze.py video.mp4 --model fire --no-display
        """
    )
    parser.add_argument("video",         help="Path to the input video file")
    parser.add_argument("--model",  "-m", required=True,
                        choices=["ppe", "fire", "fall"],
                        help="Which model to run: ppe | fire | fall")
    parser.add_argument("--output", "-o", default=None,
                        help="Save annotated output video to this path")
    parser.add_argument("--no-display",   action="store_true",
                        help="Run headless (no OpenCV window)")
    parser.add_argument("--fall-conf",    type=float, default=FALL_CONF_DEFAULT,
                        metavar="CONF",
                        help="Fall confidence threshold 0-1 (default %.2f)" % FALL_CONF_DEFAULT)
    parser.add_argument("--fall-thresh",  type=int,   default=FALL_THRESH,
                        metavar="N",
                        help="Consecutive inference frames before fall alert (default %d)" % FALL_THRESH)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("ERROR: Cannot open video: " + args.video)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    infer_h = int(height * INFER_WIDTH / width)
    sx = width  / INFER_WIDTH
    sy = height / infer_h

    print("\nVideo : " + args.video)
    print("  Size   : %dx%d  |  Src FPS: %.1f  |  Frames: %d" % (width, height, src_fps, total_frames))
    print("  Model  : %s" % args.model.upper())

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, src_fps, (width, height))
        print("  Output : " + args.output)

    frame_delay_ms = max(1, int(1000 / TARGET_FPS))
    display = not args.no_display

    try:
        if args.model == "ppe":
            run_ppe(cap, src_fps, width, height, infer_h, sx, sy,
                    writer, display, frame_delay_ms, total_frames)
        elif args.model == "fire":
            run_fire(cap, src_fps, width, height, infer_h, sx, sy,
                     writer, display, frame_delay_ms, total_frames)
        elif args.model == "fall":
            run_fall(cap, src_fps, width, height, infer_h, sx, sy,
                     writer, display, frame_delay_ms, total_frames,
                     args.fall_conf, args.fall_thresh)
    finally:
        cap.release()
        if writer:
            writer.release()
            print("\n\nSaved to: " + args.output)
        if display:
            cv2.destroyAllWindows()
        print("\nDone.\n")


if __name__ == "__main__":
    main()
