import cv2
import socket
import time

CAMERA_IP   = "172.16.144.207"
CAMERA_PORT = 8080
STREAM_URL  = f"http://{CAMERA_IP}:{CAMERA_PORT}/video"

# ── Network reachability check ────────────────────────────────────────────────
def is_host_reachable(ip, port, timeout=3):
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except (OSError, socket.timeout):
        return False

print(f"[INFO] Checking connectivity to {CAMERA_IP}:{CAMERA_PORT} ...")
if not is_host_reachable(CAMERA_IP, CAMERA_PORT):
    print(f"[ERROR] Cannot reach {CAMERA_IP}:{CAMERA_PORT}")
    print("  • Make sure your PC and the phone/camera are on the same Wi-Fi network.")
    print("  • Verify the IP Camera app is running and streaming on the phone.")
    print("  • Try opening the stream URL in a browser:", STREAM_URL)
    exit(1)

print(f"[INFO] Host reachable. Opening stream: {STREAM_URL}")

# ── Stream capture ────────────────────────────────────────────────────────────
MAX_RETRIES = 3
cap = None

for attempt in range(1, MAX_RETRIES + 1):
    cap = cv2.VideoCapture(STREAM_URL)
    if cap.isOpened():
        print(f"[INFO] Stream opened successfully (attempt {attempt})")
        break
    print(f"[WARN] Attempt {attempt}/{MAX_RETRIES} failed, retrying in 3 s …")
    cap.release()
    time.sleep(3)

if not cap or not cap.isOpened():
    print("[ERROR] Could not open the camera stream after all retries.")
    exit(1)

print("[INFO] Streaming — press ESC to quit.")
while True:
    ret, frame = cap.read()

    if not ret:
        print("[WARN] Lost connection to stream. Attempting reconnect …")
        cap.release()
        time.sleep(2)
        cap = cv2.VideoCapture(STREAM_URL)
        if not cap.isOpened():
            print("[ERROR] Reconnect failed. Exiting.")
            break
        continue

    frame = cv2.resize(frame, (640, 360))
    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) == 27:   # ESC
        break

cap.release()
cv2.destroyAllWindows()