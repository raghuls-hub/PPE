"""
Local Camera Streamer
---------------------
Run this script ON YOUR LAPTOP to host your local webcam (index 0) over the network.

Setup:
    pip install flask opencv-python

How to use:
    1. Run this script: python local_streamer.py
    2. Expose it over the internet using ngrok (in a new terminal):
       ngrok http 5000
    3. Copy the ngrok URL it gives you and add "/video_feed" to the end.
       Example: https://xxxx.ngrok-free.app/video_feed
    4. Paste that entire URL into the "Stream URL" field in your Colab Dashboard!
"""

import cv2
from flask import Flask, Response

app = Flask(__name__)
camera = None

def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
    return camera

def generate_frames():
    cam = get_camera()
    while True:
        success, frame = cam.read()
        if not success:
            continue
        
        # Resize to 640x480 for smooth internet transmission
        frame = cv2.resize(frame, (640, 480))
            
        # Encode as JPEG with decent quality
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()
        
        # Yield the multipart stream byte payload
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "Camera Streamer Running! Point your Colab App to <b>/video_feed</b>"

if __name__ == '__main__':
    print("=" * 60)
    print(" Starting Local Camera Streamer on port 5000...")
    print(" You can preview it locally at http://localhost:5000/video_feed")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, threaded=True)
