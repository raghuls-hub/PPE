import os
from flask import Flask, send_from_directory

app = Flask(__name__)

# 1. Input: Path of the folder containing your videos
VIDEO_FOLDER = r"d:\Antigravity\DT - PPE\videos"
PORT = 5000

# Route A: The actual video file streamer (what Colab needs!)
@app.route('/stream/<filename>')
def stream_video(filename):
    # send_from_directory automatically handles chunking and video seeking
    return send_from_directory(VIDEO_FOLDER, filename)

# Route B: The API endpoint for your iframe (for browser testing)
@app.route('/api/play/<filename>')
def play_video_api(filename):
    # This generates a borderless, auto-playing video player meant to live inside an iframe
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{filename}</title>
        <style>
            body {{ margin: 0; background-color: transparent; overflow: hidden; display: flex; justify-content: center; align-items: center; height: 100vh; }}
            video {{ width: 100%; height: 100%; object-fit: contain; }}
        </style>
    </head>
    <body>
        <video autoplay controls muted>
            <source src="/stream/{filename}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </body>
    </html>
    """
    return html

if __name__ == '__main__':
    # Ensure the folder exists
    if not os.path.exists(VIDEO_FOLDER):
        print(f"Error: The folder '{VIDEO_FOLDER}' does not exist.")
        exit(1)

    valid_extensions = ('.mp4', '.webm', '.ogg', '.mov')
    
    try:
        videos = [f for f in os.listdir(VIDEO_FOLDER) if f.lower().endswith(valid_extensions)]
        
        # ── Start PyNgrok Automatically ──
        from pyngrok import ngrok
        print("🌐 Starting secure Ngrok tunnel so Colab can stream the videos...")
        tunnel = ngrok.connect(PORT, bind_tls=True)
        public_url = tunnel.public_url
        
        print(f"\n✅ Server started! Found {len(videos)} videos.")
        print("=" * 70)
        print("📋 COPY & PASTE THESE NGROK LINKS INTO YOUR COLAB DASHBOARD CAMERAS:\n")
        
        for video in videos:
            # Replace spaces with %20 for valid URLs
            safe_video_name = video.replace(" ", "%20")
            print(f"Video: {video}")
            print(f"Direct Stream URL -> {public_url}/stream/{safe_video_name}")
            print("-" * 70)
            
        print("Press Ctrl+C to stop the server.")
        print("=" * 70)
        
        # Start the local server
        app.run(host='0.0.0.0', port=PORT, use_reloader=False)
        
    except ImportError:
        print("❌ Error: pyngrok is not installed. Please run: pip install pyngrok")
    except Exception as e:
        print(f"❌ An error occurred: {e}")
