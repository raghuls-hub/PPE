"""
Video Utilities for handling different stream sources.
Includes Google Drive URL conversion for direct streaming.
"""

import re


def _extract_gdrive_id(url: str):
    match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    match = re.search(r'id=([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    return None


def convert_gdrive_url(url: str) -> str:
    """
    Resolves a Google Drive link. OpenCV/FFMPEG often drops Drive HTTP streams 
    due to missing cookies/headers. This securely downloads it to a temp cache 
    file and returns the local path for flawless playback.
    """
    if not url or "drive.google.com" not in url:
        return url

    file_id = _extract_gdrive_id(url)
    if not file_id:
        return url

    import os, tempfile
    cache_path = os.path.join(tempfile.gettempdir(), f"gdrive_cache_{file_id}.mp4")
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1024:
        print(f"[VIDEO] Using cached local Drive video: {cache_path}")
        return cache_path

    print(f"[VIDEO] Downloading Google Drive video to cache (id: {file_id})...")
    try:
        import requests
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})

        dl_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        resp = session.get(dl_url, stream=True, allow_redirects=True, timeout=15)

        # Handle virus scan prompt for large files
        if "text/html" in resp.headers.get("Content-Type", ""):
            import re
            match = re.search(r'confirm=([0-9A-Za-z_-]+)', resp.text)
            if match:
                confirm = match.group(1)
                resp = session.get(
                    f"https://drive.google.com/uc?export=download&confirm={confirm}&id={file_id}",
                    stream=True, allow_redirects=True, timeout=15
                )
            else:
                # Newer Drive: uuid-based confirm
                match = re.search(r'"downloadUrl":"([^"]+)"', resp.text)
                if match:
                    import html as _html
                    new_url = _html.unescape(match.group(1))
                    resp = session.get(new_url, stream=True, allow_redirects=True, timeout=15)

        resp.raise_for_status()
        
        with open(cache_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk: 
                    f.write(chunk)
                    
        print(f"[VIDEO] Download complete. Streaming from {cache_path}")
        return cache_path

    except Exception as e:
        print(f"[VIDEO] Drive URL download failed: {e}")
        return url


def get_available_cameras() -> list:
    """
    Scans for available local camera indices (0-4).
    On Linux, checks /dev/video*. On Windows/Mac, tries to open them briefly.
    """
    import sys
    import os
    import cv2
    valid = []
    
    # Path-based check for Linux (fast, no console spam)
    if sys.platform.startswith("linux") and os.path.exists("/dev"):
        try:
            for f in os.listdir("/dev"):
                if f.startswith("video"):
                    try:
                        idx = int(f.replace("video", ""))
                        valid.append(idx)
                    except: pass
            return sorted(list(set(valid)))
        except: pass

    # Fallback/Windows: Try opening (might trigger FFMPEG warnings)
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            valid.append(i)
            cap.release()
    return valid


def safe_open_video_capture(src: str | int):
    """
    Opens cv2.VideoCapture with detection for missing camera indices
    to avoid console spam and handle failures gracefully.
    """
    import cv2
    
    # Strip whitespace to prevent FFMPEG parsing errors (e.g. " https://...")
    if isinstance(src, str):
        src = src.strip()
        
    # Resolve Drive URLs first
    if isinstance(src, str) and "drive.google.com" in src:
        src = convert_gdrive_url(src)
        
    # Convert string digit to int
    try:
        if isinstance(src, str) and src.isdigit():
            src = int(src)
    except: pass

    # If it's a local camera index (int), verify it exists
    if isinstance(src, int):
        available = get_available_cameras()
        if src not in available:
            print(f"[VIDEO] ⚠️ Camera index {src} not found. Available: {available}")
            return None

    # Open with FFMPEG backend for reliability
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        # Fallback without FFMPEG flag just in case
        cap = cv2.VideoCapture(src)
        
    return cap if cap.isOpened() else None
