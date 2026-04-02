"""
MJPEG Stream Server — serves live camera frames over HTTP as MJPEG.

A single lightweight HTTPServer runs in a daemon thread.
Each stream maps to a URL:  http://localhost:<PORT>/stream/<stream_id>

Usage:
    from utils.stream_server import get_stream_server
    server = get_stream_server(port=8765)
    
    # Register a provider function that returns a JPEG byte string or None
    server.register_stream("cam_1", lambda: engine.get_state("cam_1").get("frame"))
    
    # Then embed in Streamlit:
    # <img src="http://localhost:8765/stream/cam_1">
"""

from __future__ import annotations

import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Callable, Dict

_BOUNDARY = b"mjpegframe"

_server_instance: Optional["MJPEGServer"] = None
_server_lock     = threading.Lock()


class _MJPEGHandler(BaseHTTPRequestHandler):
    """Handles /stream/<stream_id> and /ping."""

    registry: Dict[str, Callable[[], Optional[bytes]]] = {}

    # ── Silence default request logging ──────────────────────────────────────
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        path = self.path.lstrip("/")

        # Health-check
        if path == "ping":
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"pong")
            return

        # HTML wrapper endpoint: /view/<stream_id>
        if path.startswith("view/"):
            stream_id = path[len("view/"):]
            if "?" in stream_id:
                stream_id = stream_id.split("?")[0]
            
            # Simple wrapper to ensure the <img> fits the iframe nicely
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body, html {{ margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background: #000; }}
                    img {{ width: 100%; height: 100%; object-fit: contain; display: block; }}
                </style>
            </head>
            <body>
                <img src="/stream/{stream_id}">
            </body>
            </html>
            """
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
            return

        # MJPEG stream endpoint: /stream/<stream_id>
        if path.startswith("stream/"):
            stream_id = path[len("stream/"):]
            if "?" in stream_id:
                stream_id = stream_id.split("?")[0]
            
            print(f"[STREAM] Request received for stream_id: {stream_id}")
            if stream_id in self.registry:
                print(f"[STREAM] Registry hit for {stream_id}, starting stream loop...")
                self._stream(stream_id)
            else:
                print(f"[STREAM] Registry MISS for {stream_id}. Returning 404. Current registry keys: {list(self.registry.keys())}")
                self.send_response(404)
                self.end_headers()
            return

        self.send_response(404)
        self.end_headers()

    def _stream(self, stream_id: str):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Type",
                         f"multipart/x-mixed-replace; boundary={_BOUNDARY.decode()}")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()

        blank = _blank_jpeg()   # grey placeholder for when no frame yet
        provider = self.registry[stream_id]

        while True:
            try:
                frame = provider()
                if frame is None:
                    # Provide a grey frame immediately so browser doesn't time out waiting
                    frame = blank
                
                header = (
                    b"--" + _BOUNDARY + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                )
                self.wfile.write(header + frame + b"\r\n")
                # High throughput
                time.sleep(0.01)
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                # Browser disconnected or iframe reloaded
                print(f"[STREAM] Client disconnected from {stream_id} (Reason: {type(e).__name__})")
                break
            except Exception as e:
                print(f"⚠️ Stream error on {stream_id}: {e}")
                break


class MJPEGServer:
    """Thin wrapper around HTTPServer — starts in a daemon thread."""

    def __init__(self, port: int = 8765):
        self.port = port
        self.base_url = f"http://localhost:{port}"
        self._try_expose_ngrok()
        
        self._httpd = ThreadingHTTPServer(("0.0.0.0", port), _MJPEGHandler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            daemon=True,
            name="MJPEGServer",
        )

    def _try_expose_ngrok(self):
        import os
        if "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ:
            try:
                from pyngrok import ngrok
                tunnel = ngrok.connect(self.port, bind_tls=True)
                self.base_url = tunnel.public_url
                print(f"🌐 Cloud Stream Server exposed over Ngrok at: {self.base_url}")
            except Exception as e:
                print(f"⚠️ Failed to wrap stream server with Ngrok: {e}")

    def start(self):
        self._thread.start()
        print(f"✅ MJPEG stream server started on http://localhost:{self.port}")

    def register_stream(self, stream_id: str, provider_fn: Callable[[], Optional[bytes]]):
        """Register a function that returns JPEG bytes for a given stream ID."""
        print(f"[STREAM SERVER] Registering stream provider for: {stream_id}")
        _MJPEGHandler.registry[stream_id] = provider_fn

    def unregister_stream(self, stream_id: str):
        print(f"[STREAM SERVER] Unregistering stream provider for: {stream_id}")
        _MJPEGHandler.registry.pop(stream_id, None)

    def stream_url(self, stream_id: str) -> str:
        return f"{self.base_url}/stream/{stream_id}"

    def view_url(self, stream_id: str) -> str:
        return f"{self.base_url}/view/{stream_id}"

    def is_running(self) -> bool:
        return self._thread.is_alive()

    def stop(self):
        self._httpd.shutdown()


# ── Singleton accessor ─────────────────────────────────────────────────────────

def get_stream_server(port: int = 8765) -> MJPEGServer:
    """
    Returns (and starts if needed) the global MJPEGServer.
    """
    global _server_instance
    with _server_lock:
        if _server_instance is None or not _server_instance.is_running():
            _server_instance = MJPEGServer(port=port)
            _server_instance.start()
    return _server_instance


# ── Helpers ────────────────────────────────────────────────────────────────────

def _blank_jpeg(w: int = 640, h: int = 360) -> bytes:
    """Generate a tiny grey placeholder JPEG (no OpenCV dependency)."""
    import struct, zlib

    def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    # Minimal grey JPEG (1×1 pixel, expand via CSS)
    return bytes([
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
        0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
        0x00, 0x10, 0x0B, 0x0C, 0x0E, 0x0C, 0x0A, 0x10, 0x0E, 0x0D, 0x0E, 0x12,
        0x11, 0x10, 0x13, 0x18, 0x28, 0x1A, 0x18, 0x16, 0x16, 0x18, 0x31, 0x23,
        0x25, 0x1D, 0x28, 0x3A, 0x33, 0x3D, 0x3C, 0x39, 0x33, 0x38, 0x37, 0x40,
        0x48, 0x5C, 0x4E, 0x40, 0x44, 0x57, 0x45, 0x37, 0x38, 0x50, 0x6D, 0x51,
        0x57, 0x5F, 0x62, 0x67, 0x68, 0x67, 0x3E, 0x4D, 0x71, 0x79, 0x70, 0x64,
        0x78, 0x5C, 0x65, 0x67, 0x63, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
        0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
        0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
        0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
        0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
        0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72,
        0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45,
        0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
        0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
        0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3,
        0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
        0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9,
        0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
        0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4,
        0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01,
        0x00, 0x00, 0x3F, 0x00, 0xFB, 0xD4, 0xFF, 0xD9,
    ])
