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
    Resolves a Google Drive shareable link into a direct URL
    that OpenCV/FFMPEG can stream without downloading.
    Follows all redirects to get the final content URL.
    """
    if not url or "drive.google.com" not in url:
        return url

    file_id = _extract_gdrive_id(url)
    if not file_id:
        return url

    try:
        import requests
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})

        dl_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        resp = session.get(dl_url, stream=True, allow_redirects=True, timeout=15)

        # Large files: Drive shows a confirmation HTML page
        if "text/html" in resp.headers.get("Content-Type", ""):
            # Try confirm token
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
                    return _html.unescape(match.group(1))

        # Return the final resolved URL after all redirects
        final_url = resp.url
        print(f"[VIDEO] Resolved Drive URL: {final_url[:80]}...")
        return final_url

    except Exception as e:
        print(f"[VIDEO] Drive URL resolve failed: {e}, using fallback")
        return f"https://drive.google.com/uc?export=download&id={file_id}"
