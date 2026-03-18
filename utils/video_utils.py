"""
Video Utilities for handling different stream sources.
Includes Google Drive URL conversion for direct streaming.
"""

import re

def convert_gdrive_url(url: str) -> str:
    """
    Converts a Google Drive shareable link into a direct download/stream link.
    Supports:
    - https://drive.google.com/file/d/FILE_ID/view?usp=sharing
    - https://drive.google.com/open?id=FILE_ID
    - https://drive.google.com/uc?id=FILE_ID
    """
    if not url or "drive.google.com" not in url:
        return url

    # Extract FILE_ID using regex
    # Pattern 1: /file/d/FILE_ID/
    match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url)
    if match:
        file_id = match.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"

    # Pattern 2: id=FILE_ID
    match = re.search(r'id=([a-zA-Z0-9_-]+)', url)
    if match:
        file_id = match.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"

    return url
