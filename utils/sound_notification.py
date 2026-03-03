"""
Sound Notification utility — plays a beep when attendance is marked.
Uses winsound (built-in on Windows). Falls back to console bell on other OS.
"""

import threading
import sys


def play_attendance_sound(sound_file: str = "") -> None:
    """
    Play an audio notification in a background thread.
    On Windows: plays a WAV file if it exists, otherwise plays a system beep.
    On other OS: prints console bell character.

    Args:
        sound_file: Optional path to a WAV file.
    """
    def _play():
        try:
            if sys.platform == "win32":
                import winsound
                import os
                if sound_file and os.path.exists(sound_file):
                    winsound.PlaySound(sound_file, winsound.SND_FILENAME | winsound.SND_ASYNC)
                else:
                    # Play Windows system "Asterisk" beep
                    winsound.MessageBeep(winsound.MB_ICONASTERISK)
            else:
                print("\a", end="", flush=True)
        except Exception:
            print("\a", end="", flush=True)

    thread = threading.Thread(target=_play, daemon=True)
    thread.start()
