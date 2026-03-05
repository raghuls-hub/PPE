"""
Sound Notification — plays a beep when attendance is marked.
"""

import threading
import sys


def play_attendance_sound(sound_file: str = "") -> None:
    def _play():
        try:
            if sys.platform == "win32":
                import winsound, os
                if sound_file and os.path.exists(sound_file):
                    winsound.PlaySound(sound_file, winsound.SND_FILENAME | winsound.SND_ASYNC)
                else:
                    winsound.MessageBeep(winsound.MB_ICONASTERISK)
            else:
                print("\a", end="", flush=True)
        except Exception:
            print("\a", end="", flush=True)

    threading.Thread(target=_play, daemon=True).start()
