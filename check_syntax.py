"""Quick syntax check for all project files."""
import ast, sys, os

BASE = os.path.dirname(os.path.abspath(__file__))

files = [
    "config.py",
    os.path.join("utils", "db_connection.py"),
    os.path.join("utils", "report_generator.py"),
    os.path.join("utils", "sound_notification.py"),
    os.path.join("services", "face_service.py"),
    os.path.join("services", "ppe_service.py"),
    os.path.join("services", "attendance_service.py"),
    os.path.join("services", "admin_service.py"),
    "admin_panel.py",
    "live_recognition.py",
    "main.py",
]

all_ok = True
for rel in files:
    full = os.path.join(BASE, rel)
    try:
        with open(full, encoding="utf-8") as fh:
            ast.parse(fh.read())
        print(f"  OK   {rel}")
    except SyntaxError as e:
        print(f"  FAIL {rel}: line {e.lineno} — {e.msg}")
        all_ok = False
    except FileNotFoundError:
        print(f"  MISS {rel}: file not found")
        all_ok = False

print()
print("RESULT:", "ALL PASSED ✅" if all_ok else "FAILURES ❌")
sys.exit(0 if all_ok else 1)
