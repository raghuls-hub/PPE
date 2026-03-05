import ast, os, sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

files = [
    'config.py', 'db.py', 'app.py',
    'services/__init__.py',
    'services/ppe_service.py',
    'services/fire_service.py',
    'services/fall_service.py',
    'services/face_service.py',
    'services/attendance_service.py',
    'services/admin_service.py',
    'utils/__init__.py',
    'utils/db_connection.py',
    'utils/report_generator.py',
    'utils/sound_notification.py',
    'engine/__init__.py',
    'engine/monitor_engine.py',
    'engine/attendance_engine.py',
    'pages/__init__.py',
    'pages/camera_page.py',
    'pages/attendance_page.py',
    'pages/live_monitor_page.py',
]

all_ok = True
for f in files:
    try:
        with open(f, encoding='utf-8') as fh:
            ast.parse(fh.read())
        print(f'  OK   {f}')
    except SyntaxError as e:
        print(f'  ERR  {f}: line {e.lineno} — {e.msg}')
        all_ok = False
    except FileNotFoundError:
        print(f'  MISS {f}')
        all_ok = False

print()
print('ALL SYNTAX OK!' if all_ok else 'ERRORS FOUND — see above.')
