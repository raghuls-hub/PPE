"""
Admin Panel — CLI interface to configure required PPE items.
Run: python admin_panel.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config, ensure_directories
from utils.db_connection import db_connection
from services.admin_service import AdminService
from services.ppe_service import AVAILABLE_PPE_OPTIONS
from utils.report_generator import ReportGenerator
from services.attendance_service import AttendanceService


BANNER = """
╔══════════════════════════════════════════════════════╗
║       SMART PPE ATTENDANCE SYSTEM — ADMIN PANEL      ║
╚══════════════════════════════════════════════════════╝
"""


def print_menu():
    print("""
  [1] View current PPE requirements
  [2] Configure required PPE items
  [3] View today's attendance summary
  [4] Export today's attendance to CSV
  [5] List all employees
  [6] Exit
""")


def configure_ppe(admin_svc: AdminService) -> None:
    """Interactive PPE selection menu."""
    current = admin_svc.get_required_ppe()
    options = admin_svc.get_ppe_options()

    print("\n  Available PPE items:")
    selected = set(current)

    for i, item in enumerate(options, 1):
        status = "✅" if item in selected else "☐"
        print(f"    [{i}] {status} {item}")

    print("\n  Enter numbers to toggle (comma-separated), or press ENTER to keep current:")
    raw = input("  Choice: ").strip()

    if not raw:
        print("  No changes made.")
        return

    try:
        choices = [int(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError:
        print("  ❌ Invalid input. No changes made.")
        return

    for choice in choices:
        if 1 <= choice <= len(options):
            item = options[choice - 1]
            if item in selected:
                selected.discard(item)
                print(f"  ➖ Removed: {item}")
            else:
                selected.add(item)
                print(f"  ➕ Added: {item}")
        else:
            print(f"  ⚠️  Invalid option: {choice}")

    new_list = [o for o in options if o in selected]  # Maintain order
    admin_svc.set_required_ppe(new_list)


def show_today_summary(attendance_svc: AttendanceService) -> None:
    """Print today's attendance summary."""
    records = attendance_svc.get_all_today()
    if not records:
        print("\n  📭 No attendance records today.")
        return

    print(f"\n  📊 Today's Attendance ({len(records)} records):")
    print("  " + "-"*60)
    print(f"  {'Employee ID':<12} {'Name':<20} {'PPE OK':<8} {'Time':<10}")
    print("  " + "-"*60)

    for r in records:
        ts = r.get("timestamp", "")
        time_str = ts.strftime("%H:%M:%S") if hasattr(ts, "strftime") else str(ts)[:19]
        ppe_ok = "✅" if r.get("ppe_verified") else "❌"
        print(f"  {r.get('employee_id',''):<12} {r.get('name',''):<20} {ppe_ok:<8} {time_str:<10}")


def list_employees(db) -> None:
    """List all active employees."""
    employees = list(db.employees.find({'is_active': True}, {'_id': 0, 'face_encoding': 0}))
    if not employees:
        print("\n  📭 No employees registered.")
        return

    print(f"\n  👥 Registered Employees ({len(employees)}):")
    print("  " + "-"*50)
    for emp in employees:
        print(f"  🆔 {emp.get('employee_id',''):<12} | {emp.get('name',''):<20} | {emp.get('department','')}")


def main():
    print(BANNER)
    cfg = get_config('development')
    ensure_directories(cfg)

    # Connect to DB
    db = db_connection.connect(cfg.MONGODB_URI, cfg.DB_NAME)

    admin_svc = AdminService(db)
    attendance_svc = AttendanceService(db, cfg.ATTENDANCE_COOLDOWN_MINUTES)
    report_gen = ReportGenerator(cfg.REPORTS_FOLDER)

    while True:
        print_menu()
        choice = input("  Enter choice: ").strip()

        if choice == "1":
            admin_svc.display_current_config()

        elif choice == "2":
            configure_ppe(admin_svc)

        elif choice == "3":
            show_today_summary(attendance_svc)

        elif choice == "4":
            records = attendance_svc.get_all_today()
            if records:
                path = report_gen.generate_daily_csv(records)
                print(f"\n  ✅ Exported to: {path}")
            else:
                print("\n  📭 No records to export.")

        elif choice == "5":
            list_employees(db)

        elif choice == "6":
            print("\n  👋 Exiting admin panel. Goodbye!")
            break

        else:
            print("  ❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
