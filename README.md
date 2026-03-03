# 🦺 Smart PPE Detection & Face Recognition Attendance System

An AI-powered desktop application integrating **YOLO v8 PPE detection**, **OpenCV LBPH face recognition**, and **MongoDB attendance logging** — with live camera and video file support.

---

## 📁 Project Structure

```
DT - PPE/
├── best.pt                    ← Trained YOLO model (PPE classes)
├── main.py                    ← Entry point
├── live_recognition.py        ← Core application class
├── admin_panel.py             ← Admin configuration CLI
├── config.py                  ← Central configuration
├── requirements.txt
├── .env.example               ← Environment variable template
│
├── services/
│   ├── face_service.py        ← LBPH face recognition
│   ├── ppe_service.py         ← YOLO PPE detection & validation
│   ├── attendance_service.py  ← MongoDB attendance with cooldown
│   └── admin_service.py       ← PPE config management
│
├── utils/
│   ├── db_connection.py       ← MongoDB singleton
│   ├── report_generator.py    ← CSV daily report
│   └── sound_notification.py  ← Audio notification
│
├── faces/                     ← Saved employee face images
├── snapshots/                 ← Auto-saved attendance snapshots
└── reports/                   ← Daily CSV attendance reports
```

---

## ⚙️ Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
copy .env.example .env
# Edit .env as needed (MongoDB URI, thresholds, etc.)
```

### 3. Ensure MongoDB is running
```bash
# Default: mongodb://localhost:27017
# DB name: ppe_attendance
```

---

## 🚀 Usage

### Live Camera Mode (default)
```bash
python main.py
python main.py --mode live
```

### Video File Mode
```bash
python main.py --mode video --source path/to/video.mp4
```

### Admin Panel
```bash
python main.py --admin
# or
python admin_panel.py
```

---

## ⌨️ Keyboard Controls (during recognition)

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reload employees from database |
| `n` | Register new employee (live mode) |
| `s` | Take screenshot |
| `e` | Export today's attendance to CSV |

---

## 🦺 PPE Detection Classes

The `best.pt` YOLO model detects:
```
Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest,
Person, Safety Cone, Safety Vest, machinery, vehicle
```

Configurable required items: **Hardhat**, **Mask**, **Safety Vest**

### Validation Logic
For each required PPE item:
- ✅ Positive class (e.g. `Hardhat`) **must** be detected
- ❌ Negative class (e.g. `NO-Hardhat`) **must NOT** be detected

---

## 🎨 OpenCV Overlay Color Codes

| Color | Meaning |
|-------|---------|
| 🟢 Green | Recognized employee + PPE compliant + Attendance MARKED |
| 🔴 Red | Recognized employee + PPE failed |
| 🟡 Yellow | Unknown person |
| ⚫ Gray | Face detected but encoding failed |

---

## 🗄️ MongoDB Collections

### `employees`
```json
{
  "employee_id": "EMP-001",
  "name": "John Doe",
  "department": "Engineering",
  "face_encoding": [...],
  "is_active": true
}
```

### `attendance`
```json
{
  "employee_id": "EMP-001",
  "name": "John Doe",
  "timestamp": "2026-03-03T07:00:00Z",
  "ppe_verified": true,
  "detected_ppe": ["Hardhat", "Mask"],
  "missing_ppe": [],
  "camera_source": "Live",
  "confidence_score": 0.91
}
```

### `ppe_config`
```json
{
  "required_ppe": ["Hardhat", "Safety Vest"],
  "updated_at": "2026-03-03T07:00:00Z"
}
```

---

## 🔧 Configuration (`config.py` / `.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `DB_NAME` | `ppe_attendance` | Database name |
| `FACE_MATCH_THRESHOLD` | `70` | LBPH max distance (lower = stricter) |
| `PPE_CONFIDENCE_THRESHOLD` | `0.45` | YOLO confidence cutoff |
| `ATTENDANCE_COOLDOWN_MINUTES` | `5` | Prevent duplicate attendance |
| `CAMERA_INDEX` | `0` | Webcam device index |
| `SAVE_SNAPSHOTS` | `true` | Save photo when attendance marked |
| `SOUND_ENABLED` | `true` | Play sound on attendance mark |
| `PROCESS_EVERY_N_FRAMES` | `3` | YOLO runs every N frames (performance) |

---

## 📝 First-Time Setup: Register an Employee

1. Start the system: `python main.py --mode live`
2. Press `n` to open registration mode
3. Enter Employee ID, Name, Department
4. Position face in frame → press `SPACE` to capture
5. Employee is saved to MongoDB and immediately recognized

---

## 📊 Daily Report

Press `e` during recognition or use the admin panel (option 4) to export a CSV:
```
reports/attendance_2026-03-03.csv
```
