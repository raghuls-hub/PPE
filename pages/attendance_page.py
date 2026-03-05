"""
Attendance Management Page

Three sub-tabs:
  1. Workers — CRUD for workers + face registration via camera
  2. PPE Config — configure which PPE items are required
  3. Live Attendance — live camera feed + today's attendance log
"""

from __future__ import annotations

import os
import base64
import time
from datetime import datetime, date
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components
import numpy as np

import db
import config as cfg

STREAM_PORT = 8765


def _get_attendance_thread():
    return st.session_state.get("attendance_thread")


def _b64(jpg: bytes | None) -> str:
    if jpg is None:
        return ""
    return base64.b64encode(jpg).decode()


# ── Sub-Tab 1: Workers ────────────────────────────────────────────────────────

def _render_workers():
    st.markdown("### 👷 Worker Management")

    # Add worker form
    with st.expander("➕ Add New Worker", expanded=False):
        with st.form("add_worker_form", clear_on_submit=True):
            col1, col2, col3 = st.columns(3)
            emp_id = col1.text_input("Employee ID", placeholder="EMP-001")
            name   = col2.text_input("Full Name",   placeholder="John Doe")
            dept   = col3.text_input("Department",  placeholder="Operations")
            submitted = st.form_submit_button("Add Worker", use_container_width=True)
            if submitted:
                if not emp_id or not name:
                    st.error("Employee ID and Name are required.")
                else:
                    wid = db.add_worker(emp_id, name, dept or "General")
                    if wid:
                        st.success(f"Worker **{name}** added.")
                        st.rerun()
                    else:
                        st.error(f"Employee ID **{emp_id}** already exists.")

    # Worker list
    workers = db.get_workers()
    if not workers:
        st.info("No workers registered yet.")
        return

    st.markdown("---")
    hcols = st.columns([2, 3, 2, 2, 2, 2])
    for c, h in zip(hcols, ["Emp ID", "Name", "Department", "Face", "Register", "Delete"]):
        c.markdown(f"**{h}**")

    for w in workers:
        wid = w["_id"]
        c1, c2, c3, c4, c5, c6 = st.columns([2, 3, 2, 2, 2, 2])
        c1.write(w["emp_id"])
        c2.write(w["name"])
        c3.write(w.get("department", "—"))

        has_face = bool(w.get("face_encoding"))
        c4.markdown("✅ Enrolled" if has_face else "❌ Not enrolled")

        # Face registration (opens a dialog-style expander)
        if c5.button("📷 Register", key=f"reg_{wid}"):
            st.session_state["registering_worker"] = wid
            st.session_state["registering_name"]   = w["name"]

        if c6.button("🗑 Delete", key=f"del_{wid}"):
            db.delete_worker(wid)
            st.success(f"Worker {w['name']} deactivated.")
            st.rerun()

    # ── Face Registration Flow ─────────────────────────────────────────────────
    reg_wid = st.session_state.get("registering_worker")
    if reg_wid:
        reg_name = st.session_state.get("registering_name", "Worker")
        st.markdown("---")
        st.markdown(f"#### 📸 Face Registration — {reg_name}")

        att_cameras = db.get_cameras(cam_type="attendance")
        if not att_cameras:
            st.warning("No attendance camera configured. Add one in 📷 Camera Management.")
        else:
            cam_options = {c["name"]: c for c in att_cameras}
            sel_cam = st.selectbox("Select Camera for Registration", list(cam_options.keys()),
                                   key="reg_cam_sel")
            cam = cam_options[sel_cam]

            col_a, col_b = st.columns(2)
            frame_ph = col_a.empty()
            info_ph  = col_b.empty()

            if st.button("🟢 Start Capturing Faces", key="start_capture"):
                from engine.attendance_engine import capture_face_samples, NUM_SAMPLES

                progress_bar = st.progress(0)
                status_txt   = st.empty()

                captured_frames = []

                def _cb(i, total, jpg_bytes):
                    progress_bar.progress(i / total)
                    status_txt.write(f"Captured {i}/{total} samples")
                    frame_ph.markdown(
                        f'<img src="data:image/jpeg;base64,{_b64(jpg_bytes)}" '
                        f'style="width:100%;border-radius:8px;"/>',
                        unsafe_allow_html=True
                    )
                    captured_frames.append(jpg_bytes)

                with st.spinner(f"Opening camera and capturing {NUM_SAMPLES} face samples…"):
                    encodings = capture_face_samples(cam["stream_url"], progress_cb=_cb)

                if encodings:
                    # Save face images and encoding to DB
                    face_img_path = os.path.join(cfg.FACES_DIR,
                        f"{reg_wid}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.jpg")
                    if captured_frames:
                        import cv2, numpy as np
                        frame_np = cv2.imdecode(np.frombuffer(captured_frames[0], np.uint8), cv2.IMREAD_COLOR)
                        cv2.imwrite(face_img_path, frame_np)

                    ok = db.save_worker_face(
                        worker_id=reg_wid,
                        face_encoding=encodings[0],
                        all_encodings=encodings,
                        image_path=face_img_path
                    )
                    if ok:
                        st.success(f"✅ Face registered for **{reg_name}**!")
                        # Reload faces in running attendance thread if any
                        t = _get_attendance_thread()
                        if t:
                            t.reload_faces()
                        del st.session_state["registering_worker"]
                        del st.session_state["registering_name"]
                        st.rerun()
                    else:
                        st.error("Failed to save face encoding to database.")
                else:
                    st.error("Face capture failed or timed out. Try again with better lighting.")

            if st.button("❌ Cancel Registration", key="cancel_reg"):
                del st.session_state["registering_worker"]
                st.rerun()


# ── Sub-Tab 2: PPE Config ─────────────────────────────────────────────────────

def _render_ppe_config():
    st.markdown("### 🦺 PPE Requirements")
    st.markdown(
        "Toggle which PPE items are **required** for attendance to be marked.\n"
        "Workers missing any required item will **not** have attendance recorded."
    )

    current = db.get_required_ppe()

    st.markdown("---")
    selected = []
    cols = st.columns(len(cfg.AVAILABLE_PPE_OPTIONS))
    for col, item in zip(cols, cfg.AVAILABLE_PPE_OPTIONS):
        icons = {"Hardhat": "⛑️", "Mask": "😷", "Safety Vest": "🦺"}
        checked = col.checkbox(f"{icons.get(item,'')} {item}", value=(item in current), key=f"ppe_{item}")
        if checked:
            selected.append(item)

    st.markdown("---")
    col1, col2 = st.columns([2, 6])
    if col1.button("💾 Save PPE Config", type="primary", use_container_width=True):
        ok = db.set_required_ppe(selected)
        if ok:
            st.success(f"PPE requirements saved: **{', '.join(selected) if selected else 'None'}**")
        else:
            st.error("Failed to save PPE config.")

    st.markdown("")
    st.info(f"**Currently required:** {', '.join(current) if current else 'None (all workers pass automatically)'}")


# ── Sub-Tab 3: Live Attendance ────────────────────────────────────────────────

def _render_live_attendance():
    st.markdown("### 📹 Live Attendance Feed")

    att_cameras = db.get_cameras(cam_type="attendance")
    if not att_cameras:
        st.warning("No attendance cameras configured. Add one in 📷 Camera Management and set type to 'attendance'.")
        return

    cam_opts = {c["name"]: c for c in att_cameras}
    sel_name = st.selectbox("Select Attendance Camera", list(cam_opts.keys()), key="att_cam_sel")
    cam = cam_opts[sel_name]

    col_start, col_stop, _ = st.columns([2, 2, 6])

    if col_start.button("▶ Start Feed", type="primary", use_container_width=True):
        # Stop any existing thread
        existing = _get_attendance_thread()
        if existing:
            existing.stop()
        from engine.attendance_engine import AttendanceThread
        from utils.stream_server import get_stream_server
        
        t = AttendanceThread(cam)
        t.start()
        st.session_state["attendance_thread"] = t
        st.session_state["att_running"] = True
        
        server = get_stream_server(port=STREAM_PORT)
        stream_key = f"attendance_{cam['_id']}"
        def make_provider(th=t):
            return lambda: th.get_state().get("frame") if th.get_state() else None
        server.register_stream(stream_key, make_provider())

    if col_stop.button("⏹ Stop Feed", use_container_width=True):
        t = _get_attendance_thread()
        if t:
            t.stop()
        from utils.stream_server import get_stream_server
        server = get_stream_server(port=STREAM_PORT)
        server.unregister_stream(f"attendance_{cam['_id']}")
        st.session_state["att_running"] = False

    running = st.session_state.get("att_running", False)
    t = _get_attendance_thread()

    feed_col, log_col = st.columns([3, 2])

    with feed_col:
        if running and t and t.is_alive():
            stream_key = f"attendance_{cam['_id']}"
            stream_url = f"http://localhost:{STREAM_PORT}/view/{stream_key}?t={int(time.time())}"

            st.markdown(
                f'<iframe src="{stream_url}" width="100%" height="380" '
                f'style="border:none; border-radius:10px;" allowfullscreen></iframe>',
                unsafe_allow_html=True,
            )

            # Last-marked badge — refreshes on page interaction
            state = t.get_state()
            last  = state.get("last_marked")
            if last:
                st.success(
                    f"✅ **{last['name']}** ({last['emp_id']}) marked at {last['time']}"
                )
        else:
            components.html(
                """
                <html>
                <body style="margin:0;padding:0;background:#111;overflow:hidden;">
                  <div style="height:380px;display:flex;align-items:center;
                       justify-content:center;flex-direction:column;gap:10px;
                       color:#444;font-family:sans-serif;">
                    <span style="font-size:2.5rem;">📷</span>
                    <span>Feed stopped — press ▶ Start Feed</span>
                  </div>
                </body>
                </html>
                """,
                height=385,
                scrolling=False,
            )

    with log_col:
        st.markdown("#### Today's Attendance Log")
        records = db.get_today_attendance()
        if not records:
            st.info("No attendance marked today.")
        else:
            for r in records[:20]:
                ts = r["timestamp"].strftime("%H:%M:%S") if isinstance(r["timestamp"], datetime) else str(r["timestamp"])
                icon = "✅" if r.get("ppe_ok") else "❌"
                st.markdown(
                    f'<div style="background:#1e2030;border-radius:6px;padding:6px 10px;'
                    f'margin-bottom:4px;">'
                    f'<b>{icon} {r["name"]}</b> '
                    f'<span style="color:#888;font-size:0.85em">({r["emp_id"]})</span><br>'
                    f'<span style="color:#aaa;font-size:0.8em">🕐 {ts}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # Date filter
        st.markdown("---")
        st.markdown("#### Filter by Date")
        pick_date = st.date_input("Date", value=date.today(), key="att_date_pick")
        hist = db.get_attendance_by_date(datetime.combine(pick_date, datetime.min.time()))
        if hist:
            import pandas as pd
            rows = []
            for r in hist:
                ts = r["timestamp"].strftime("%H:%M:%S") if isinstance(r["timestamp"], datetime) else ""
                rows.append({
                    "Time": ts,
                    "Name": r.get("name", ""),
                    "Emp ID": r.get("emp_id", ""),
                    "PPE OK": "✅" if r.get("ppe_ok") else "❌",
                    "Detected PPE": ", ".join(r.get("detected_ppe", [])),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No records for selected date.")




# ── Main Render ────────────────────────────────────────────────────────────────

def render():
    st.markdown("## ✅ Attendance Management")

    tab_workers, tab_ppe, tab_live = st.tabs([
        "👷 Workers",
        "🦺 PPE Config",
        "📹 Live Attendance",
    ])

    with tab_workers:
        _render_workers()
    with tab_ppe:
        _render_ppe_config()
    with tab_live:
        _render_live_attendance()
