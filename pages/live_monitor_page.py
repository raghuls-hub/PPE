"""
Live Monitor Page
Displays all 'monitor' cameras in a grid using real MJPEG iframes.
No Streamlit reruns needed for video — the stream updates continuously.
"""

from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Dict, List

import streamlit as st
import streamlit.components.v1 as components

import db
from engine.monitor_engine import MonitorEngine
from utils.stream_server import get_stream_server

STREAM_PORT = 8765


def _get_engine() -> MonitorEngine:
    if "monitor_engine" not in st.session_state:
        st.session_state.monitor_engine = MonitorEngine()
    return st.session_state.monitor_engine


def _ensure_server():
    """Start MJPEG server once and keep reference in session state."""
    if "stream_server" not in st.session_state or \
            not st.session_state.stream_server.is_running():
        st.session_state.stream_server = get_stream_server(port=STREAM_PORT)
    return st.session_state.stream_server


def render():
    st.markdown("## 🖥️ Live Monitor")

    engine = _get_engine()
    server = _ensure_server()

    cameras = db.get_cameras(cam_type="monitor")
    if not cameras:
        st.warning("No **monitor** cameras configured. "
                   "Add cameras in 📷 Camera Management and set type to **monitor**.")
        return

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl_col1, ctrl_col2, *_ = st.columns([2, 2, 6])

    with ctrl_col1:
        if st.button("▶ Start All", use_container_width=True, type="primary"):
            for cam in cameras:
                cid = cam["_id"]
                engine.start_camera(cam)
                def make_provider(c=cid):
                    return lambda: engine.get_state(c).get("frame") if engine.get_state(c) else None
                server.register_stream(cid, make_provider())
            st.session_state["monitor_running"] = True
            st.rerun()

    with ctrl_col2:
        if st.button("⏹ Stop All", use_container_width=True):
            engine.stop_all()
            for cam in cameras:
                server.unregister_stream(cam["_id"])
            st.session_state["monitor_running"] = False
            st.rerun()

    # Per-camera toggles
    with st.expander("🎛️ Individual Camera Controls", expanded=False):
        tog_cols = st.columns(min(4, max(1, len(cameras))))
        for i, cam in enumerate(cameras):
            with tog_cols[i % len(tog_cols)]:
                cid   = cam["_id"]
                is_on = engine.is_running(cid)
                label = f"{'🟢' if is_on else '🔴'} {cam['name']}"
                if st.button(label, key=f"toggle_{cid}", use_container_width=True):
                    if is_on:
                        engine.stop_camera(cid)
                        server.unregister_stream(cid)
                    else:
                        engine.start_camera(cam)
                        def make_provider(c=cid):
                            return lambda: engine.get_state(c).get("frame") if engine.get_state(c) else None
                        server.register_stream(cid, make_provider())
                    st.rerun()

    st.markdown("---")

    # ── Active alert banner ────────────────────────────────────────────────────
    alerts = db.get_recent_alerts(limit=5)
    unacked = [a for a in alerts if not a.get("acknowledged")]
    for a in unacked:
        ts   = a["timestamp"].strftime("%H:%M:%S") if isinstance(a["timestamp"], datetime) else ""
        icon = {"fire": "🔥", "fall": "🚨", "ppe": "⚠️"}.get(a["alert_type"], "⚠️")
        st.error(f"{icon} **{a['alert_type'].upper()}** — {a['message']} at {ts}")

    # ── Camera grid (MJPEG iframes) ────────────────────────────────────────────
    cols_count = min(2, len(cameras))
    rows       = math.ceil(len(cameras) / cols_count)
    cam_idx    = 0

    for _ in range(rows):
        cols = st.columns(cols_count)
        for col in cols:
            if cam_idx >= len(cameras):
                break
            cam    = cameras[cam_idx]
            cid    = cam["_id"]
            is_on  = engine.is_running(cid)
            state  = engine.get_state(cid)

            has_alert = state and (
                state.get("ppe_alert") or
                state.get("fire_alert") or
                state.get("fall_alert")
            )
            border  = "#ff2222" if has_alert else "#2c2f3a"
            alert_labels = []
            if state:
                if state.get("fire_alert"): alert_labels.append("🔥 FIRE")
                if state.get("fall_alert"): alert_labels.append("🚨 FALL")
                if state.get("ppe_alert"):  alert_labels.append("⚠️ PPE VIOLATION")

            fps = state.get("fps", 0.0) if state else 0.0

            with col:
                # ── Header tile ───────────────────────────────────────────────
                st.markdown(
                    f'<div style="border:2px solid {border};border-radius:10px 10px 0 0;'
                    f'background:#1a1d2e;padding:6px 12px;display:flex;'
                    f'justify-content:space-between;align-items:center;">'
                    f'<span style="color:#ddd;font-weight:600">{cam["name"]}</span>'
                    f'<span style="color:#888;font-size:0.78em">'
                    f'{"🟢 " + f"{fps:.1f} fps" if is_on else "🔴 stopped"}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # ── Alert badges ──────────────────────────────────────────────
                if alert_labels:
                    st.markdown(
                        f'<div style="background:#500;padding:4px 10px;'
                        f'font-weight:bold;color:#fff;font-size:0.85em;">'
                        f'{" &nbsp;|&nbsp; ".join(alert_labels)}</div>',
                        unsafe_allow_html=True,
                    )

                # ── MJPEG iframe (Replaced with native Streamlit image) ──
                if is_on:
                    if "active_placeholders" not in locals():
                        active_placeholders = []
                    frame_ph = st.empty()
                    active_placeholders.append((cid, frame_ph))
                else:
                    # Stopped placeholder
                    components.html(
                        f"""
                        <html>
                        <body style="margin:0;padding:0;background:#111;overflow:hidden;">
                          <div style="height:320px;display:flex;align-items:center;
                               justify-content:center;color:#444;
                               font-family:sans-serif;font-size:1em;flex-direction:column;gap:8px;">
                            <span style="font-size:2rem;">📷</span>
                            <span>Camera not started</span>
                            <span style="font-size:0.8em;color:#333">{cam.get('location','')}</span>
                          </div>
                        </body>
                        </html>
                        """,
                        height=325,
                        scrolling=False,
                    )

                # Bottom border cap
                st.markdown(
                    f'<div style="border:2px solid {border};border-top:none;'
                    f'border-radius:0 0 10px 10px;background:#1a1d2e;'
                    f'padding:4px 12px;color:#555;font-size:0.75em;">'
                    f'{cam.get("location","")}&nbsp;</div>',
                    unsafe_allow_html=True,
                )

            cam_idx += 1

    # ── Alert log ─────────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📋 Recent Alerts", expanded=False):
        full_alerts = db.get_recent_alerts(limit=30)
        if not full_alerts:
            st.info("No alerts recorded yet.")
        else:
            for a in full_alerts:
                ts   = a["timestamp"].strftime("%Y-%m-%d %H:%M:%S") \
                       if isinstance(a["timestamp"], datetime) else str(a["timestamp"])
                icon = {"fire": "🔥", "fall": "🚨", "ppe": "⚠️"}.get(a["alert_type"], "⚠️")
                ack  = "✅" if a.get("acknowledged") else "🔴"
                c1, c2 = st.columns([9, 1])
                c1.write(f"{ack} {icon} **{a['alert_type'].upper()}** — {a['message']} — *{ts}*")
                if not a.get("acknowledged"):
                    if c2.button("Ack", key=f"ack_{a['_id']}"):
                        db.acknowledge_alert(a["_id"])
                        st.rerun()

    # ── Live Rendering Loop ───────────────────────────────────────────────────
    if locals().get("active_placeholders"):
        last_frames = {}
        while True:
            for cid, ph in active_placeholders:
                state = engine.get_state(cid)
                if state:
                    frame = state.get("frame")
                    if frame is not None and frame != last_frames.get(cid):
                        last_frames[cid] = frame
                        ph.image(frame, width="stretch")
            time.sleep(0.03)
