"""
Camera Management Page
"""

import streamlit as st
import db


def render():
    st.markdown("## 📷 Camera Management")
    st.markdown("Add and manage all cameras used by the system.")

    # ── Add Camera Form ────────────────────────────────────────────────────────
    with st.expander("➕ Add New Camera", expanded=True):
        st.info("**Tip for IP Webcam & Testing:** Use the MJPEG endpoint for live feeds (e.g., `/stream.mjpeg`). For testing, you can also paste a **Google Drive shareable link** to a video file, and the system will automatically stream it!")
        
        if st.button("💻 Use Integrated Webcam (Index 0)"):
            st.session_state["pending_stream_url"] = "0"
            st.rerun()

        with st.form("add_camera_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                name     = st.text_input("Camera Name", placeholder="e.g. Gate Camera 1")
                location = st.text_input("Location", placeholder="e.g. Main Entrance")
            with col2:
                stream_url = st.text_input("Stream URL / Camera Index",
                                           value=st.session_state.get("pending_stream_url", ""),
                                           placeholder="rtsp://... or 0 for USB")
                cam_type   = st.selectbox("Camera Type",
                                          ["monitor", "attendance"],
                                          format_func=lambda t: "🔍 Live Monitor" if t == "monitor" else "✅ Attendance")

            submitted = st.form_submit_button("Add Camera", width="stretch")
            if submitted:
                if not name or not stream_url:
                    st.error("Name and Stream URL are required.")
                else:
                    cid = db.add_camera(name, stream_url, location, cam_type)
                    if "pending_stream_url" in st.session_state:
                        del st.session_state["pending_stream_url"]
                    st.success(f"Camera **{name}** added! (ID: {cid})")
                    st.rerun()

    # ── Camera List ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 All Cameras")

    cameras = db.get_cameras()
    if not cameras:
        st.info("No cameras added yet. Use the form above to add one.")
        return

    # Headers
    hcol = st.columns([3, 3, 2, 2, 1])
    for col, label in zip(hcol, ["Name", "Stream URL", "Location", "Type", "Action"]):
        col.markdown(f"**{label}**")
    st.markdown('<hr style="margin:4px 0 8px">', unsafe_allow_html=True)

    for cam in cameras:
        col1, col2, col3, col4, col5 = st.columns([3, 3, 2, 2, 1])
        col1.write(cam["name"])
        col2.code(cam["stream_url"], language=None)
        col3.write(cam.get("location", "—"))

        badge = "🔍 Monitor" if cam.get("type") == "monitor" else "✅ Attendance"
        col4.write(badge)

        if col5.button("🗑", key=f"del_cam_{cam['_id']}", help="Delete camera"):
            db.delete_camera(cam["_id"])
            st.success(f"Camera **{cam['name']}** deleted.")
            st.rerun()
