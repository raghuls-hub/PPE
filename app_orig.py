"""
CCTV AI Admin Dashboard — Streamlit App Entry Point
====================================================
Run with:
    streamlit run app.py
"""

import os
import sys

# ── Ensure the cctv_ai_system dir is on path ──────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CCTV AI Admin Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom dark CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #10121c;
    color: #dde1f0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #14172b 0%, #1a1e35 100%);
    border-right: 1px solid #252a42;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.95rem;
    padding: 6px 0;
}

/* ── Main area ── */
.block-container {
    padding: 1.5rem 2rem;
    max-width: 100%;
    background: #10121c;
}

/* ── Cards ── */
div[data-testid="stExpander"] {
    background: #1a1e30;
    border: 1px solid #252a42;
    border-radius: 12px;
}

/* ── Metric ── */
div[data-testid="stMetric"] {
    background: #1a1e30;
    border: 1px solid #2a2f4a;
    border-radius: 10px;
    padding: 12px 16px;
}
div[data-testid="stMetricValue"] {
    color: #5b8def;
    font-size: 1.8rem;
    font-weight: 700;
}

/* ── Buttons ── */
button[kind="primary"] {
    background: linear-gradient(135deg, #4a6cf7, #6a3de8) !important;
    border: none !important;
    border-radius: 8px !important;
    color: #fff !important;
    font-weight: 600 !important;
}
button[kind="primary"]:hover {
    background: linear-gradient(135deg, #5a7cff, #7a4df8) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(74,108,247,0.4);
}

/* ── Tables ── */
div[data-testid="stDataFrame"] {
    border: 1px solid #252a42;
    border-radius: 10px;
}

/* ── Inputs ── */
input, textarea, select {
    background: #1a1e30 !important;
    border: 1px solid #2a2f4a !important;
    color: #dde1f0 !important;
    border-radius: 8px !important;
}

/* ── Alert boxes ── */
div[data-testid="stAlert"] {
    border-radius: 10px;
    border-left-width: 4px;
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
    background: transparent;
    color: #8892b0;
    font-weight: 500;
    border-bottom: 2px solid transparent;
    border-radius: 0;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #5b8def;
    border-bottom: 2px solid #5b8def;
}

/* ── Code blocks ── */
code {
    background: #1e2340 !important;
    color: #a8d8ea !important;
    border-radius: 4px !important;
    padding: 2px 5px !important;
}

/* ── Divider ── */
hr {
    border-color: #252a42;
}

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Camera tile ── */
.camera-tile {
    background: #1a1e30;
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 12px;
    border: 2px solid #2a2f4a;
    transition: border-color 0.3s;
}
.camera-tile.alert {
    border-color: #ff2222;
    box-shadow: 0 0 20px rgba(255,30,30,0.3);
    animation: pulse-red 1.5s infinite;
}
@keyframes pulse-red {
    0%   { box-shadow: 0 0 10px rgba(255,30,30,0.3); }
    50%  { box-shadow: 0 0 25px rgba(255,30,30,0.7); }
    100% { box-shadow: 0 0 10px rgba(255,30,30,0.3); }
}
</style>
""", unsafe_allow_html=True)


# ── Session State Init ─────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "monitor_running":       False,
        "att_running":           False,
        "attendance_thread":     None,
        "registering_worker":    None,
        "registering_name":      None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── Sidebar Navigation ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
        <div style="text-align:center;padding:16px 0 24px">
            <div style="font-size:2.5rem;">🛡️</div>
            <div style="font-size:1.15rem;font-weight:700;color:#5b8def;">CCTV AI Admin</div>
            <div style="font-size:0.75rem;color:#5a6285;margin-top:2px;">Smart Safety Dashboard</div>
        </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        options=["📷 Camera Management", "✅ Attendance", "🖥️ Live Monitor"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Quick DB health check
    try:
        import db
        cameras = db.get_cameras()
        workers = db.get_workers()
        col1, col2 = st.columns(2)
        col1.metric("Cameras", len(cameras))
        col2.metric("Workers", len(workers))
    except Exception as e:
        st.error(f"DB Error: {e}")

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.7rem;color:#3a4060;text-align:center;">'
        'CCTV AI Admin v1.0<br>© 2026 DT-PPE</div>',
        unsafe_allow_html=True
    )


# ── Page Routing ───────────────────────────────────────────────────────────────

if page == "📷 Camera Management":
    from pages.camera_page import render
    render()

elif page == "✅ Attendance":
    from pages.attendance_page import render
    render()

elif page == "🖥️ Live Monitor":
    from pages.live_monitor_page import render
    render()
