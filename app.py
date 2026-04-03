import cv2
import streamlit as st
import time
import base64

st.set_page_config(page_title="Video Stream Debugger", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
        background-color: #0a0a0f;
        color: #e8e8f0;
    }
    .stApp {
        background: #0a0a0f;
    }
    h1, h2, h3 {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
    }
    .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }

    /* Header */
    .stream-header {
        border-left: 4px solid #7c3aed;
        padding: 0.4rem 1rem;
        margin-bottom: 1.5rem;
    }
    .stream-header h2 {
        margin: 0;
        font-size: 1.8rem;
        color: #f0f0ff;
        letter-spacing: -0.5px;
    }
    .stream-header p {
        margin: 0.2rem 0 0;
        font-size: 0.85rem;
        color: #6b6b8a;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Input */
    .stTextInput > div > div > input {
        background: #13131f !important;
        border: 1px solid #2a2a40 !important;
        border-radius: 6px !important;
        color: #c8c8e8 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.88rem !important;
        padding: 0.6rem 1rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #7c3aed !important;
        box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.25) !important;
    }
    .stTextInput label {
        color: #8888aa !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Buttons */
    .stButton > button {
        background: #13131f !important;
        border: 1px solid #2a2a40 !important;
        border-radius: 6px !important;
        color: #c8c8e8 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
        padding: 0.5rem 1.2rem !important;
        width: 100%;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        border-color: #7c3aed !important;
        color: #ffffff !important;
        background: #1a1030 !important;
    }

    /* FPS badge */
    .fps-badge {
        display: inline-block;
        background: #13131f;
        border: 1px solid #2a2a40;
        border-radius: 6px;
        padding: 0.5rem 1.2rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        color: #a78bfa;
        margin-top: 0.8rem;
    }
    .fps-badge span {
        color: #e8e8f0;
        font-weight: 700;
        font-size: 1.1rem;
    }

    /* Status messages */
    .stAlert {
        border-radius: 6px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.82rem !important;
    }

    /* Image container */
    .stImage {
        border-radius: 8px;
        overflow: hidden;
    }
    [data-testid="stImage"] img {
        border-radius: 8px;
        border: 1px solid #1e1e2e;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="stream-header">
    <h2>Stream Debugger</h2>
    <p>raw network · no ai · no threading · no database</p>
</div>
""", unsafe_allow_html=True)

# Initialize state
if "monitoring" not in st.session_state:
    st.session_state.monitoring = False
if "url" not in st.session_state:
    st.session_state.url = ""
if "frames" not in st.session_state:
    st.session_state.frames = 0
if "start_time" not in st.session_state:
    st.session_state.start_time = None

url_input = st.text_input("STREAM URL", value=st.session_state.url, placeholder="rtsp://... or http://...")

col1, col2 = st.columns([1, 1])
with col1:
    start_clicked = st.button("▶  Start Monitor")
with col2:
    stop_clicked = st.button("⏹  Stop")

if start_clicked and url_input:
    st.session_state.monitoring = True
    st.session_state.url = url_input
    st.session_state.frames = 0
    st.session_state.start_time = time.time()
    # Release any old capture
    if "cap" in st.session_state:
        st.session_state.cap.release()
        del st.session_state["cap"]
    st.rerun()

if stop_clicked:
    st.session_state.monitoring = False
    if "cap" in st.session_state:
        st.session_state.cap.release()
        del st.session_state["cap"]
    st.rerun()

if st.session_state.monitoring and st.session_state.url:
    clean_url = st.session_state.url.strip()

    if "cap" not in st.session_state:
        st.info(f"Connecting to: `{clean_url}`")
        cap = cv2.VideoCapture(clean_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        st.session_state.cap = cap
    else:
        cap = st.session_state.cap

    if not cap.isOpened():
        st.error("❌ Failed to open stream. Check your URL or Ngrok tunnel.")
        st.session_state.monitoring = False
    else:
        ret, frame = cap.read()
        if not ret:
            # Try reopening once before giving up
            cap.release()
            cap = cv2.VideoCapture(clean_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            st.session_state.cap = cap
            ret, frame = cap.read()

        if not ret:
            st.warning("⚠️ Stream dropped or ended — `cap.read()` returned False.")
            st.session_state.monitoring = False
        else:
            # Bypass st.image() entirely — encode as base64 data URI and inject via markdown.
            # st.image() registers frames in Streamlit's MemoryMediaFileStorage and loses them
            # on fast reruns before the browser fetches them, causing MediaFileStorageError.
            _, buffer = cv2.imencode('.jpg', frame)
            b64 = base64.b64encode(buffer).decode('utf-8')
            st.markdown(
                f'<img src="data:image/jpeg;base64,{b64}" '
                f'style="width:100%;border-radius:8px;border:1px solid #1e1e2e;" />',
                unsafe_allow_html=True
            )

            st.session_state.frames += 1
            elapsed = time.time() - st.session_state.start_time
            if elapsed > 0:
                fps = st.session_state.frames / elapsed
                st.markdown(
                    f'<div class="fps-badge">⚡ Raw FPS &nbsp;→&nbsp; <span>{fps:.2f}</span></div>',
                    unsafe_allow_html=True
                )

            time.sleep(0.01)
            st.rerun()