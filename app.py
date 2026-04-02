import cv2
import streamlit as st
import time

st.set_page_config(page_title="Video Stream Debugger", layout="wide")

st.markdown("## 🔍 Raw Stream Debugger")
st.markdown("This utterly bypasses all databases, threading, and AI logic to isolate the literal networking playback speed.")

# Initialize state
if "monitoring" not in st.session_state:
    st.session_state.monitoring = False
if "url" not in st.session_state:
    st.session_state.url = ""

url_input = st.text_input("Stream URL:", value=st.session_state.url)

col1, col2 = st.columns(2)
with col1:
    if st.button("▶ Start Monitor") and url_input:
        st.session_state.monitoring = True
        st.session_state.url = url_input
        st.session_state.frames = 0
        st.session_state.start_time = time.time()
        st.rerun()
with col2:
    if st.button("⏹ Stop"):
        st.session_state.monitoring = False
        if "cap" in st.session_state:
            st.session_state.cap.release()
            del st.session_state["cap"]
        st.rerun()

if st.session_state.monitoring and st.session_state.url:
    clean_url = st.session_state.url.strip()
    
    if "cap" not in st.session_state:
        st.info(f"Connecting directly to: `{clean_url}`")
        cap = cv2.VideoCapture(clean_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        st.session_state.cap = cap
    else:
        cap = st.session_state.cap

    if not cap.isOpened():
        st.error("❌ Failed to open video stream. Ensure your laptop Ngrok is running and URL is correct.")
        st.session_state.monitoring = False
    else:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ `cap.read()` failed. The stream dropped or ended!")
            st.session_state.monitoring = False
        else:
            # We must convert BGR -> RGB securely because this is pure OpenCV now
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(rgb_frame, use_container_width=True)
            
            st.session_state.frames += 1
            elapsed = time.time() - st.session_state.start_time
            if elapsed > 0:
                fps = st.session_state.frames / elapsed
                st.markdown(f"### ⚡ Raw Playback FPS: `{fps:.2f}`")

            # Force Streamlit to render UI and pull the next frame
            time.sleep(0.01)
            st.rerun()
