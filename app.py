import cv2
import streamlit as st
import time

st.set_page_config(page_title="Video Stream Debugger", layout="wide")

st.markdown("## 🔍 Raw Stream Debugger")
st.markdown("This utterly bypasses all databases, threading, and AI logic to isolate the literal networking playback speed.")

# Initialize state
if "monitoring" not in st.session_state:
    st.session_state.monitoring = False

url = st.text_input("Stream URL (e.g., https://.../stream/fall.mp4):")

col1, col2 = st.columns(2)
with col1:
    if st.button("▶ Start Monitor", use_container_width=True) and url:
        st.session_state.monitoring = True
        st.rerun()
with col2:
    if st.button("⏹ Stop", use_container_width=True):
        st.session_state.monitoring = False
        st.rerun()

if st.session_state.monitoring and url:
    clean_url = url.strip()
    st.info(f"Connecting directly to: `{clean_url}`")
    
    cap = cv2.VideoCapture(clean_url)
    # Aggressively reduce frame buffer to prioritize real-time over stutter
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        st.error("❌ Failed to open video stream. Ensure your laptop Ngrok is running and URL is correct.")
    else:
        st.success("✅ OpenCV Connected! Rendering raw frames...")
        
        frame_placeholder = st.empty()
        stats_placeholder = st.empty()
        
        frames_grabbed = 0
        start_time = time.time()
        
        while st.session_state.monitoring:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ `cap.read()` failed. The stream dropped or ended!")
                break
            
            # Use width="stretch" to prevent Streamlit 1.40+ deprecation warnings
            # We must convert BGR -> RGB securely because this is pure OpenCV now
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, width="stretch")
            
            frames_grabbed += 1
            elapsed = time.time() - start_time
            
            if elapsed > 0.5:  # Update FPS display every half-second
                fps = frames_grabbed / elapsed
                stats_placeholder.markdown(f"### ⚡ Raw Playback FPS: `{fps:.2f}`")
                
            # A tiny sleep keeps the Streamlit WebSocket from totally locking up the Stop button
            time.sleep(0.01)
            
        cap.release()
