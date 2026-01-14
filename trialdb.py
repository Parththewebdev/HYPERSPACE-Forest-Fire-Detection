import streamlit as st
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import cv2
import folium
from streamlit_folium import st_folium
from satellite_risk import get_fire_risk
from video_locations import video_locations
import os

st.set_page_config(page_title="Forest Fire Detection", layout="wide")

# Load model once
@st.cache_resource
def load_model():
    return YOLO("fire.pt")

model = load_model()

VIDEO_DIR = "videos"
videos = sorted([v for v in os.listdir(VIDEO_DIR) if v.endswith(".mp4")])

if not videos:
    st.error("No videos found in /videos folder")
    st.stop()

if "vid_idx" not in st.session_state:
    st.session_state.vid_idx = 0

current_video_name = videos[st.session_state.vid_idx]
current_video_path = os.path.join(VIDEO_DIR, current_video_name)

lat, lon = video_locations.get(current_video_name, (30.1, 79.2))
risk = get_fire_risk(lat, lon)

# Title and risk display
st.markdown("<h1 style='text-align: center;'>FOREST FIRE DETECTION</h1>", unsafe_allow_html=True)
st.markdown(f"<h3 style='text-align: center;'>Satellite Fire Risk: {risk}</h3>", unsafe_allow_html=True)


# Top layout
col_map, col_video = st.columns(2)

with col_map:
    st.subheader("Fire Location Map")
    m = folium.Map(location=[lat, lon], zoom_start=6)
    folium.Marker([lat, lon], tooltip=current_video_name, icon=folium.Icon(color="red")).add_to(m)
    st_folium(m, width=500, height=350)

with col_video:
    st.subheader("Video Source")
    frame_placeholder = st.empty()

# Navigation buttons
st.write("")
btn_col1, btn_col2, btn_col3 = st.columns([3,1,1])

with btn_col2:
    if st.button("Prev"):
        st.session_state.vid_idx = (st.session_state.vid_idx - 1) % len(videos)
        st.rerun()

with btn_col3:
    if st.button("Next"):
        st.session_state.vid_idx = (st.session_state.vid_idx + 1) % len(videos)
        st.rerun()


# Logs below buttons
st.subheader("Detection Logs")
if "log" not in st.session_state:
    st.session_state.log = []

log_placeholder = st.empty()

# Frame index state
if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0

# Reset frame when switching video
if "last_video" not in st.session_state or st.session_state.last_video != current_video_name:
    st.session_state.frame_idx = 0
    st.session_state.log = []
    st.session_state.last_video = current_video_name

# Video frame processing (one frame per run)
cap = cv2.VideoCapture(current_video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_idx)

ret, frame = cap.read()
cap.release()

if ret:
    results = model(frame, conf=0.4)
    annotated = results[0].plot()

    if len(results[0].boxes) > 0:
        time_str = datetime.now().strftime("%H:%M:%S")
        st.session_state.log.append({"Time": time_str, "Alert": f"Fire detected — Risk {risk}"})

    frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB", width=700)

    st.session_state.frame_idx += 1
    st.rerun()

# Show logs
if st.session_state.log:
    df = pd.DataFrame(st.session_state.log)
    log_placeholder.table(df)
