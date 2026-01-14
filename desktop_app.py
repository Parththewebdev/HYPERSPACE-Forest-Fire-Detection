import tkinter as tk
from ultralytics import YOLO
import cv2
from PIL import Image, ImageTk
import os
from satellite_risk import get_fire_risk
from video_locations import video_locations
from datetime import datetime
import folium
import tempfile
import webbrowser

# ---------- Config ----------
VIDEO_DIR = "videos"
BG = "#121212"
FG = "#E0E0E0"
ACCENT = "#FF5252"

model = YOLO("fire.pt")
videos = sorted([v for v in os.listdir(VIDEO_DIR) if v.endswith(".mp4")])
vid_idx = 0
cap = None
running = False

# ---------- UI ----------
root = tk.Tk()
root.title("Forest Fire Detection")
root.geometry("1200x800")
root.configure(bg=BG)

# Title
title = tk.Label(root, text="FOREST FIRE DETECTION", font=("Segoe UI", 22, "bold"), bg=BG, fg=FG)
title.grid(row=0, column=0, columnspan=2, pady=10)

risk_label = tk.Label(root, text="", font=("Segoe UI", 14), bg=BG, fg=ACCENT)
risk_label.grid(row=1, column=0, columnspan=2)

# Panels
map_frame = tk.Frame(root, width=550, height=400, bg="#1E1E1E")
map_frame.grid(row=2, column=0, padx=20, pady=10)
map_frame.grid_propagate(False)

video_frame = tk.Frame(root, width=550, height=400, bg="#1E1E1E")
video_frame.grid(row=2, column=1, padx=20, pady=10)
video_frame.grid_propagate(False)

map_label = tk.Label(map_frame, text="Fire Location Map", bg="#1E1E1E", fg=FG)
map_label.pack(pady=5)

video_label_title = tk.Label(video_frame, text="Video Source", bg="#1E1E1E", fg=FG)
video_label_title.pack(pady=5)

map_widget = tk.Label(map_frame)
map_widget.pack()

video_widget = tk.Label(video_frame)
video_widget.pack()

# Buttons
btn_frame = tk.Frame(root, bg=BG)
btn_frame.grid(row=3, column=0, columnspan=2, pady=15)

def styled_btn(text, cmd):
    return tk.Button(btn_frame, text=text, command=cmd, bg="#1E1E1E", fg=FG,
                     activebackground="#333", relief="flat", padx=20, pady=8)

styled_btn("Prev", lambda: switch_video(-1)).pack(side=tk.LEFT, padx=20)
styled_btn("Next", lambda: switch_video(1)).pack(side=tk.LEFT, padx=20)

# Logs
log_box = tk.Text(root, height=8, width=140, bg="#1E1E1E", fg="#00FF9C", borderwidth=0)
log_box.grid(row=4, column=0, columnspan=2, pady=10)

# ---------- Logic ----------
def update_risk():
    name = videos[vid_idx]
    lat, lon = video_locations.get(name, (30.1, 79.2))
    risk = get_fire_risk(lat, lon)
    risk_label.config(text=f"Satellite Fire Risk: {risk}")
    update_map(lat, lon)

def update_map(lat, lon):
    m = folium.Map(location=[lat, lon], zoom_start=6)
    folium.Marker([lat, lon]).add_to(m)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    m.save(tmp.name)
    webbrowser.open(tmp.name)

def start_video():
    global cap, running
    if cap:
        cap.release()

    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, videos[vid_idx]))
    running = True
    update_risk()
    play_loop()

def play_loop():
    if not running:
        return

    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()

    results = model(frame, conf=0.4)
    annotated = results[0].plot()

    if len(results[0].boxes) > 0:
        t = datetime.now().strftime("%H:%M:%S")
        log_box.insert(tk.END, f"[{t}] Fire detected in {videos[vid_idx]}\n")
        log_box.see(tk.END)

    frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img = img.resize((520, 340))
    imgtk = ImageTk.PhotoImage(img)

    video_widget.imgtk = imgtk
    video_widget.config(image=imgtk)

    root.after(30, play_loop)

def switch_video(delta):
    global vid_idx
    vid_idx = (vid_idx + delta) % len(videos)
    start_video()

# ---------- Start ----------
start_video()
root.mainloop()
