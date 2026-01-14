# 🌲 Forest Fire Detection System

An AI-powered system that detects forest fires from video feeds using computer vision and combines it with satellite-based environmental risk analysis for early warning and better disaster response.

## 🔍 Overview

This project uses:

A deep learning model (YOLO) to detect fire and smoke from drone or surveillance video.

Satellite imagery (Sentinel-2 via Google Earth Engine) to calculate vegetation health (NDVI) and fire risk.

A dashboard to visualize detections, location, risk levels, and logs.

The goal is to enable early detection of forest fires, reduce response time, and minimize environmental and economic damage.

## 🚀 Features

Fire and smoke detection using YOLO

Bounding box visualization on video frames

Multi-video support with navigation

Satellite-based fire risk classification (Low / Medium / High)

Interactive map showing fire location

Detection logs with timestamps

Scalable and modular architecture

## 🧠 Technology Stack
| Component	| Technology |
|------|------|
| Language	| Python 3.11 |
| Computer Vision	| OpenCV |
| Deep Learning	| YOLO (Ultralytics) |
| Satellite | Data	Google Earth Engine |
| Mapping |	Folium |
| Dashboard	| Streamlit |
## 📂 Project Structure
```.
├── dashboard.py
├── fire.pt
├── satellite_risk.py
├── video_locations.py
├── requirements.txt
└── videos/
    ├── fire1.mp4
    ├── fire2.mp4
    ├── ...
```
## ⚙️ Installation
1️⃣ Clone the repository
`git clone https://github.com/yourusername/forest-fire-detection.git
cd forest-fire-detection`

2️⃣ Create and activate a virtual environment
`python -m venv venv
venv\Scripts\activate`   # Windows

3️⃣ Install dependencies
`pip install -r requirements.txt`

4️⃣ Authenticate Google Earth Engine
`earthengine authenticate`

▶️ Running the App
`python -m streamlit run dashboard.py`


Then open the browser link shown in the terminal.

## 📊 Fire Risk Classification

Satellite-based risk is computed using NDVI:

NDVI Value	Risk Level
> 0.5	Low
0.3 – 0.5	Medium
< 0.3	High
🧪 How It Works

Video frames are extracted using OpenCV.

YOLO detects fire and smoke patterns.

Satellite NDVI is computed for the corresponding location.

Results are visualized on the dashboard with logs and map.

Alerts and risk information support faster response.

## 🌍 Applications

- Forest monitoring agencies

- Disaster management authorities

- Wildlife conservation organizations

- Smart city surveillance systems

- Climate and environmental research

## 📝 Future Improvements

- Severity classification (small/medium/large fire)

- Fire spread direction prediction

- Cloud deployment for large-scale monitoring

- Integration with alert systems (SMS/Telegram)

- Support for live drone streams

## 🤝 Contributing

Contributions are welcome!
Please open an issue or submit a pull request for improvements or bug fixes.

📄 License

This project is licensed under the MIT License.
