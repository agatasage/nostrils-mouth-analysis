# Facial landmark analysis: monitoring the width of the nostril axis and the state of the mouth (open/closed) during speech and silence

This project performs real-time facial landmark analysis on video using MediaPipe.  
It extracts facial metrics such as mouth activity, nostril dynamics, and eye-normalized measurements, then exports both annotated video and structured CSV data.

---

## Features

- Face landmark detection using MediaPipe Tasks API
- Mouth Activity Ratio (MAR) computation
- Nostril geometry analysis (width + axis estimation)
- Eye-distance normalization for scale invariance
- Annotated video output
- Structured CSV export of frame-by-frame measurements

---

## Project Structure

project_root/
├── src/
│ ├── main.py # Entry point
│ ├── config.py # Paths and constants
│ ├── core/
│ │ ├── analyzer.py # Face analysis logic
│ │ ├── geometry.py # Math utilities
│ ├── utils/
│ │ ├── io.py # CSV utilities
│
├── models/
│ └── face_landmarker.task
│
├── data/
│ ├── input/ # Input videos
│ └── output/
│ ├── video/ # Processed videos
│ └── patient_001.csv # Output metrics
│
└── requirements.txt


---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/face-analysis.git
cd face-analysis

### 2. Create virtual environment (optional)
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

### 3. Install dependencies
```pip install -r requirements.txt
