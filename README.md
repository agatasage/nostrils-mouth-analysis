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

```text
project_root/
├── src/
│   ├── main.py              # Entry point
│   ├── config.py            # Paths and constants
│   ├── core/
│   │   ├── analyzer.py      # Face analysis logic
│   │   ├── geometry.py      # Math utilities
│   ├── utils/
│   │   ├── io.py            # CSV utilities
│
├── models/
│   └── face_landmarker.task
│
├── data/
│   ├── input/               # Input videos
│   └── output/
│       ├── video/           # Processed videos
│       └── patient_001.csv # Output metrics
│
├── requirements.txt
└── README.md
```

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
pip install -r requirements.txt

---
## Requirements
Python 3.9+
OpenCV
NumPy
MediaPipe

---

## Usage
```bash
python -m src.main

## Output Files

### Processed Video

Saved to:

```text
data/output/video/
```

Includes:
- Facial landmarks overlay  
- Mouth state (OPEN / CLOSED)  
- Real-time metric visualization  

---

### CSV Metrics

Saved as:

```text
data/output/patient_001.csv
```

Each row contains:

- `frame` – frame index  
- `time_sec` – timestamp  
- `mar` – mouth activity ratio  
- `mouth_open` – binary state (0/1)  
- `nostril_width` – normalized distance  
- `left_long`, `left_short` – nostril geometry  
- `right_long`, `right_short`  
- `eye_distance`  

---

## Key Metrics

### Mouth Activity Ratio (MAR)

Measures mouth opening:

```text
MAR = vertical_lip_distance / mouth_width
```

Used to detect whether the mouth is open or closed.

---

### Normalized Nostril Width

Scale-invariant measurement:

```text
nostril_width = nostril_distance / eye_distance
```

Ensures consistency across different face sizes and camera distances.
