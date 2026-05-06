from pathlib import Path

MODEL_PATH = "models/face_landmarker.task"

VIDEO_PATH = "data/input/input.mp4"

OUTPUT_DIR = Path("data/output/video")
OUTPUT_VIDEO = OUTPUT_DIR / "output.mp4"

PATIENT_ID = "patient_001"

LEFT_NOSTRIL_IDS = [79, 20, 60, 75, 166, 239, 238, 59]
RIGHT_NOSTRIL_IDS = [309, 459, 458, 250, 290, 305, 289, 392]