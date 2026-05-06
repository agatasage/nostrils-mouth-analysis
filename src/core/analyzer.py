import cv2
import mediapipe as mp

from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarkerOptions, FaceLandmarker

from src.core.geometry import (
    get_point,
    distance,
    compute_axes,
    compute_center
)
from src.config import LEFT_NOSTRIL_IDS, RIGHT_NOSTRIL_IDS, MODEL_PATH


class FaceAnalyzer:
    def __init__(self):
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

    def process_frame(self, frame, frame_id):
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = self.landmarker.detect_for_video(mp_image, frame_id * 33)

        if not result.face_landmarks:
            return frame, None

        landmarks = result.face_landmarks[0]

        # --- MOUTH ---
        top_lip = get_point(landmarks, 13, w, h)
        bottom_lip = get_point(landmarks, 14, w, h)
        left_mouth = get_point(landmarks, 78, w, h)
        right_mouth = get_point(landmarks, 308, w, h)

        mar = distance(top_lip, bottom_lip) / (distance(left_mouth, right_mouth) + 1e-6)
        mouth_open = mar > 0.05

        # --- NOSTRILS ---
        left_nostril = compute_center(LEFT_NOSTRIL_IDS, landmarks, w, h)
        right_nostril = compute_center(RIGHT_NOSTRIL_IDS, landmarks, w, h)

        nostril_width = distance(left_nostril, right_nostril)

        # --- NORMALIZATION ---
        left_eye = get_point(landmarks, 33, w, h)
        right_eye = get_point(landmarks, 263, w, h)
        eye_dist = distance(left_eye, right_eye)

        nostril_width /= (eye_dist + 1e-6)

        # --- AXES ---
        _, left_long, left_short = compute_axes(LEFT_NOSTRIL_IDS, landmarks, w, h)
        _, right_long, right_short = compute_axes(RIGHT_NOSTRIL_IDS, landmarks, w, h)

        # --- DRAW ---
        cv2.circle(frame, left_nostril.astype(int), 4, (0, 0, 255), -1)
        cv2.circle(frame, right_nostril.astype(int), 4, (255, 0, 0), -1)

        cv2.putText(frame, f"MAR: {mar:.2f}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"Mouth: {'OPEN' if mouth_open else 'CLOSED'}",
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 0, 0) if mouth_open else (0, 255, 0), 2)

        cv2.putText(frame, f"Nostril width: {nostril_width:.3f}",
                    (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 0), 2)

        return frame, {
            "frame": frame_id,
            "mar": mar,
            "mouth_open": int(mouth_open),
            "nostril_width": nostril_width,
            "left_long": left_long,
            "left_short": left_short,
            "right_long": right_long,
            "right_short": right_short,
            "eye_distance": eye_dist
        }