import cv2
from pathlib import Path

from src.core.analyzer import FaceAnalyzer
from src.utils.io import save_csv
from src.config import VIDEO_PATH, OUTPUT_VIDEO, OUTPUT_DIR, PATIENT_ID


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    analyzer = FaceAnalyzer()
    data_log = []

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, data = analyzer.process_frame(frame, frame_id)

        if data:
            data["time_sec"] = frame_id / fps
            data_log.append(data)

        cv2.imshow("Face Analysis", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    #save_csv(f"{PATIENT_ID}.csv", data_log)

    output_path = Path("data/output/csv") / f"{PATIENT_ID}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_csv(output_path, data_log)


if __name__ == "__main__":
    main()