"""Live person detection from a USB camera.

Usage:
    python live_people.py                  # camera index 0
    python live_people.py --camera 1       # pick another camera
    python live_people.py --conf 0.5 --model yolov8s.pt

Press q (or Esc) in the window to quit.
"""
import argparse
import time

import cv2
from ultralytics import YOLO

PERSON_CLASS = 0  # COCO class id for "person"


def parse_args():
    p = argparse.ArgumentParser(description="Live person detection from a USB camera.")
    p.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    p.add_argument("--model", default="yolov8n.pt", help="weights file")
    p.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    p.add_argument("--width", type=int, default=1280, help="requested capture width")
    p.add_argument("--height", type=int, default=720, help="requested capture height")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)

    # CAP_DSHOW opens USB cameras much faster on Windows than the default backend.
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Try --camera 1.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    prev = time.time()
    fps = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame read failed; exiting.")
                break

            result = model.predict(frame, classes=[PERSON_CLASS], conf=args.conf, verbose=False)[0]
            annotated = result.plot()  # draws boxes + confidence

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(annotated, f"People: {len(result.boxes)}  FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow("Live people detection (q to quit)", annotated)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
