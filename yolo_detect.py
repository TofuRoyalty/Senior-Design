"""Basic YOLO object detection for testing.

Usage:
    python yolo_detect.py image.jpg            # run on an image
    python yolo_detect.py video.mp4            # run on a video file
    python yolo_detect.py 0                    # run on webcam 0 (press q to quit)
    python yolo_detect.py image.jpg --model yolov8s.pt --conf 0.4
"""
import argparse

import cv2
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Run YOLO detection on an image, video, or webcam.")
    p.add_argument("source", help="image/video path, or webcam index (e.g. 0)")
    p.add_argument("--model", default="yolov8n.pt", help="weights file (auto-downloaded if missing)")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--save", action="store_true", help="save annotated output to runs/detect")
    p.add_argument("--no-show", action="store_true", help="don't open a display window")
    return p.parse_args()


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    model = YOLO(args.model)

    for result in model.predict(source, conf=args.conf, stream=True, save=args.save, verbose=False):
        names = result.names
        detections = [
            (names[int(b.cls)], float(b.conf), [round(v) for v in b.xyxy[0].tolist()])
            for b in result.boxes
        ]
        print(f"{result.path}: {len(detections)} detection(s)")
        for label, conf, box in detections:
            print(f"  {label:<15} {conf:.2f}  {box}")

        if not args.no_show:
            cv2.imshow("YOLO", result.plot())
            # Images wait for a keypress; video/webcam advance automatically.
            if cv2.waitKey(0 if isinstance(source, str) and source.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp")) else 1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
