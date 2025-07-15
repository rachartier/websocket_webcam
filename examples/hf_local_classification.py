# pyright: basic

import time

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image
from transformers.pipelines import pipeline

from websocket_camera import WebSocketCamera

detector = pipeline("object-detection", model="facebook/detr-resnet-50")

SERVER_URI = "ws://localhost:8011"


def detect_objects(image: npt.NDArray[np.uint8]) -> list[dict]:
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    results = detector(pil_img)
    return results


def draw_boxes(image: npt.NDArray[np.uint8], detections: list[dict]) -> np.ndarray:
    for det in detections:
        box = det["box"]
        label = det["label"]
        score = det["score"]
        x_min, y_min, x_max, y_max = (
            int(box["xmin"]),
            int(box["ymin"]),
            int(box["xmax"]),
            int(box["ymax"]),
        )
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        text = f"{label}: {score:.2f}"
        cv2.putText(
            image,
            text,
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    return image


if __name__ == "__main__":
    cam = WebSocketCamera(SERVER_URI)
    cam.start()
    print("Connected to server.")

    try:
        while True:
            frame = cam.get_frame()
            if frame is not None:
                detections = detect_objects(frame)
                frame_with_boxes = draw_boxes(frame.copy(), detections)

                cv2.imshow("Webcam Object Detection", frame_with_boxes)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.05)
    finally:
        cam.stop()
        cv2.destroyAllWindows()
