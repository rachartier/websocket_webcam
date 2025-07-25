import os
import sys
import time

import cv2

# Ensure src/ is in sys.path for import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.websocket_camera import WebSocketCamera

SERVER_URI: str = "ws://localhost:8011"

if __name__ == "__main__":
    cam = WebSocketCamera(
        SERVER_URI,
        0,
        fps=60,
        reconnect_delay=2.0,
    )
    cam.connect()

    print("Connected to server.")

    try:
        while True:
            frame = cam.get_latest_frame()
            cv2.imshow("Webcam Frame", frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                break

            time.sleep(0.01)
    finally:
        cam.disconnect()
        cv2.destroyAllWindows()
