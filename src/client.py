import time

import cv2
from PIL import Image

from websocket_camera import WebSocketCamera

SERVER_URI: str = "ws://localhost:8011"

if __name__ == "__main__":
    cam = WebSocketCamera(SERVER_URI)
    cam.start()

    print("Connected to server.")

    try:
        while True:
            frame = cam.get_frame()
            if frame is not None:
                cv2.imshow("Webcam Frame", frame)
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # you can process the PIL image here

            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                break

            time.sleep(0.01)
    finally:
        cam.stop()
        cv2.destroyAllWindows()
