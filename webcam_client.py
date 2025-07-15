import asyncio

import cv2
import numpy as np
import websockets
from PIL import Image

SERVER_URI = "ws://localhost:8010"


async def receive_frames():
    async with websockets.connect(SERVER_URI) as websocket:
        print("Connected to server.")
        while True:
            data = await websocket.recv()
            np_data = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

            if frame is None:
                print("Failed to decode frame")
                continue

            cv2.imshow("Webcam Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # you can process the PIL image here

        cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(receive_frames())
