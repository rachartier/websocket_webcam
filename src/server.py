import asyncio

import cv2
import websockets
from websockets.asyncio.server import ServerConnection

HOST: str = "localhost"
PORT: int = 8011
TARGET_FPS: int = 60


def calculate_sleep_time(
    start_time: float,
    end_time: float,
    target_fps: int,
) -> float:
    elapsed_time = end_time - start_time
    sleep_time = max(0, (1 / target_fps) - elapsed_time)

    return sleep_time


async def send_frames(websocket: ServerConnection) -> None:
    cap: cv2.VideoCapture = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    try:
        while True:
            time_start = asyncio.get_event_loop().time()

            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                print("Failed to encode frame")
                continue

            try:
                await websocket.send(buffer.tobytes())
            except websockets.exceptions.ConnectionClosed:
                print("Client disconnected")
                break

            sleep_time = calculate_sleep_time(
                time_start,
                asyncio.get_event_loop().time(),
                TARGET_FPS,
            )
            await asyncio.sleep(sleep_time)

    finally:
        cap.release()


async def main() -> None:
    async with websockets.serve(send_frames, HOST, PORT):
        print(f"WebSocket server started at ws://{HOST}:{PORT}")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
