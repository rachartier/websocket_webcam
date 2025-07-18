import asyncio
import json
from types import TracebackType
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import websockets
from websockets.asyncio.server import ServerConnection

HOST: str = "localhost"
PORT: int = 8011
DEFAULT_FPS: int = 60


class CameraStream:
    def __init__(self, camera_index: int, options: dict[str, Any]) -> None:
        self.camera_index: int = camera_index
        self.options: dict[str, Any] = options.copy()
        self._fps: int = self.options.get("fps", DEFAULT_FPS)
        self.width: int = self.options.get("width", -1)
        self.height: int = self.options.get("height", -1)
        self.cap: cv2.VideoCapture | None = None

    @property
    def target_fps(self) -> int:
        return self._fps

    def update_options(self, new_options: dict[str, int]) -> None:
        self.options.update(new_options)
        if "fps" in new_options:
            self._fps = int(new_options["fps"])

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.cap = None
            raise RuntimeError(f"Cannot open camera {self.camera_index}")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        if self.cap:
            self.cap.release()
            self.cap = None

    def read_frame(self) -> tuple[bool, cv2.typing.MatLike | None]:
        if not self.cap:
            return False, None
        return self.cap.read()


def parse_camera_params(path: str) -> tuple[int, dict[str, int]]:
    """Parse camera index and options from WebSocket path"""
    parsed = urlparse(path)
    try:
        camera_index = int(parsed.path.strip("/") or 0)
    except ValueError:
        camera_index = 0

    options: dict[str, Any] = {}
    query_params = parse_qs(parsed.query)

    fps = query_params.get("fps", [None])[0]
    width = query_params.get("width", [None])[0]
    height = query_params.get("height", [None])[0]

    if fps is not None:
        options["fps"] = int(fps)

    if width is not None:
        options["width"] = int(width)

    if height is not None:
        options["height"] = int(height)

    return camera_index, options


def calculate_sleep_time(time_start: float, time_end: float, target_fps: int) -> float:
    """
    Calculate the sleep time to maintain the target FPS.
    If target_fps is 0, return 0 to avoid sleeping.
    """
    if target_fps <= 0:
        return 0.0
    elapsed_time = time_end - time_start
    target_interval = 1.0 / target_fps
    sleep_time = max(0.0, target_interval - elapsed_time)
    return sleep_time


async def send_frames(websocket: ServerConnection, camera_stream: CameraStream) -> None:
    """
    Continuously send raw JPEG frame bytes to the client at the target FPS.
    This runs independently for each client connection until the client disconnects.
    """
    try:
        with camera_stream:
            while True:
                time_start = asyncio.get_running_loop().time()
                ret, frame = camera_stream.read_frame()

                if not ret:
                    print(
                        f"Failed to grab frame from camera {camera_stream.camera_index}"
                    )
                    break

                if (
                    ret
                    and frame is not None
                    and camera_stream.width > 0
                    and camera_stream.height > 0
                ):
                    frame = cv2.resize(
                        frame, (camera_stream.width, camera_stream.height)
                    )

                ret, buffer = cv2.imencode(".jpg", frame)
                if not ret:
                    print(
                        f"Failed to encode frame from camera {camera_stream.camera_index}"
                    )
                    continue

                try:
                    await websocket.send(buffer.tobytes())
                except websockets.exceptions.ConnectionClosed:
                    print(
                        f"Client disconnected from camera {camera_stream.camera_index}"
                    )
                    break
                sleep_time = calculate_sleep_time(
                    time_start,
                    asyncio.get_running_loop().time(),
                    camera_stream.target_fps,
                )
                await asyncio.sleep(sleep_time)
    except RuntimeError as e:
        print(e)


async def handle_messages(
    websocket: ServerConnection, camera_stream: CameraStream
) -> None:
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                options = data.get("options")
                if options and "fps" in options:
                    fps = int(options["fps"])
                    if 1 <= fps <= 120:
                        camera_stream.update_options({"fps": fps})
                        print(
                            f"Updated camera {camera_stream.camera_index} FPS to: {fps}"
                        )
                    else:
                        print(f"Invalid FPS value: {fps}. Must be between 1 and 120.")
                else:
                    print(f"Invalid message format. Use 'options' field.")
            except Exception as e:
                print(f"Invalid message: {message}, error: {e}")
    except websockets.exceptions.ConnectionClosed:
        print(
            f"Client disconnected from camera {camera_stream.camera_index} message handler"
        )


async def handle_client(websocket: ServerConnection) -> None:
    """
    For each client connection, start a continuous frame stream and listen for option updates.
    """
    path = ""
    if websocket.request is not None:
        path = websocket.request.path

    camera_index, options = parse_camera_params(path)
    camera_stream = CameraStream(camera_index, options)
    print(f"Client connected to camera {camera_index} with options: {options}")

    _ = await asyncio.gather(
        send_frames(websocket, camera_stream),
        handle_messages(websocket, camera_stream),
        return_exceptions=True,
    )


async def main() -> None:
    async with websockets.serve(handle_client, HOST, PORT):
        print(f"WebSocket server started at ws://{HOST}:{PORT}")
        print("Usage:")
        print(f"  ws://{HOST}:{PORT}/0         - Camera 0 with default options")
        print(f"  ws://{HOST}:{PORT}/1?fps=30  - Camera 1 with 30 FPS")
        print('Send JSON messages to update options, e.g.: {"options": {"fps": 30}}\n')
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
