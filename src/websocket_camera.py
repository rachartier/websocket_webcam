import asyncio
import logging
import threading
import time
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt
import websockets


class WebSocketCameraError(Exception):
    """Custom exception for WebSocketCamera errors."""


class WebSocketCameraNotConnectedError(WebSocketCameraError):
    """Exception raised when trying to read a frame without an active connection."""


class WebSocketCameraFrameError(WebSocketCameraError):
    """Exception raised when there is an error with the frame data."""


class WebSocketCameraConnectionClosedError(WebSocketCameraError):
    """Exception raised when the WebSocket connection is closed unexpectedly."""


class WebSocketCamera:
    def __init__(self, server_uri: str, reconnect_delay: float = 2.0):
        self.server_uri: str = server_uri
        self.reconnect_delay: float = reconnect_delay

        self._frame: npt.NDArray[np.uint8] | None = None
        self._lock: threading.Lock = threading.Lock()
        self._stop_event: threading.Event = threading.Event()
        self._thread: threading.Thread | None = None
        self._logger: logging.Logger = logging.getLogger(__name__)

    def connect(self, warmup_read: bool = True, timeout_ms: int = 2000) -> None:
        """
        Start the WebSocket feed client in a separate thread.

        Args:
            warmup_read (bool): If True, perform a warmup read to ensure the connection is established.
            timeout_ms (int): Timeout in milliseconds for the warmup read.

        Raises:
            WebSocketCameraNotConnectedError: If the connection cannot be established or no frame is received within the timeout.
        """

        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._thread.start()

        if warmup_read:
            time_start = time.time_ns()

            while self._frame is None and not self._stop_event.is_set():
                try:
                    current_time = time.time_ns()
                    elapsed_time_ms = (
                        current_time - time_start
                    ) / 1e6  # Convert to milliseconds

                    if elapsed_time_ms >= timeout_ms:
                        raise WebSocketCameraNotConnectedError(
                            "Failed to receive a frame within the specified timeout."
                        )

                    time.sleep(0.1)  # Sleep to avoid busy waiting
                except asyncio.CancelledError:
                    break

    def disconnect(self) -> None:
        """
        Stop the WebSocket feed client and wait for the thread to finish.
        """

        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def read(self) -> npt.NDArray[np.uint8]:
        """
        Get the latest frame from the WebSocket feed.

        Returns:
            npt.NDArray[np.uint8] | None: The latest frame as a NumPy array, or None if no frame is available.
        """
        with self._lock:
            frame = self._frame.copy() if self._frame is not None else None

        if frame is None:
            raise WebSocketCameraFrameError(
                "No frame available. Ensure the camera is connected and streaming."
            )

        return frame

    def _run_async_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._connect_and_receive())
        finally:
            loop.close()

    async def _connect_and_receive(self) -> None:
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(self.server_uri) as websocket:
                    self._logger.info(f"Connected to {self.server_uri}")
                    await self._receive_frames(websocket)
            except Exception as e:
                self._logger.error(f"Connection error: {e}")

                if not self._stop_event.is_set():
                    await asyncio.sleep(self.reconnect_delay)
                else:
                    raise WebSocketCameraNotConnectedError(
                        "WebSocket connection was stopped before it could be established."
                    )

    async def _receive_frames(self, websocket: websockets.ClientConnection) -> None:
        while not self._stop_event.is_set():
            try:
                data = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                if isinstance(data, str):
                    data = data.encode()

                frame: npt.NDArray[np.uint8] | None = cast(
                    npt.NDArray[np.uint8] | None,
                    cv2.imdecode(
                        np.frombuffer(data, dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    ),
                )

                if frame is not None:
                    with self._lock:
                        self._frame = frame

            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                self._logger.warning("WebSocket connection closed")
                raise WebSocketCameraConnectionClosedError(
                    "WebSocket connection closed unexpectedly."
                )
