"""
WebSocket Camera Client

A WebSocket camera client for receiving video frames from a remote server.

Example:
    camera = WebSocketCamera("ws://localhost:8000/stream")
    camera.connect()
    frame = camera.read()
    camera.disconnect()

    # Or use context manager:
    with WebSocketCamera("ws://localhost:8000/stream") as camera:
        frame = camera.read()
"""

import asyncio
import logging
import threading
import time
from types import TracebackType
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt
import websockets


class WebSocketCameraError(Exception):
    """Base exception for WebSocketCamera errors."""


class WebSocketCameraNotConnectedError(WebSocketCameraError):
    """Exception raised when trying to read without connection."""


class WebSocketCameraFrameError(WebSocketCameraError):
    """Exception raised when no frame is available."""


class WebSocketCamera:
    """
    Simple WebSocket camera client for receiving video frames.

    Args:
        server_uri: WebSocket server URI (e.g., "ws://localhost:8000/stream")
        reconnect_delay: Delay between reconnection attempts in seconds
    """

    def __init__(self, server_uri: str, reconnect_delay: float = 2.0) -> None:
        if not server_uri.startswith(("ws://", "wss://")):
            raise ValueError("server_uri must start with ws:// or wss://")

        self.server_uri: str = server_uri
        self.reconnect_delay: float = reconnect_delay

        self._frame: npt.NDArray[np.uint8] | None = None
        self._lock: threading.Lock = threading.Lock()
        self._stop_event: threading.Event = threading.Event()
        self._thread: threading.Thread | None = None
        self._logger: logging.Logger = logging.getLogger(__name__)

    def __enter__(self) -> "WebSocketCamera":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit."""
        self.disconnect()

    @property
    def is_connected(self) -> bool:
        """Check if camera is connected."""
        return self._thread is not None and self._thread.is_alive()

    def connect(self, timeout: float = 5.0) -> None:
        """
        Start the WebSocket camera client.

        Args:
            timeout: Seconds to wait for first frame
        """
        if self.is_connected:
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._thread.start()

        # Wait for first frame
        start_time = time.time()
        while self._frame is None and (time.time() - start_time) < timeout:
            if self._stop_event.is_set():
                raise WebSocketCameraNotConnectedError("Connection failed")
            time.sleep(0.1)

        if self._frame is None:
            self.disconnect()
            raise WebSocketCameraNotConnectedError("No frame received within timeout")

    def disconnect(self) -> None:
        """Stop the WebSocket camera client."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None

        with self._lock:
            self._frame = None

    def fetch_last_frame(self) -> npt.NDArray[np.uint8]:
        """
        Get the latest frame.

        Returns:
            Latest frame as numpy array

        Raises:
            WebSocketCameraNotConnectedError: If not connected
            WebSocketCameraFrameError: If no frame available
        """
        if not self.is_connected:
            raise WebSocketCameraNotConnectedError("Camera not connected")

        with self._lock:
            if self._frame is None:
                raise WebSocketCameraFrameError("No frame available")
            return self._frame.copy()

    def _run_async_loop(self) -> None:
        """Run async event loop in background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._connect_and_receive())
        except Exception as e:
            self._logger.error(f"Connection error: {e}")
        finally:
            loop.close()

    async def _connect_and_receive(self) -> None:
        """Connection loop with auto-reconnection."""
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(self.server_uri) as websocket:
                    self._logger.info(f"Connected to {self.server_uri}")
                    await self._receive_frames(websocket)
            except Exception as e:
                self._logger.warning(f"Connection lost: {e}")
                if not self._stop_event.is_set():
                    await asyncio.sleep(self.reconnect_delay)

    async def _receive_frames(self, websocket: websockets.ClientConnection) -> None:
        """Receive frames from WebSocket."""
        while not self._stop_event.is_set():
            try:
                data = await asyncio.wait_for(websocket.recv(), timeout=5.0)

                # Convert string to bytes if needed
                if isinstance(data, str):
                    data = data.encode()

                # Decode frame
                frame_array = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                with self._lock:
                    self._frame = cast(npt.NDArray[np.uint8], frame)

            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                self._logger.info("WebSocket connection closed")
                break
            except Exception as e:
                self._logger.error(f"Frame error: {e}")
                continue
