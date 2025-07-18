"""
WebSocket Camera Client

A WebSocket camera client for receiving video frames from a remote server.

Example:
    # Connect to camera 0
    camera = WebSocketCamera("ws://localhost:8011")
    camera.connect()
    frame = camera.get_latest_frame()
    camera.disconnect()

    # Connect to camera 1 with custom FPS
    camera = WebSocketCamera("ws://localhost:8011", camera_index=1, fps=30)

    # Or use context manager:
    with WebSocketCamera("ws://localhost:8011", camera_index=2) as camera:
        frame = camera.get_latest_frame()
        # Change FPS dynamically
        camera.set_fps(15)
"""

import asyncio
import json
import logging
import threading
import time
from types import TracebackType
from typing import cast
from urllib.parse import urlencode

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
    WebSocket camera client for receiving video frames from multiple cameras.

    Args:
        server_uri: WebSocket server base URI (e.g., "ws://localhost:8011")
        camera_index: Camera index (defaults to 0)
        fps: Target FPS for the camera stream (optional)
        reconnect_delay: Delay between reconnection attempts in seconds
    """

    def __init__(
        self,
        server_uri: str,
        camera_index: int = 0,
        fps: int = 60,
        width: int | None = None,
        height: int | None = None,
        reconnect_delay: float = 2.0,
    ) -> None:
        if not server_uri.startswith(("ws://", "wss://")):
            raise ValueError("server_uri must start with ws:// or wss://")

        self.server_uri: str = server_uri.rstrip("/")
        self.camera_index: int = camera_index
        self.fps: int = fps
        self.width: int | None = width
        self.height: int | None = height
        self.reconnect_delay: float = reconnect_delay

        self._frame: npt.NDArray[np.uint8] | None = None
        self._lock: threading.Lock = threading.Lock()
        self._stop_event: threading.Event = threading.Event()
        self._thread: threading.Thread | None = None
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._websocket: websockets.ClientConnection | None = None

        self._full_uri: str = self._build_uri()

    def _build_uri(self) -> str:
        """Build the full WebSocket URI with camera index and options."""
        uri: str = f"{self.server_uri}/{self.camera_index}"

        query_params: dict[str, int] = {
            "fps": self.fps,
            "width": self.width if self.width is not None else -1,
            "height": self.height if self.height is not None else -1,
        }
        uri += f"?{urlencode(query_params)}"

        return uri

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

    def set_fps(self, fps: int) -> None:
        """
        Change the FPS of the camera stream dynamically.

        Args:
            fps: New target FPS (1-120)
        """
        if not (1 <= fps <= 120):
            raise ValueError("FPS must be between 1 and 120")

        self.fps = fps

        if self._websocket and not self._websocket.closed:
            asyncio.run_coroutine_threadsafe(
                self._send_options_update({"fps": fps}), self._get_event_loop()
            )

    def set_options(self, **options: int) -> None:
        """
        Update camera options dynamically.

        Args:
            **options: Camera options (e.g., fps=30)
        """
        if self._websocket and not self._websocket.closed:
            asyncio.run_coroutine_threadsafe(
                self._send_options_update(options), self._get_event_loop()
            )

    def _get_event_loop(self) -> asyncio.AbstractEventLoop:
        """Get the event loop from the background thread."""
        if self._thread and hasattr(self._thread, "_loop"):
            return self._thread._loop  # type: ignore
        raise RuntimeError("No event loop available")

    async def _send_options_update(self, options: dict[str, int]) -> None:
        """Send options update to server."""
        if self._websocket:
            try:
                message: str = json.dumps({"options": options})
                await self._websocket.send(message)
                self._logger.info(f"Updated options: {options}")
            except Exception as e:
                self._logger.error(f"Failed to send options update: {e}")

    def get_latest_frame(self) -> npt.NDArray[np.uint8]:
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
        loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        self._thread._loop = loop

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
                async with websockets.connect(self._full_uri) as websocket:
                    self._websocket = websocket
                    self._logger.info(
                        f"Connected to camera {self.camera_index} at {self._full_uri}"
                    )
                    await self._receive_frames(websocket)
            except Exception as e:
                self._logger.warning(f"Connection lost: {e}")
                if not self._stop_event.is_set():
                    await asyncio.sleep(self.reconnect_delay)
            finally:
                self._websocket = None

    async def _receive_frames(self, websocket: websockets.ClientConnection) -> None:
        """Receive only binary frames from WebSocket."""
        while not self._stop_event.is_set():
            try:
                data = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                frame_array: npt.NDArray[np.uint8] = np.frombuffer(data, dtype=np.uint8)
                frame: npt.NDArray[np.uint8] | None = cv2.imdecode(
                    frame_array, cv2.IMREAD_COLOR
                )
                if frame is not None:
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
