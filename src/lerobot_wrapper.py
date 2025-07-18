"""
LeRobot Camera Wrapper for WebSocketCamera

This module provides a wrapper that adapts the WebSocketCamera class
to the LeRobot Camera interface.
"""

from dataclasses import dataclass
from typing import Any, override

import cv2
from cv2.typing import MatLike

from lerobot.cameras import Camera, CameraConfig, ColorMode
from websocket_camera import WebSocketCamera


@dataclass
class WebSocketCameraConfig(CameraConfig):
    server_uri: str
    reconnect_delay: float = 2.0


class WebSocketCameraWrapper(Camera):
    """
    LeRobot-compatible wrapper for WebSocketCamera.

    This wrapper adapts the WebSocketCamera to implement the LeRobot Camera interface,
    allowing it to be used within the LeRobot ecosystem.

    Args:
        config: Camera configuration
        server_uri: WebSocket server URI
    """

    def __init__(
        self,
        config: WebSocketCameraConfig,
    ):
        super().__init__(config)
        self._camera: WebSocketCamera = WebSocketCamera(
            config.server_uri,
            fps=config.fps,
            reconnect_delay=config.reconnect_delay,
        )

    @property
    @override
    def is_connected(self) -> bool:
        """Check if the camera is currently connected."""
        return self._camera.is_connected

    @staticmethod
    @override
    def find_cameras() -> list[dict[str, Any]]:
        """
        Find available WebSocket cameras.

        Note: WebSocket cameras cannot be automatically discovered.
        This returns an empty list as discovery requires manual configuration.
        """
        return []

    @override
    def connect(self, warmup: bool = True) -> None:
        """
        Establish connection to the WebSocket camera.

        Args:
            warmup: If True, waits for first frame before returning
        """
        timeout = 5.0 if warmup else 0.1
        self._camera.connect(timeout=timeout)

    @override
    def read(self, color_mode: ColorMode | None = None) -> MatLike:
        """
        Capture and return a single frame from the camera.

        Args:
            color_mode: Desired color mode (RGB, BGR, GRAY).
                        If None, returns in BGR format (OpenCV default).

        Returns:
            MatLike: Captured frame as a numpy array

        Raises:
            WebSocketCameraError: If camera is not connected or no frame available
        """
        frame = self._camera.get_latest_frame()

        if color_mode == ColorMode.RGB:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame

    @override
    def async_read(self, timeout_ms: float = 5000) -> MatLike:
        """
        Asynchronously capture and return a single frame.

        Note: WebSocketCamera operates asynchronously internally,
        so this delegates to the regular read method.

        Args:
            timeout_ms: Timeout in milliseconds (unused for WebSocket camera)

        Returns:
            MatLike: Captured frame as a numpy array
        """
        return self.read()

    @override
    def disconnect(self) -> None:
        """Disconnect from the camera"""
        self._camera.disconnect()
