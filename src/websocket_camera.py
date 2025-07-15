import asyncio
import logging
import threading
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt
import websockets


class WebSocketCamera:
    def __init__(self, server_uri: str, reconnect_delay: float = 2.0):
        self.server_uri: str = server_uri
        self.reconnect_delay: float = reconnect_delay

        self._frame: npt.NDArray[np.uint8] | None = None
        self._lock: threading.Lock = threading.Lock()
        self._stop_event: threading.Event = threading.Event()
        self._thread: threading.Thread | None = None
        self._logger: logging.Logger = logging.getLogger(__name__)

    def start(self) -> None:
        """
        Start the WebSocket camera client in a separate thread.
        """

        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """
        Stop the WebSocket camera client and wait for the thread to finish.
        """

        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def get_frame(self) -> npt.NDArray[np.uint8] | None:
        """
        Get the latest frame from the WebSocket camera.

        Returns:
            npt.NDArray[np.uint8] | None: The latest frame as a NumPy array, or None if no frame is available.
        """
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

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
                await self._receive_frames()
            except Exception as e:
                self._logger.error(f"Connection error: {e}")
                if not self._stop_event.is_set():
                    await asyncio.sleep(self.reconnect_delay)

    async def _receive_frames(self) -> None:
        async with websockets.connect(self.server_uri) as websocket:
            self._logger.info(f"Connected to {self.server_uri}")

            while not self._stop_event.is_set():
                try:
                    data = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    if isinstance(data, str):
                        data = data.encode()

                    frame: npt.NDArray[np.uint8] | None = cast(
                        npt.NDArray[np.uint8] | None,
                        cv2.imdecode(
                            np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR
                        ),
                    )

                    if frame is not None:
                        with self._lock:
                            self._frame = frame

                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    self._logger.warning("WebSocket connection closed")
                    break
