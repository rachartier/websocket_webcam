import time

from websocket_camera import WebSocketCamera

# Adapted from: https://huggingface.co/docs/lerobot/cameras?shell_restart=Open+CV+Camera#use-cameras

SERVER_URI: str = "ws://localhost:8011"

camera = WebSocketCamera(SERVER_URI)
camera.connect()

time.sleep(1)  # Allow some time for the connection to establish


async def async_main():
    try:
        for i in range(10):
            frame = await camera.async_read(timeout_ms=200)
            if frame is not None:
                print(f"Async frame {i} shape:", frame.shape)
            else:
                print(f"Frame {i} not received in time.")
    finally:
        print("Disconnected from server.")
        camera.disconnect()


if __name__ == "__main__":
    import asyncio

    asyncio.run(async_main())
