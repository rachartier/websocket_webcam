import time

from lerobot_wrapper import WebSocketCameraConfig, WebSocketCameraWrapper

# Adapted from: https://huggingface.co/docs/lerobot/cameras?shell_restart=Open+CV+Camera#use-cameras

SERVER_URI: str = "ws://localhost:8011"

config = WebSocketCameraConfig(
    server_uri=SERVER_URI,
    width=256,
    height=256,
)

camera = WebSocketCameraWrapper(config)
camera.connect()

time.sleep(1)  # Allow some time for the connection to establish

if __name__ == "__main__":
    try:
        for i in range(10):
            frame = camera.async_read(timeout_ms=2000)
            print(f"Async frame {i} shape:", frame.shape)
    finally:
        print("Disconnected from server.")
        camera.disconnect()
