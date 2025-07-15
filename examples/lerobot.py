import time

from websocket_camera import WebSocketCamera

# Adapted from: https://huggingface.co/docs/lerobot/cameras?shell_restart=Open+CV+Camera#use-cameras

SERVER_URI: str = "ws://localhost:8011"

camera = WebSocketCamera(SERVER_URI)
camera.connect()

time.sleep(1)  # Allow some time for the connection to establish


if __name__ == "__main__":
    try:
        for i in range(10):
            frame = camera.fetch_last_frame()

            print(f"Sync frame {i} shape:", frame.shape)
    finally:
        print("Disconnected from server.")
        camera.disconnect()
