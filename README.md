You need to create 2 separate environments for this project, otherwise WSL2/Powershell will not like using the same environment for both.

## Create the first environment

(Use requirements_with_torch.txt if you want to try `hf_local_classification.py` example)

### On Windows
```powershell
python -m venv .venv_win
.\.venv_win\Scripts\Activate.ps1

pip install -r requirements.txt
```

### On WSL2
```bash
python3 -m venv .venv_wsl
source .venv_wsl/bin/activate
pip install -r requirements.txt
```


## Execute the scripts

First, start the Windows server script:
```powershell
python .\src\server.py
```

Then, in a separate WSL2 terminal, run the demo client script:
```bash
python3 examples/webcam_preview.py
```

You should see the webcam feed from the Windows machine in a WSL2 window.

## Using the WebSocketCamera class

The main reusable camera logic is in `src/websocket_camera.py`.
To use the thread-safe WebSocketCamera class in your own code, import it as follows:

```python
from websocket_camera import WebSocketCamera
```

## Examples

The `examples/` directory contains sample scripts demonstrating how to use the core functionality of this project. These scripts provide practical usage scenarios, such as running local image classification or previewing webcam streams.

- `hf_local_classification.py`: Example of running local image classification using Hugging Face models.
- `webcam_preview.py`: Simple script to preview webcam video in a window.
