You need to create 2 separate environments for this project, otherwise WSL2/Powershell will not like using the same environment for both.

## Create the first environment

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


## Execute the script

First, start the Windows server script:
```powershell
python .\webcam_server.py
```

Then, in a separate WSL2 terminal, run the client script:
```bash
python3 webcam_client.py
```

You should see the webcam feed from the Windows machine in a WSL2 window.
