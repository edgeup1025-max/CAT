import json, os, subprocess
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "mlflow_config.json"
with open(CONFIG_PATH) as f:
    cfg = json.load(f)

tracking_uri = cfg["tracking_uri"]
port = cfg.get("default_port", 5000)

cmd = [
    "mlflow", "ui", "--backend-store-uri", tracking_uri, "--port",
    str(port)
]

print(f"[INFO] Starting MLflow UI at {tracking_uri} (port={port})...")
subprocess.run(cmd)
