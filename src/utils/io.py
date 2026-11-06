from datetime import datetime
from pathlib import Path
import json


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_run_dir(output_root, name):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = str(name).replace(" ", "_")
    return ensure_dir(Path(output_root) / f"{safe_name}_{stamp}")


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
