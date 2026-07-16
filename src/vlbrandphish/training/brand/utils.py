from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def print_config(cfg: Dict[str, Any]) -> None:
    print("\n======== CONFIG ========")
    print(json.dumps(cfg, indent=2, ensure_ascii=False))
    print("========================\n")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
