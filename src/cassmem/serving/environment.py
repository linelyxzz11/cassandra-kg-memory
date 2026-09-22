"""Repository paths and local environment loading for experiment entry points."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def load_local_env() -> None:
    """Load an ignored ``.env`` file without overriding process variables."""

    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# Kept as a short alias for existing experiment entry points.
env = load_local_env
