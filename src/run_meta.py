"""Run receipts (Phase 2, Step 4): record exactly what produced a result."""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from settings import CONFIG_PATH, PROJECT_ROOT, load_config

PACKAGES = ["numpy", "pandas", "scikit-learn", "pydantic"]
WINDOWS_GIT = r"C:\Program Files\Git\cmd\git.exe"


def _git(*args: str) -> str:
    git = shutil.which("git") or WINDOWS_GIT
    try:
        out = subprocess.run([git, *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def write_run_meta(name: str, out_dir: Path | None = None, extra: dict | None = None) -> Path:
    """Write <name>_run_meta.json and return its path."""
    cfg = load_config()
    out_dir = Path(out_dir) if out_dir else cfg["paths"]["artifacts"]
    out_dir.mkdir(parents=True, exist_ok=True)
    status = _git("status", "--porcelain")
    meta = {
        "name": name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": cfg["project"]["seed"],
        "project_version": cfg["project"]["version"],
        "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_uncommitted_changes": status not in ("", "unknown"),
        "python_version": platform.python_version(),
        "packages": {p: _pkg(p) for p in PACKAGES},
    }
    if extra:
        meta["extra"] = extra
    path = out_dir / f"{name}_run_meta.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path