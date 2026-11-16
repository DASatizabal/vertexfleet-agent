"""Plain logging plus an append-only JSONL audit log."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from settings import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def audit(event_type: str, **payload) -> dict:
    record = {"ts": datetime.now(timezone.utc).isoformat(), "event": event_type, **payload}
    path = load_config()["paths"]["audit_log"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return record
