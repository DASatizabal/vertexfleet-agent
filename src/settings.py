"""Loads configs/config.yaml and environment secrets. Single source of settings."""
from __future__ import annotations

import os
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
load_dotenv(PROJECT_ROOT / ".env")  # local secrets file, never committed


@lru_cache(maxsize=1)
def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: PROJECT_ROOT / v for k, v in cfg["paths"].items()}
    return cfg


def secret(name: str, default: str | None = None) -> str | None:
    """Secrets come only from environment variables."""
    return os.environ.get(name, default)


def kill_switch_on() -> bool:
    return str(secret("VF_KILL_SWITCH", "false")).lower() == "true"


def set_seed(seed: int | None = None) -> int:
    seed = seed if seed is not None else load_config()["project"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    return seed
