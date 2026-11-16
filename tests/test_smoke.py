"""Smoke tests: the skeleton is wired correctly."""
import importlib

from fastapi.testclient import TestClient

from settings import load_config, set_seed

MODULES = ["perceive", "proposition", "knowledge", "inference",
           "entailment", "policy", "actions", "orchestrator"]


def test_config_loads_with_seed():
    cfg = load_config()
    assert cfg["project"]["seed"] == 42
    assert cfg["paths"]["audit_log"].name == "audit.jsonl"


def test_seed_is_reproducible():
    import numpy as np
    set_seed(42); a = np.random.rand(3)
    set_seed(42); b = np.random.rand(3)
    assert (a == b).all()


def test_all_agent_modules_import():
    for m in MODULES:
        importlib.import_module(f"agent.{m}")


def test_health_endpoint():
    from api.main import app
    r = TestClient(app).get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_kill_switch_halts_orchestrator(monkeypatch):
    monkeypatch.setenv("VF_KILL_SWITCH", "true")
    from agent.orchestrator import run_cycle
    assert run_cycle({})["status"] == "halted"
