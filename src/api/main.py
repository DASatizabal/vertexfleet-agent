"""FastAPI entry point. Run with: uvicorn api.main:app --app-dir src"""
from fastapi import FastAPI

from settings import kill_switch_on, load_config

cfg = load_config()
app = FastAPI(title="VertexFleet Agent", version=cfg["project"]["version"])


@app.get("/health")
def health():
    return {"status": "ok", "version": cfg["project"]["version"]}


@app.get("/status")
def status():
    return {"kill_switch": kill_switch_on(), "max_iterations": cfg["agent"]["max_iterations"]}
