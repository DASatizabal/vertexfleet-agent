"""VertexFleet bootstrap: run once from the project folder with  python bootstrap.py

Creates the folder tree and starter files. Safe to re-run: existing files are never overwritten.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
print("Project folder:", ROOT)

FOLDERS = [
    "src/agent", "src/api", "src/ui",
    "tests",
    "notebooks",
    "data/raw", "data/processed", "data/synthetic",
    "configs",
    "docs",
    "03_logic", "04_uncertainty", "05_ml/artifacts",
    "logs",
    ".github/workflows",
]
for f in FOLDERS:
    (ROOT / f).mkdir(parents=True, exist_ok=True)

# .gitkeep lets Git track otherwise-empty folders
for f in ["data/raw", "data/processed", "data/synthetic", "logs", "05_ml/artifacts"]:
    (ROOT / f / ".gitkeep").touch(exist_ok=True)

print("Folders ready")

from textwrap import dedent

def write(rel_path: str, text: str) -> None:
    p = ROOT / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        print(f"skipped (exists): {rel_path}")
        return
    p.write_text(dedent(text).lstrip("\n"), encoding="utf-8")
    print(f"created:          {rel_path}")

write("requirements.txt", """
    fastapi>=0.110
    uvicorn>=0.29
    pydantic>=2.6
    pyyaml>=6.0
    numpy>=1.26
    pandas>=2.1
    scikit-learn>=1.4
    joblib>=1.3
    httpx>=0.27
    python-dotenv>=1.0
    pytest>=8.0
""")

write("requirements-dev.txt", """
    -r requirements.txt
    gradio>=4.0
    matplotlib>=3.8
    seaborn>=0.13
    pgmpy>=0.1.25
    hmmlearn>=0.3
    python-constraint>=1.4
    shap>=0.45
    torch>=2.2
    transformers>=4.40
    mlflow>=2.12
""")

write("configs/config.yaml", """
    project:
      name: vertexfleet-agent
      version: 0.1.0
      seed: 42

    paths:            # relative to project root, resolved with pathlib
      data_raw: data/raw
      data_processed: data/processed
      data_synthetic: data/synthetic
      artifacts: 05_ml/artifacts
      audit_log: logs/audit.jsonl

    simulation:
      n_vehicles: 50
      n_zones: 8
      n_days: 90

    thresholds:
      percept_min_quality: 0.6        # below this a percept is marked low-quality
      entailment_support_min: 0.8     # probabilistic support needed to call something "supported"
      abstain_below_confidence: 0.55  # agent abstains / escalates below this
      failure_risk_high: 0.7          # predicted failure prob that triggers a maintenance proposal

    agent:
      max_iterations: 5
      require_approval_for: [dispatch_maintenance, reposition_fleet]
      allowed_tools: [query_fleet_db, lookup_service_history, write_report, propose_action]

    models:
      demand_forecaster: {name: baseline_mean, version: "0.0.1"}
      failure_classifier: {name: baseline_logistic, version: "0.0.1"}
      maintenance_note_nlp: {name: tfidf_baseline, version: "0.0.1"}

    features:
      use_llm_component: false
      enable_ui: true
""")

write(".env.example", """
    # Copy to .env (never commit .env) and fill in values.
    VF_ADMIN_TOKEN=
    VF_KILL_SWITCH=false
    # Only needed if you use a hosted LLM later:
    VF_LLM_API_KEY=
""")

write(".gitignore", """
    .env
    .venv/
    __pycache__/
    *.pyc
    .pytest_cache/
    .ipynb_checkpoints/
    logs/*.jsonl
    data/raw/*
    !data/raw/.gitkeep
    mlruns/
""")

write("pytest.ini", """
    [pytest]
    pythonpath = src
    testpaths = tests
""")

write("src/settings.py", '''
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
''')

write("src/logging_setup.py", '''
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
            f.write(json.dumps(record, default=str) + "\\n")
        return record
''')

STAGES = {
    "perceive":    ("PERCEPTS",       "Capture raw observations with source, timestamp, quality and validation status.", "Percept"),
    "proposition": ("PROPOSITIONS",   "Turn percepts into testable symbolic statements and model features; flag contradictions.", "Proposition"),
    "knowledge":   ("KNOWLEDGE BASE", "Store facts, rules, constraints, ontology relations, model metadata and provenance.", "KnowledgeFact / Rule"),
    "inference":   ("INFERENCE",      "Apply logic, search/CSP, Bayesian, temporal and learned models; emit an inspectable trace.", "InferenceTrace"),
    "entailment":  ("ENTAILMENT",     "Decide entailed / contradicted / uncertain / unknown. Model confidence is NOT proof.", "EntailmentResult"),
    "policy":      ("POLICY GATE",    "Check goals, hard constraints, utility/risk and approval rules before any action.", "ActionProposal"),
    "actions":     ("ACTION",         "Run only allow-listed tools; log every ToolResult; respect the kill switch.", "ToolResult"),
}

for module, (stage, purpose, schema) in STAGES.items():
    write(f"src/agent/{module}.py", f'''
        """Stage: {stage}

        Purpose: {purpose}
        Main output schema: {schema} (defined in Phase 2, Step 3)
        """


        def run(*args, **kwargs):
            raise NotImplementedError("{module}.run is filled in during Phase 4")
    ''')

write("src/agent/orchestrator.py", '''
    """Runs the loop: observe > update KB > reason > entail > plan > approve > act > observe result.

    Bounded by config agent.max_iterations and stopped immediately by the kill switch.
    """
    from settings import kill_switch_on, load_config


    def run_cycle(raw_inputs):
        if kill_switch_on():
            return {"status": "halted", "reason": "kill switch is on"}
        _max_iter = load_config()["agent"]["max_iterations"]
        raise NotImplementedError("orchestrator.run_cycle is filled in during Phase 4")
''')

write("src/agent/__init__.py", '"""Six-stage VertexFleet agent."""\n')
write("src/api/__init__.py", "\n")
write("src/ui/__init__.py", "\n")

write("src/api/main.py", '''
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
''')

write("Dockerfile", """
    FROM python:3.11-slim
    WORKDIR /app
    ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    COPY src ./src
    COPY configs ./configs
    COPY 05_ml/artifacts ./05_ml/artifacts
    RUN mkdir -p logs && useradd --create-home appuser && chown -R appuser /app
    USER appuser
    EXPOSE 8000
    HEALTHCHECK --interval=30s --timeout=5s --retries=3 \\
      CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
    CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
""")

write(".dockerignore", """
    .git
    .env
    data/raw
    notebooks
    *.ipynb
    logs/*.jsonl
""")

write(".github/workflows/ci.yml", """
    name: ci
    on: [push, pull_request]
    permissions:
      contents: read
    jobs:
      test:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v4
          - uses: actions/setup-python@v5
            with:
              python-version: "3.11"
          - run: pip install -r requirements.txt
          - run: pytest -q
      docker:
        needs: test
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v4
          - run: docker build -t vertexfleet-agent:ci .
          - run: |
              docker run -d -p 8000:8000 --name vf vertexfleet-agent:ci
              sleep 8
              curl --fail http://localhost:8000/health
""")

write("01_problem_and_PEAS.md", """
    # 01 Problem and PEAS: VertexFleet Autonomous Fleet Demand & Maintenance Agent

    ## Product and users
    ## Business / learning goal
    ## Stakeholders
    ## Goals
    ## Non-goals
    ## PEAS
    | Element | Description |
    |---|---|
    | Performance measure | |
    | Environment | |
    | Actuators | |
    | Sensors | |
    ## Environment assumptions (observable? deterministic? static? discrete?)
    ## Autonomy boundary (what the agent may do alone vs. needs approval)
    ## Six-stage agent cycle
    ## Risks
    ## Baseline plan
""")

write("03_logic/README.md", """
    # 03 Logic
    Planned files: propositions.md (vocabulary), rules_horn.md (CNF/Horn rules),
    ontology_fol.md (FOL predicates), entailment_cases.py (at least 5 cases).
""")
write("04_uncertainty/README.md", """
    # 04 Uncertainty
    Planned files: bayes_net.py (failure-risk Bayesian network), temporal_hmm.py (vehicle health states / demand forecasting).
""")
write("05_ml/README.md", """
    # 05 ML
    Planned files: preprocess.py, train.py, evaluate.py, model_card.md, artifacts/ (saved models).
""")
write("docs/README.md", """
    # docs
    Planned: runbook.md, monitoring_and_rollback.md, responsible_ai.md, demo_script.md.
""")

write("tests/test_smoke.py", '''
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
''')

write("README.md", """
    # VertexFleet Analytics: Autonomous Fleet Demand & Maintenance Intelligence Agent

    Course project (synthetic data only). Six-stage agent: PERCEPTS > PROPOSITIONS > KNOWLEDGE BASE >
    INFERENCE > ENTAILMENT > ACTION, with approval gates, audit log and kill switch.

    ## Layout
    | Path | What lives there |
    |---|---|
    | 01_problem_and_PEAS.md | Problem formulation and PEAS |
    | 02_simulation.ipynb | Synthetic environment and algorithm benchmarks |
    | 03_logic/ | Propositions, Horn/CNF rules, FOL ontology, entailment tests |
    | 04_uncertainty/ | Bayesian network and temporal models |
    | 05_ml/ | Preprocessing, training, evaluation, model card, artifacts |
    | src/agent/ | One module per reasoning stage plus orchestrator |
    | src/api/, src/ui/ | FastAPI service and Gradio UI |
    | configs/ | config.yaml (no secrets) |
    | tests/ | pytest suites |

    ## Run tests
        pip install -r requirements.txt
        pytest -q

    ## Run the API locally
        uvicorn api.main:app --app-dir src --port 8000
""")

write(".vscode/settings.json", """
    {
      "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe",
      "python.testing.pytestEnabled": true,
      "python.testing.pytestArgs": ["tests"],
      "python.analysis.extraPaths": ["${workspaceFolder}/src"],
      "jupyter.notebookFileRoot": "${workspaceFolder}",
      "terminal.integrated.env.windows": {"PYTHONPATH": "${workspaceFolder}/src"}
    }
""")

write(".env", """
    VF_ADMIN_TOKEN=change-me-to-a-long-random-string
    VF_KILL_SWITCH=false
    VF_LLM_API_KEY=
""")

print("\nBootstrap finished. Next: pip install -r requirements.txt, then pytest -q")

