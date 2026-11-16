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
