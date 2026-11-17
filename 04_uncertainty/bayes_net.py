"""04_uncertainty/bayes_net.py: Bayesian network for 7-day failure risk.

Structure (arrows point from cause to effect):

    Rain ----\\
    OldVehicle -> Wear -> Vibration
    Overdue -/       |--> EngineTemp
                     +--> FailureSoon

Wear is the vehicle's health. It is learned from historical inspection labels
(the simulator's hidden state) and is NOT observed when the agent runs: the agent
sees only Vibration, EngineTemp, Rain, OldVehicle and Overdue, and the network
infers Wear and FailureSoon from that evidence.

Run directly:   python 04_uncertainty/bayes_net.py
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", message=".*pgmpy.*", category=FutureWarning)
from pgmpy.estimators import BayesianEstimator  # noqa: E402
from pgmpy.inference import VariableElimination  # noqa: E402
from pgmpy.models import DiscreteBayesianNetwork  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from settings import load_config  # noqa: E402

EDGES = [("Rain", "Wear"), ("OldVehicle", "Wear"), ("Overdue", "Wear"),
         ("Wear", "Vibration"), ("Wear", "EngineTemp"), ("Wear", "FailureSoon")]
EVIDENCE_VARS = ["Rain", "OldVehicle", "Overdue", "Vibration", "EngineTemp"]
STATE_NAMES = {
    "Rain": ["no", "yes"], "OldVehicle": ["no", "yes"], "Overdue": ["no", "yes"],
    "Wear": ["Healthy", "Worn", "Critical"],
    "Vibration": ["low", "medium", "high"], "EngineTemp": ["normal", "warm", "hot"],
    "FailureSoon": ["no", "yes"],
}


def discretize(telemetry: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Turn continuous telemetry into the discrete states the network uses."""
    rules = cfg["logic_rules"]
    return pd.DataFrame({
        "Rain": np.where(telemetry.rain == 1, "yes", "no"),
        "OldVehicle": np.where(telemetry.age_years >= 5, "yes", "no"),
        "Overdue": np.where(telemetry.days_since_service > rules["overdue_days"], "yes", "no"),
        "Wear": telemetry.true_state.replace({"Failed": "Critical"}),
        "Vibration": pd.cut(telemetry.vibration, [-np.inf, 3.0, rules["vibration_high"], np.inf],
                            labels=["low", "medium", "high"]).astype(str),
        "EngineTemp": pd.cut(telemetry.engine_temp, [-np.inf, 92.0, rules["engine_hot"], np.inf],
                             labels=["normal", "warm", "hot"]).astype(str),
        "FailureSoon": np.where(telemetry.fails_within_7d == 1, "yes", "no"),
    })


def fit_network(train: pd.DataFrame) -> DiscreteBayesianNetwork:
    """Learn the conditional probability tables. The BDeu prior avoids zero probabilities."""
    model = DiscreteBayesianNetwork(EDGES)
    estimator = BayesianEstimator(model, train, state_names=STATE_NAMES)
    model.add_cpds(*estimator.get_parameters(prior_type="BDeu", equivalent_sample_size=10))
    model.check_model()
    return model


def p_failure(infer: VariableElimination, evidence: dict) -> float:
    """P(FailureSoon = yes | evidence)."""
    q = infer.query(["FailureSoon"], evidence=evidence, show_progress=False)
    return float(q.values[STATE_NAMES["FailureSoon"].index("yes")])


def p_wear(infer: VariableElimination, evidence: dict) -> dict:
    q = infer.query(["Wear"], evidence=evidence, show_progress=False)
    return {s: round(float(v), 3) for s, v in zip(STATE_NAMES["Wear"], q.values)}


def load_split(cfg: dict):
    path = cfg["paths"]["data_synthetic"] / "telemetry.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 02_simulation.ipynb first.")
    tele = pd.read_csv(path, parse_dates=["date"])
    split = tele.date.min() + pd.Timedelta(days=cfg["sim_details"]["train_days"])
    return tele[tele.date < split], tele[tele.date >= split]


def main() -> dict:
    cfg = load_config()
    train_raw, test_raw = load_split(cfg)
    train, test = discretize(train_raw, cfg), discretize(test_raw, cfg)
    model = fit_network(train)
    infer = VariableElimination(model)

    print("Network edges:", EDGES)
    print("\nCPT for FailureSoon given Wear:")
    print(model.get_cpds("FailureSoon"))

    # Inference under evidence: the questions the agent will ask
    queries = {
        "no evidence": {},
        "high vibration": {"Vibration": "high"},
        "high vibration + rain": {"Vibration": "high", "Rain": "yes"},
        "high vibration + hot engine": {"Vibration": "high", "EngineTemp": "hot"},
        "low vibration + normal temp": {"Vibration": "low", "EngineTemp": "normal"},
        "old, overdue, medium vibration": {"OldVehicle": "yes", "Overdue": "yes", "Vibration": "medium"},
    }
    print("\nP(FailureSoon = yes | evidence) and the inferred hidden Wear state:")
    query_results = {}
    for name, ev in queries.items():
        pf, pw = p_failure(infer, ev), p_wear(infer, ev)
        query_results[name] = {"p_failure": round(pf, 3), "p_wear": pw}
        print(f"  {name:32s} P(fail)={pf:.3f}   Wear={pw}")

    # Evaluate on the held-out days using only observable evidence
    cache = {}
    scores = []
    for row in test[EVIDENCE_VARS].itertuples(index=False):
        key = tuple(row)
        if key not in cache:
            cache[key] = p_failure(infer, dict(zip(EVIDENCE_VARS, key)))
        scores.append(cache[key])
    y = (test.FailureSoon == "yes").astype(int)
    auc = roc_auc_score(y, scores)
    print(f"\nHeld-out ROC AUC using only observable evidence: {auc:.3f}")
    print("(Phase 1 naive Bayes baseline: 0.847)")

    results = {"edges": EDGES, "queries": query_results, "test_roc_auc": round(auc, 3)}
    out = cfg["paths"]["artifacts"] / "bayes_net_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved", out)
    return results


if __name__ == "__main__":
    main()
