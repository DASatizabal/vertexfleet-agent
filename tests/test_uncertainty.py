"""Uncertainty model tests: Bayesian network inference and HMM state recovery."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pgmpy")
pytest.importorskip("hmmlearn")
pytestmark = [pytest.mark.filterwarnings("ignore::FutureWarning"),
              pytest.mark.filterwarnings("ignore::DeprecationWarning")]  # library notices, not our code

from settings import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config()
TELEMETRY = CFG["paths"]["data_synthetic"] / "telemetry.csv"
needs_data = pytest.mark.skipif(not TELEMETRY.exists(), reason="run 02_simulation.ipynb first")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "04_uncertainty" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bn = _load("bayes_net")
hm = _load("temporal_hmm")


@pytest.fixture(scope="module")
def infer():
    from pgmpy.inference import VariableElimination
    train, _ = bn.load_split(CFG)
    return VariableElimination(bn.fit_network(bn.discretize(train, CFG)))


@needs_data
def test_probabilities_are_valid(infer):
    p = bn.p_failure(infer, {"Vibration": "high"})
    assert 0.0 <= p <= 1.0
    assert abs(sum(bn.p_wear(infer, {}).values()) - 1.0) < 1e-6


@needs_data
def test_evidence_moves_belief_in_the_right_direction(infer):
    low = bn.p_failure(infer, {"Vibration": "low", "EngineTemp": "normal"})
    high = bn.p_failure(infer, {"Vibration": "high", "EngineTemp": "hot"})
    assert high > low + 0.3


@needs_data
def test_inference_works_with_missing_evidence(infer):
    # The agent often lacks some sensors; the network must still answer.
    assert 0.0 <= bn.p_failure(infer, {"Rain": "yes"}) <= 1.0


@needs_data
def test_hmm_beats_memoryless_threshold_rule():
    tele = pd.read_csv(TELEMETRY, parse_dates=["date"])
    result = hm.part_a(CFG, tele)
    assert result["hmm_accuracy"] > result["threshold_accuracy"]


def test_hmm_components_named_by_vibration():
    class Fake:
        means_ = np.array([[5.0, 100.0], [2.0, 88.0], [3.2, 93.0]])
    assert hm.label_components(Fake()) == {1: "Healthy", 2: "Worn", 0: "Critical"}
