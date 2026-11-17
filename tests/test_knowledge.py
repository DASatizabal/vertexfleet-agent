"""Knowledge base tests: chaining, provenance, contradictions and the entailment cases."""
import importlib.util
from pathlib import Path

import pytest

from agent.knowledge import Atom, KnowledgeBase
from schemas import Rule

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("entailment_cases", ROOT / "03_logic" / "entailment_cases.py")
cases_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cases_module)


@pytest.mark.parametrize("case", cases_module.CASES, ids=lambda c: c["name"])
def test_entailment_case(case):
    result, _ = cases_module.run_case(case)
    assert result.status == case["expected"]


def test_at_least_five_cases():
    assert len(cases_module.CASES) >= 5


def test_derived_fact_records_rule_and_sources():
    kb = KnowledgeBase()
    a = kb.tell("OverdueService(V12)")
    b = kb.tell("HighVibration(V12)")
    kb.forward_chain()
    fact = kb.facts[Atom.parse("NeedsInspection(V12)")]
    assert fact.provenance[0] == "R1"
    assert {a.id, b.id} <= set(fact.provenance)
    assert any(line.startswith("R1 fired") for line in kb.trace)


def test_variables_bind_consistently():
    kb = KnowledgeBase()
    kb.tell("OverdueService(V01)")
    kb.tell("HighVibration(V02)")      # different vehicles: R1 must NOT fire
    kb.forward_chain()
    assert Atom.parse("NeedsInspection(V01)") not in kb.facts
    assert Atom.parse("NeedsInspection(V02)") not in kb.facts


def test_contradiction_is_detected_not_overwritten():
    kb = KnowledgeBase()
    kb.tell("InService(V03)")
    kb.tell("AvailableForDispatch(V03)")
    kb.forward_chain()
    pairs = kb.contradictions()
    assert len(pairs) == 1
    assert Atom.parse("AvailableForDispatch(V03)") in kb.facts   # the percept is kept, not deleted


def test_model_confidence_never_becomes_entailment():
    kb = KnowledgeBase()
    result = kb.ask("FailureSoon(V09)", model_support=0.99)
    assert result.status == "uncertain"


def test_added_rule_is_used():
    kb = KnowledgeBase()
    kb.add_rule(Rule(id="R99", kind="horn", body=["LowBattery(v)"], head="NeedsCharge(v)", source="test"))
    kb.tell("LowBattery(V20)")
    assert kb.ask("NeedsCharge(V20)").status == "entailed"


def test_bad_atoms_are_rejected():
    kb = KnowledgeBase()
    with pytest.raises(ValueError):
        kb.tell("high vibration on V12")
    with pytest.raises(ValueError):
        kb.tell("HighVibration(v)")      # facts cannot contain variables
