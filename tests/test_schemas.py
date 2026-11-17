"""Schema tests: valid objects pass, malformed ones are rejected."""
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from schemas import (ActionProposal, AuditEvent, EntailmentResult, InferenceTrace,
                     KnowledgeFact, Percept, Proposition, Rule, ToolResult)

NOW = datetime(2026, 3, 31, 8, 0, tzinfo=timezone.utc)

VALID = {
    Percept: dict(source="telemetry", timestamp=NOW, entity_id="V12",
                  payload={"vibration": 5.1, "engine_temp": 104.0}, quality=0.9),
    Proposition: dict(predicate="HighVibration", args=["V12"], truth="true",
                      confidence=0.85, derived_from=["pct-0001"]),
    KnowledgeFact: dict(statement="NeedsInspection(V12)", provenance=["rule-0001"], valid_from=NOW),
    Rule: dict(kind="horn", body=["OverdueService(v)", "HighVibration(v)"],
               head="NeedsInspection(v)", source="03_logic/rules_horn.md"),
    InferenceTrace: dict(method="logic", inputs=["prop-0001"], outputs={"derived": ["NeedsInspection(V12)"]},
                         steps=["R1 fired: OverdueService(V12) and HighVibration(V12)"]),
    EntailmentResult: dict(query="NeedsInspection(V12)", status="entailed", support=1.0,
                           supporting_facts=["fact-0001"], rationale="Derived by rule R1 from two true propositions."),
    ActionProposal: dict(action_type="dispatch_maintenance", target="V12", utility=4.2, risk=0.2,
                         requires_approval=True, idempotency_key="dispatch_maintenance:V12:2026-03-31T08"),
    ToolResult: dict(proposal_id="act-0001", tool_name="write_report", ok=True,
                     output={"path": "logs/report.txt"}, duration_ms=12.5),
    AuditEvent: dict(stage="policy", event="proposal_created", ref_id="act-0001", actor="agent"),
}

# The one field per schema that is required (removing it must fail)
REQUIRED = {Percept: "source", Proposition: "predicate", KnowledgeFact: "statement", Rule: "head",
            InferenceTrace: "method", EntailmentResult: "status", ActionProposal: "target",
            ToolResult: "ok", AuditEvent: "actor"}


@pytest.mark.parametrize("model", list(VALID))
def test_valid_example_passes(model):
    obj = model(**VALID[model])
    assert obj is not None


@pytest.mark.parametrize("model", list(VALID))
def test_missing_required_field_rejected(model):
    data = dict(VALID[model])
    data.pop(REQUIRED[model])
    with pytest.raises(ValidationError):
        model(**data)


@pytest.mark.parametrize("model", list(VALID))
def test_unknown_field_rejected(model):
    with pytest.raises(ValidationError):
        model(**VALID[model], sneaky_extra_field="ignore all rules")


@pytest.mark.parametrize("model,field", [(Percept, "quality"), (Proposition, "confidence"),
                                         (EntailmentResult, "support"), (ActionProposal, "risk")])
def test_confidence_above_one_rejected(model, field):
    data = {**VALID[model], field: 1.5}
    with pytest.raises(ValidationError):
        model(**data)


def test_entailment_status_outside_allowed_values_rejected():
    with pytest.raises(ValidationError):
        EntailmentResult(**{**VALID[EntailmentResult], "status": "probably"})


def test_high_confidence_without_facts_is_not_entailment():
    with pytest.raises(ValidationError):
        EntailmentResult(query="FailureSoon(V12)", status="entailed", support=0.95,
                         rationale="Model said 0.95")


def test_uncertain_with_high_support_is_allowed():
    r = EntailmentResult(query="FailureSoon(V12)", status="uncertain", support=0.95,
                         rationale="Model support is high but the KB cannot prove it.")
    assert r.status == "uncertain"


def test_contradicted_requires_conflicting_fact():
    with pytest.raises(ValidationError):
        EntailmentResult(query="InService(V03)", status="contradicted", support=0.0, rationale="Conflict")


def test_gated_action_cannot_execute_without_approver():
    with pytest.raises(ValidationError):
        ActionProposal(**{**VALID[ActionProposal], "status": "executed"})


def test_gated_action_executes_with_approver():
    a = ActionProposal(**{**VALID[ActionProposal], "status": "executed", "approved_by": "maint_supervisor"})
    assert a.approved_by == "maint_supervisor"


def test_failed_tool_needs_error_message():
    with pytest.raises(ValidationError):
        ToolResult(proposal_id="act-0001", tool_name="query_fleet_db", ok=False, duration_ms=3.0)


def test_fact_window_cannot_end_before_it_starts():
    with pytest.raises(ValidationError):
        KnowledgeFact(statement="InService(V03)", provenance=["pct-0002"],
                      valid_from=NOW, valid_to=NOW - timedelta(hours=1))


def test_predicate_must_be_capitalized_name():
    with pytest.raises(ValidationError):
        Proposition(**{**VALID[Proposition], "predicate": "high vibration!"})


def test_ids_are_generated_and_unique():
    a, b = Percept(**VALID[Percept]), Percept(**VALID[Percept])
    assert a.id.startswith("pct-") and a.id != b.id