"""Typed contracts passed between the six agent stages (Phase 2, Step 3).

Every model forbids unknown fields, and every confidence is a number from 0 to 1.
A malformed object is rejected here, at the dock, instead of breaking a later stage.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Confidence = Annotated[float, Field(ge=0, le=1)]
ShortText = Annotated[str, StringConstraints(min_length=1, max_length=300)]


def _id(prefix: str):
    return lambda: f"{prefix}-{uuid4().hex[:8]}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# 1. PERCEPTS
class Percept(Strict):
    id: str = Field(default_factory=_id("pct"))
    source: Literal["trips", "telemetry", "service", "weather", "tool"]
    timestamp: datetime
    entity_id: str = Field(min_length=1)          # vehicle, zone or tool-call id
    payload: dict[str, Any]
    quality: Confidence
    validation_status: Literal["valid", "suspect", "invalid"] = "valid"


# 2. PROPOSITIONS
class Proposition(Strict):
    id: str = Field(default_factory=_id("prop"))
    predicate: str = Field(pattern=r"^[A-Z][A-Za-z]*$")   # e.g. HighVibration
    args: list[str]                                        # e.g. ["V12"]
    truth: Literal["true", "false", "unknown"]
    confidence: Confidence
    derived_from: list[str] = Field(min_length=1)          # percept ids
    features: dict[str, float] = Field(default_factory=dict)


# 3. KNOWLEDGE BASE
class KnowledgeFact(Strict):
    id: str = Field(default_factory=_id("fact"))
    statement: str = Field(min_length=1)                   # e.g. NeedsInspection(V12)
    provenance: list[str] = Field(min_length=1)            # ids of propositions, rules or models
    valid_from: datetime
    valid_to: datetime | None = None

    @model_validator(mode="after")
    def _window(self):
        if self.valid_to is not None and self.valid_to < self.valid_from:
            raise ValueError("valid_to cannot be earlier than valid_from")
        return self


class Rule(Strict):
    id: str = Field(default_factory=_id("rule"))
    kind: Literal["horn", "constraint"]
    body: list[str] = Field(min_length=1)                  # e.g. ["OverdueService(v)", "HighVibration(v)"]
    head: str = Field(min_length=1)                        # e.g. NeedsInspection(v)
    source: str = Field(min_length=1)                      # where the rule came from


# 4. INFERENCE
class InferenceTrace(Strict):
    id: str = Field(default_factory=_id("inf"))
    method: Literal["logic", "csp", "bayes", "hmm", "ml", "search"]
    inputs: list[str]                                      # ids of facts / propositions used
    outputs: dict[str, Any]
    model_version: str | None = None
    steps: list[ShortText] = Field(default_factory=list)   # short evidence records, not chain-of-thought


# 5. ENTAILMENT
class EntailmentResult(Strict):
    query: str = Field(min_length=1)
    status: Literal["entailed", "contradicted", "uncertain", "unknown"]
    support: Confidence
    supporting_facts: list[str] = Field(default_factory=list)
    conflicting_facts: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def _status_matches_evidence(self):
        # Model confidence alone is never proof: "entailed" needs facts behind it.
        if self.status == "entailed" and (not self.supporting_facts or self.conflicting_facts):
            raise ValueError("entailed requires supporting facts and no conflicting facts")
        if self.status == "contradicted" and not self.conflicting_facts:
            raise ValueError("contradicted requires at least one conflicting fact")
        return self


# 6. ACTION
ActionType = Literal["query_fleet_db", "lookup_service_history", "write_report",
                     "propose_action", "dispatch_maintenance", "reposition_fleet"]


class ActionProposal(Strict):
    id: str = Field(default_factory=_id("act"))
    action_type: ActionType
    target: str = Field(min_length=1)                      # vehicle or zone id
    utility: float
    risk: Confidence
    requires_approval: bool
    approved_by: str | None = None
    status: Literal["proposed", "approved", "rejected", "executed", "blocked"] = "proposed"
    idempotency_key: str = Field(min_length=8)

    @model_validator(mode="after")
    def _approval_gate(self):
        if self.requires_approval and self.status in ("approved", "executed") and not self.approved_by:
            raise ValueError("an approval-gated action cannot be approved or executed without approved_by")
        return self


class ToolResult(Strict):
    proposal_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    ok: bool
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    duration_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def _error_when_failed(self):
        if not self.ok and not self.error:
            raise ValueError("a failed tool call must include an error message")
        return self


# AUDIT TRAIL
class AuditEvent(Strict):
    ts: datetime = Field(default_factory=_now)
    stage: Literal["perceive", "proposition", "knowledge", "inference",
                   "entailment", "policy", "action", "system"]
    event: str = Field(min_length=1)
    ref_id: str | None = None
    actor: Literal["agent", "human", "system"]
    details: dict[str, Any] = Field(default_factory=dict)