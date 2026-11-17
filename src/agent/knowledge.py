"""Stage 3: KNOWLEDGE BASE.

Stores facts with provenance, applies Horn rules by forward chaining (with variables),
detects contradictions, and answers "is this entailed?" questions.

Notation: atoms look like  HighVibration(V12)  or  Adjacent(Z3, Z5).
A leading "not " marks a negated atom:  not AvailableForDispatch(V03).
In rules, lowercase arguments (v, z, y) are variables; anything else is a constant.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import count

from schemas import EntailmentResult, KnowledgeFact, Rule
from settings import load_config

ATOM = re.compile(r"^(not\s+)?([A-Z][A-Za-z]*)\(([^()]*)\)$")


@dataclass(frozen=True)
class Atom:
    negated: bool
    predicate: str
    args: tuple[str, ...]

    @classmethod
    def parse(cls, text: str) -> "Atom":
        m = ATOM.match(text.strip())
        if not m:
            raise ValueError(f"Not a valid atom: {text!r}")
        args = tuple(a.strip() for a in m.group(3).split(",") if a.strip())
        return cls(bool(m.group(1)), m.group(2), args)

    def negate(self) -> "Atom":
        return Atom(not self.negated, self.predicate, self.args)

    def substitute(self, binding: dict[str, str]) -> "Atom":
        return Atom(self.negated, self.predicate, tuple(binding.get(a, a) for a in self.args))

    def __str__(self) -> str:
        return f"{'not ' if self.negated else ''}{self.predicate}({', '.join(self.args)})"


def is_variable(term: str) -> bool:
    return term[:1].islower()


def default_rules() -> list[Rule]:
    """The rules documented in 03_logic/rules_horn.md (ids must match)."""
    spec = [
        ("R1", "horn", ["OverdueService(v)", "HighVibration(v)"], "NeedsInspection(v)"),
        ("R2", "horn", ["EngineHot(v)"], "NeedsInspection(v)"),
        ("R3", "horn", ["NeedsInspection(v)", "LowBattery(v)"], "HighPriority(v)"),
        ("R4", "horn", ["HighDemand(z)", "RainForecast(z)"], "DemandSurge(z)"),
        ("R5", "horn", ["DemandSurge(z)", "Adjacent(z, y)", "LowDemand(y)"], "RepositionCandidate(y, z)"),
        ("R6", "horn", ["NeedsInspection(v)"], "not AvailableForDispatch(v)"),
        ("R7", "constraint", ["InService(v)"], "not AvailableForDispatch(v)"),
    ]
    return [Rule(id=i, kind=k, body=b, head=h, source="03_logic/rules_horn.md") for i, k, b, h in spec]


class KnowledgeBase:
    def __init__(self, rules: list[Rule] | None = None):
        self.facts: dict[Atom, KnowledgeFact] = {}
        self.rules: list[Rule] = list(rules) if rules is not None else default_rules()
        self.trace: list[str] = []          # short evidence records: which rule fired and why
        self._ids = count(1)

    # ---- adding knowledge ----
    def tell(self, statement: str, provenance: list[str] | None = None) -> KnowledgeFact:
        """Add a fact. Never overwrites: a conflicting fact is kept and shows up in contradictions()."""
        atom = Atom.parse(statement)
        if any(is_variable(a) for a in atom.args):
            raise ValueError(f"Facts must be ground (no variables): {statement}")
        if atom in self.facts:
            return self.facts[atom]
        fact = KnowledgeFact(id=f"fact-{next(self._ids):04d}", statement=str(atom),
                             provenance=provenance or ["told"], valid_from=datetime.now(timezone.utc))
        self.facts[atom] = fact
        return fact

    def add_rule(self, rule: Rule) -> None:
        for text in [*rule.body, rule.head]:
            Atom.parse(text)                 # fail early on a malformed rule
        self.rules.append(rule)

    # ---- reasoning ----
    def _match(self, pattern: Atom, binding: dict[str, str]):
        """Yield every extended binding under which pattern matches a known fact."""
        for fact in list(self.facts):
            if fact.negated != pattern.negated or fact.predicate != pattern.predicate \
                    or len(fact.args) != len(pattern.args):
                continue
            new = dict(binding)
            for p, f in zip(pattern.args, fact.args):
                if is_variable(p):
                    if new.setdefault(p, f) != f:
                        break
                elif p != f:
                    break
            else:
                yield new, fact

    def _satisfy(self, body: list[Atom], binding: dict[str, str], used: list[Atom]):
        if not body:
            yield binding, used
            return
        for new, fact in self._match(body[0].substitute(binding), binding):
            yield from self._satisfy(body[1:], new, used + [fact])

    def forward_chain(self, max_rounds: int = 50) -> list[KnowledgeFact]:
        """Apply every rule until nothing new can be derived. Returns the new facts."""
        derived: list[KnowledgeFact] = []
        for _ in range(max_rounds):
            new_this_round = []
            for rule in self.rules:
                body = [Atom.parse(b) for b in rule.body]
                head = Atom.parse(rule.head)
                for binding, used in list(self._satisfy(body, {}, [])):
                    conclusion = head.substitute(binding)
                    if conclusion in self.facts:
                        continue
                    support_ids = [self.facts[u].id for u in used]
                    fact = self.tell(str(conclusion), provenance=[rule.id, *support_ids])
                    self.trace.append(f"{rule.id} fired: {' and '.join(map(str, used))} => {conclusion}")
                    new_this_round.append(fact)
            if not new_this_round:
                return derived
            derived.extend(new_this_round)
        return derived

    def contradictions(self) -> list[tuple[KnowledgeFact, KnowledgeFact]]:
        """Pairs where both X and not X are known."""
        return [(self.facts[a], self.facts[a.negate()])
                for a in self.facts if not a.negated and a.negate() in self.facts]

    def support_chain(self, atom: Atom) -> list[str]:
        """Fact ids behind a conclusion, following provenance back to told facts."""
        chain, todo = [], [self.facts[atom].id]
        by_id = {f.id: f for f in self.facts.values()}
        while todo:
            fid = todo.pop()
            if fid in chain or fid not in by_id:
                continue
            chain.append(fid)
            todo.extend(by_id[fid].provenance)
        return chain

    def ask(self, query: str, model_support: float | None = None) -> EntailmentResult:
        """Is the query entailed by the KB? Model support can make it 'uncertain', never 'entailed'."""
        self.forward_chain()
        q = Atom.parse(query)
        threshold = load_config()["thresholds"]["entailment_support_min"]
        has_q, has_not_q = q in self.facts, q.negate() in self.facts
        if has_q and has_not_q:
            return EntailmentResult(query=query, status="contradicted", support=0.0,
                                    conflicting_facts=[self.facts[q].id, self.facts[q.negate()].id],
                                    rationale=f"The KB holds both {q} and {q.negate()}; flagged for review.")
        if has_q:
            return EntailmentResult(query=query, status="entailed", support=1.0,
                                    supporting_facts=self.support_chain(q),
                                    rationale=f"{q} follows from known facts and rules.")
        if has_not_q:
            return EntailmentResult(query=query, status="contradicted", support=0.0,
                                    conflicting_facts=self.support_chain(q.negate()),
                                    rationale=f"The KB proves {q.negate()}.")
        if model_support is not None:
            level = "meets" if model_support >= threshold else "is below"
            return EntailmentResult(query=query, status="uncertain", support=model_support,
                                    rationale=f"Not provable from the KB. Model support {model_support:.2f} "
                                              f"{level} the {threshold} threshold, but a model score is not proof.")
        return EntailmentResult(query=query, status="unknown", support=0.0,
                                rationale="No facts, rules or model evidence about this query.")
