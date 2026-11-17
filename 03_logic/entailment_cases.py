"""03_logic/entailment_cases.py: entailment test cases for the knowledge base.

Run directly to print a results table:   python 03_logic/entailment_cases.py
The same cases run automatically in tests/test_knowledge.py.
"""
from agent.knowledge import KnowledgeBase

CASES = [
    {"name": "1. Entailed by chaining",
     "facts": ["OverdueService(V12)", "HighVibration(V12)"],
     "query": "NeedsInspection(V12)", "model_support": None, "expected": "entailed"},
    {"name": "2. Contradicted by a constraint",
     "facts": ["InService(V03)"],
     "query": "AvailableForDispatch(V03)", "model_support": None, "expected": "contradicted"},
    {"name": "3. Unknown: not enough evidence",
     "facts": ["HighVibration(V07)"],
     "query": "NeedsInspection(V07)", "model_support": None, "expected": "unknown"},
    {"name": "4. Model says 0.92, KB cannot prove it",
     "facts": ["HighVibration(V07)"],
     "query": "FailureSoon(V07)", "model_support": 0.92, "expected": "uncertain"},
    {"name": "5. New percept conflicts with the KB",
     "facts": ["InService(V03)", "AvailableForDispatch(V03)"],
     "query": "AvailableForDispatch(V03)", "model_support": None, "expected": "contradicted"},
    {"name": "6. Two-step chain with a binary relation",
     "facts": ["HighDemand(Z3)", "RainForecast(Z3)", "Adjacent(Z3, Z5)", "LowDemand(Z5)"],
     "query": "RepositionCandidate(Z5, Z3)", "model_support": None, "expected": "entailed"},
]


def run_case(case: dict):
    kb = KnowledgeBase()
    for f in case["facts"]:
        kb.tell(f, provenance=["case"])
    return kb.ask(case["query"], model_support=case["model_support"]), kb


if __name__ == "__main__":
    for case in CASES:
        result, kb = run_case(case)
        mark = "PASS" if result.status == case["expected"] else "FAIL"
        print(f"[{mark}] {case['name']}: {result.status} (expected {case['expected']})")
        print(f"       {result.rationale}")
        for line in kb.trace:
            print(f"       trace: {line}")
