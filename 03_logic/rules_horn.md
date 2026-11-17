# 03 Logic: Rules in Horn and CNF Form

Variables are lowercase (v = vehicle, z and y = zones). Each rule has an id that appears in fact provenance and in the inference trace, so every derived fact can be traced to the rule that produced it.
The same rules are loaded in code by `default_rules()` in `src/agent/knowledge.py`.

A Horn clause is a CNF clause with at most one positive literal. R1 to R5 are definite clauses (exactly one positive literal). R6 and R7 have a negated head, so their CNF form has no positive literal: they act as integrity constraints.

| Id | Kind | Horn form | CNF form |
|---|---|---|---|
| R1 | horn | OverdueService(v) AND HighVibration(v) => NeedsInspection(v) | NOT OverdueService(v) OR NOT HighVibration(v) OR NeedsInspection(v) |
| R2 | horn | EngineHot(v) => NeedsInspection(v) | NOT EngineHot(v) OR NeedsInspection(v) |
| R3 | horn | NeedsInspection(v) AND LowBattery(v) => HighPriority(v) | NOT NeedsInspection(v) OR NOT LowBattery(v) OR HighPriority(v) |
| R4 | horn | HighDemand(z) AND RainForecast(z) => DemandSurge(z) | NOT HighDemand(z) OR NOT RainForecast(z) OR DemandSurge(z) |
| R5 | horn | DemandSurge(z) AND Adjacent(z, y) AND LowDemand(y) => RepositionCandidate(y, z) | NOT DemandSurge(z) OR NOT Adjacent(z, y) OR NOT LowDemand(y) OR RepositionCandidate(y, z) |
| R6 | horn | NeedsInspection(v) => NOT AvailableForDispatch(v) | NOT NeedsInspection(v) OR NOT AvailableForDispatch(v) |
| R7 | constraint | InService(v) => NOT AvailableForDispatch(v) | NOT InService(v) OR NOT AvailableForDispatch(v) |

## Inference method

Forward chaining: start from the known facts, fire every rule whose body is satisfied (binding variables consistently across the body), add the head as a new fact with provenance `[rule id, supporting fact ids]`, and repeat until nothing new appears. This is sound and complete for Horn clauses and runs in time linear in the size of the knowledge base.

## Worked example (entailment case 6)

1. Told: HighDemand(Z3), RainForecast(Z3), Adjacent(Z3, Z5), LowDemand(Z5)
2. R4 fires with z = Z3: DemandSurge(Z3)
3. R5 fires with z = Z3, y = Z5: RepositionCandidate(Z5, Z3)
4. Query RepositionCandidate(Z5, Z3) is entailed; its support chain lists R5, R4 and the four told facts.
