# 03 Logic: Proposition Vocabulary

Every proposition is a predicate applied to constants: vehicles (V01 to V50), zones (Z1 to Z8), depots (D1 to D3).
Percept-level predicates are set by `src/agent/proposition.py` from raw percepts using thresholds in `configs/config.yaml`.
Derived predicates are only ever produced by rules in `rules_horn.md`, never set directly.

## Percept-level predicates (set from data)

| Predicate | Meaning | Set true when | Source percept |
|---|---|---|---|
| HighVibration(v) | Vehicle v vibrates abnormally | vibration > logic_rules.vibration_high (4.0 mm/s) | telemetry |
| EngineHot(v) | Vehicle v runs hot | engine_temp > logic_rules.engine_hot (100 C) | telemetry |
| LowBattery(v) | Vehicle v battery is degraded | battery_health < 0.6 | telemetry |
| OverdueService(v) | Vehicle v is past due for service | days_since_service > logic_rules.overdue_days (20) | service history |
| InService(v) | Vehicle v is at a depot being serviced | service record open for v | service history |
| AvailableForDispatch(v) | Vehicle v reports itself ready for trips | vehicle status ping says available | telemetry / tool |
| HighDemand(z) | Zone z demand forecast is high | forecast > 75th percentile for that hour | trips (forecast) |
| LowDemand(z) | Zone z demand forecast is low | forecast < 25th percentile for that hour | trips (forecast) |
| RainForecast(z) | Rain expected in zone z | rain flag = 1 for the forecast hour | weather |
| Adjacent(z, y) | Zones z and y border each other | from the city map | static map |
| LocatedIn(v, z) | Vehicle v is currently in zone z | latest position ping | telemetry |

## Derived predicates (produced by rules only)

| Predicate | Meaning | Rule |
|---|---|---|
| NeedsInspection(v) | Vehicle v should be inspected | R1, R2 |
| HighPriority(v) | Inspection for v is urgent | R3 |
| DemandSurge(z) | Zone z will need extra vehicles | R4 |
| RepositionCandidate(y, z) | Move a vehicle from zone y to zone z | R5 |
| not AvailableForDispatch(v) | Vehicle v must not take trips | R6, R7 |

## Not a proposition

Model outputs such as "failure probability 0.92" are **evidence**, not facts. They enter the agent as model support in the entailment check and can make a conclusion `uncertain`, never `entailed`.

## Contradictions

A contradiction exists when the knowledge base holds both X and not X. Example: `InService(V03)` (so R7 derives `not AvailableForDispatch(V03)`) while a new percept says `AvailableForDispatch(V03)`. The knowledge base keeps both facts, flags the pair, and the agent refuses to act on that vehicle until a human resolves it.
