# 01 Problem and PEAS: VertexFleet Autonomous Fleet Demand & Maintenance Agent

VertexFleet Analytics is a fictitious transport analytics company. All data in this project is synthetic and generated in 02_simulation.ipynb.

## Product and users
An intelligent agent that forecasts hourly trip demand per zone and 7-day vehicle failure risk, then recommends where to position vehicles and which vehicles to send for maintenance. Every recommendation shows the evidence behind it, and high-impact actions wait for human approval.

Users:
- Fleet operations manager: approves repositioning plans
- Maintenance supervisor: approves maintenance dispatches and manages service bays
- Dispatcher: views demand forecasts and vehicle status

## Business / learning goal
Business: reduce unmet trip requests and unplanned breakdowns without exceeding service bay or technician capacity.
Learning: demonstrate a transparent six-stage agent (percepts, propositions, knowledge base, inference, entailment, action) that combines search, CSP, logic, Bayesian, temporal and machine learning methods.

## Stakeholders
| Stakeholder | Interest |
|---|---|
| Fleet operations manager | Fewer unmet trips, efficient vehicle placement |
| Maintenance supervisor | Fewer breakdowns, balanced bay workload |
| Dispatchers | Accurate, timely demand forecasts |
| Riders | Shorter wait times, reliable vehicles |
| Course instructor / reviewer | Traceable reasoning, reproducible results, all 13 course topics used |

## Goals
1. Forecast hourly demand per zone more accurately than a mean-by-hour baseline (lower RMSE).
2. Predict vehicles likely to fail within 7 days with higher recall than a logistic regression baseline at the same precision.
3. Recommend repositioning and maintenance plans that never violate a hard constraint.
4. Log every observation, decision and action so a reviewer can trace any recommendation end to end.
5. Abstain and escalate to a human when evidence is weak or contradictory.

## Non-goals
- Controlling real vehicles or connecting to real fleet systems
- Using real customer, rider or driver data
- Pricing, fares or revenue optimization
- Driver scheduling, performance or HR decisions
- Fully autonomous execution of high-impact actions

## Problem formulation
- State: for each vehicle (location zone, health telemetry, hours since service, in service or available) plus zone demand forecasts, weather signal, bay and technician availability, and the current hour.
- Actions: query fleet data, look up service history, write a report, propose repositioning, propose maintenance dispatch.
- Transition model: each hour, demand arrives by zone, vehicles move and degrade according to a hidden health Markov chain (Healthy, Worn, Critical, Failed), and approved actions change vehicle location or service status.
- Goal test: no zone has unmet demand above threshold and no vehicle with high entailed failure risk is left in service, while all hard constraints hold.
- Path cost: unmet trips + breakdown penalty + repositioning miles + maintenance cost.

## PEAS
| Element | Description |
|---|---|
| Performance measure | Unmet-demand rate; breakdowns per 1,000 vehicle-hours; demand forecast RMSE; failure-model recall at fixed precision; share of recommendations approved without override; zero hard-constraint violations |
| Environment | Simulated city with 8 zones, 50 vehicles and 3 depots with service bays; hourly time steps over 90 days; simulated weather (rain, temperature) affecting demand and vehicle wear |
| Actuators | Allow-listed tools only: query_fleet_db, lookup_service_history, write_report, propose_action (reposition or maintenance), all logged and approval-gated where required |
| Sensors | Trip demand feed by zone and hour; vehicle telemetry (vibration, engine temperature, battery, odometer); service history records; weather signal; results of the agent's own tool calls |

## Environment assumptions
| Property | Value | Why |
|---|---|---|
| Observable | Partially | True vehicle health is hidden; only noisy telemetry is seen |
| Deterministic | Stochastic | Demand and failures are random |
| Episodic | Sequential | Today's maintenance choice affects tomorrow's failures |
| Static | Dynamic | Demand and vehicle health change while the agent decides |
| Discrete | Discrete time, mixed values | Hourly steps; telemetry values are continuous |
| Agents | Single agent | One fleet agent; humans approve actions |

## Autonomy boundary
| Level | Actions |
|---|---|
| Agent may do alone | Query data, run forecasts and models, write reports, create proposals |
| Needs human approval | Dispatch a vehicle to maintenance; reposition vehicles between zones |
| Never allowed | Delete records; override a vehicle safety lockout; act while the kill switch is on; call any tool not on the allow-list |

## Six-stage agent cycle
1. PERCEPTS: raw observations captured with source, timestamp, quality score and validation status.
2. PROPOSITIONS: percepts turned into testable statements such as HighVibration(V12) or HighDemand(Zone3, 17:00), plus model features; contradictions flagged.
3. KNOWLEDGE BASE: facts, Horn rules, hard constraints, ontology relations, model versions and provenance.
4. INFERENCE: forward chaining, A* and CSP, Bayesian network, HMM and ML models, each producing an inspectable trace.
5. ENTAILMENT: each conclusion labeled entailed, contradicted, uncertain or unknown. Model confidence alone is never treated as proof.
6. ACTION: the policy gate checks goals, constraints, risk and approval rules; only allow-listed tools run; each result is logged and fed back as a new percept.

## Risks
| Risk | Mitigation |
|---|---|
| Model overconfidence treated as fact | Entailment stage separates "supported by a model" from "entailed by the KB" |
| Contradictory or noisy sensor data | Quality scores, contradiction checks, abstain-and-escalate below confidence threshold |
| Unsafe or runaway actions | Approval gates, allow-listed tools, bounded iterations, kill switch, audit log |
| Prompt injection through maintenance notes | Note text is treated as data, never as instructions; policy gate blocks unsafe tool calls |
| Unfair service across zones | Fairness test on repositioning share relative to demand per zone |
| Data drift after deployment | PSI drift monitoring, evaluation thresholds, documented rollback |

## Baseline plan
| Task | Baseline | Planned improvement |
|---|---|---|
| Demand forecasting | Mean demand by zone and hour | Regularized and ensemble regressors, LSTM |
| Failure prediction | Logistic regression | SVM, random forest, gradient boosting, calibrated |
| Maintenance note triage | TF-IDF + logistic regression | Fine-tuned DistilBERT |
| Routing to depot | Breadth-first search | A* with distance heuristic |
| Bay and technician assignment | Plain backtracking CSP | Backtracking with MRV heuristic |
| Maintenance decision | Hand-written rules | Rules + Bayesian network + HMM health states |