# 04 Uncertainty

Two models that reason under uncertainty. Run each from the project folder (environment on):

    python 04_uncertainty/bayes_net.py
    python 04_uncertainty/temporal_hmm.py

Results are saved to `05_ml/artifacts/bayes_net_results.json` and `05_ml/artifacts/temporal_results.json`.

## bayes_net.py: Bayesian network for 7-day failure risk

**Question it answers:** given what the sensors show right now, how likely is this vehicle to fail within 7 days, and what is its hidden wear level?

Structure: Rain, OldVehicle and Overdue influence Wear; Wear causes Vibration, EngineTemp and FailureSoon. Wear is learned from historical inspection labels but is never observed at run time, so the network infers it from the symptoms. Conditional probability tables are learned with a BDeu prior so no combination gets probability zero.

Key results (seed 42):

| Evidence | P(fail within 7 days) | Most likely Wear |
|---|---|---|
| none | 0.17 | Healthy (0.64) |
| high vibration | 0.64 | Critical (0.73) |
| high vibration + hot engine | 0.78 | Critical (0.99) |
| low vibration + normal temp | 0.05 | Healthy (0.93) |
| old, overdue, medium vibration | 0.30 | Worn (0.77) |

Held-out ROC AUC is 0.838, close to the Phase 1 naive Bayes (0.847). The network gives up a little accuracy by binning sensor values into low/medium/high, and in return it explains its answer through the inferred Wear state and still answers when some sensors are missing.

**Where it is uncertain:** "old, overdue, medium vibration" gives P(fail) = 0.30 with Wear spread across Healthy, Worn and Critical. That is not strong enough to act on. The agent will mark the conclusion `uncertain` and escalate rather than dispatch maintenance.

## temporal_hmm.py: reasoning over time

**Part A, hidden health states.** A 3-state Gaussian hidden Markov model reads each vehicle's daily vibration and engine temperature as a sequence and recovers the hidden state (Healthy, Worn, Critical). Because it uses the whole sequence, one noisy reading does not flip the state.

| Method | Hidden-state accuracy (held-out days) |
|---|---|
| Day-by-day vibration threshold (no memory) | 0.819 |
| Gaussian HMM (Viterbi decoding) | 0.976 |

The learned transition matrix shows wear building slowly (Healthy stays Healthy 94% of days) and the 25% Critical-to-Healthy jump reflects repairs. The agent uses the HMM's filtered belief on the latest day as its estimate of each vehicle's current state.

**Where it is uncertain:** on the last day, vehicles such as V38 sit at about 80% Worn and 20% Critical. The HMM reports that split instead of forcing a label, and the policy treats it as evidence, not proof.

**Part B, demand forecast.** The daily and weekly pattern (zone x hour x weekday/weekend) is learned from dry hours only, then multiplied by a rain lift (x1.37) learned from training data.

| Method | Demand RMSE (held-out days) |
|---|---|
| Phase 1 mean by zone/hour/weekend | 3.527 |
| Seasonal profile + rain adjustment | 3.267 |
