"""04_uncertainty/temporal_hmm.py: temporal reasoning over sequential percepts.

Part A: a Gaussian hidden Markov model recovers each vehicle's hidden health state
        (Healthy / Worn / Critical) from its daily vibration and engine temperature.
        Because it uses the sequence, one noisy reading does not flip the state,
        unlike a day-by-day threshold rule.
Part B: a seasonal demand forecast (zone x hour x weekday/weekend pattern from dry hours,
        scaled by a learned rain lift) compared with the Phase 1 mean baseline.

Run directly:   python 04_uncertainty/temporal_hmm.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error

from settings import load_config

HEALTH = ["Healthy", "Worn", "Critical"]
OBS = ["vibration", "engine_temp"]


def sequences(tele: pd.DataFrame):
    tele = tele.sort_values(["vehicle_id", "date"])
    lengths = tele.groupby("vehicle_id", sort=True).size().to_numpy()
    return tele, tele[OBS].to_numpy(), lengths


def fit_hmm(X: np.ndarray, lengths: np.ndarray, seed: int) -> GaussianHMM:
    hmm = GaussianHMM(n_components=3, covariance_type="diag", n_iter=200, random_state=seed)
    hmm.fit(X, lengths)
    return hmm


def label_components(hmm: GaussianHMM) -> dict[int, str]:
    """HMM states are unnamed numbers; name them by average vibration (lowest = Healthy)."""
    order = np.argsort(hmm.means_[:, 0])
    return {int(c): HEALTH[i] for i, c in enumerate(order)}


def threshold_baseline(df: pd.DataFrame) -> np.ndarray:
    """Day-by-day rule with no memory: classify from today's vibration only."""
    return np.where(df.vibration > 4.5, "Critical", np.where(df.vibration > 3.0, "Worn", "Healthy"))


def part_a(cfg: dict, tele: pd.DataFrame) -> dict:
    seed = cfg["project"]["seed"]
    split = tele.date.min() + pd.Timedelta(days=cfg["sim_details"]["train_days"])
    train, test = tele[tele.date < split], tele[tele.date >= split]

    _, X_train, len_train = sequences(train)
    hmm = fit_hmm(X_train, len_train, seed)
    names = label_components(hmm)

    test_sorted, X_test, len_test = sequences(test)
    decoded = hmm.predict(X_test, len_test)                     # Viterbi: most likely state path
    pred = np.array([names[int(s)] for s in decoded])
    truth = test_sorted.true_state.replace({"Failed": "Critical"}).to_numpy()
    base = threshold_baseline(test_sorted)

    acc_hmm, acc_base = accuracy_score(truth, pred), accuracy_score(truth, base)
    print("Learned transition matrix (rows = today, cols = tomorrow):")
    trans = pd.DataFrame(hmm.transmat_, index=[names[i] for i in range(3)],
                         columns=[names[i] for i in range(3)]).loc[HEALTH, HEALTH].round(3)
    print(trans)
    print("\nState means (vibration, engine_temp):")
    for c in sorted(range(3), key=lambda c: HEALTH.index(names[c])):
        print(f"  {names[c]:8s} {hmm.means_[c].round(2)}")
    print(f"\nHidden-state accuracy on held-out days: HMM {acc_hmm:.3f} vs threshold rule {acc_base:.3f}")
    print("Confusion matrix for the HMM (rows = truth, cols = predicted):")
    print(pd.DataFrame(confusion_matrix(truth, pred, labels=HEALTH), index=HEALTH, columns=HEALTH))

    # Filtering: current-state belief for each vehicle on the last day (what the agent uses)
    last_beliefs = {}
    start = 0
    for vid, n in zip(test_sorted.vehicle_id.unique(), len_test):
        post = hmm.predict_proba(X_test[start:start + n])[-1]
        last_beliefs[vid] = {names[c]: round(float(p), 3) for c, p in enumerate(post)}
        start += n
    risky = sorted(last_beliefs.items(), key=lambda kv: -kv[1]["Critical"])[:5]
    print("\nVehicles most likely Critical on the last day:")
    for vid, b in risky:
        print(f"  {vid}: {b}")

    return {"hmm_accuracy": round(acc_hmm, 3), "threshold_accuracy": round(acc_base, 3),
            "transition_matrix": trans.to_dict(), "top_critical_last_day": dict(risky)}


def part_b(cfg: dict, trips: pd.DataFrame) -> dict:
    split = trips.timestamp.min() + pd.Timedelta(days=cfg["sim_details"]["train_days"])
    trips = trips.sort_values(["zone", "timestamp"]).copy()
    trips["is_weekend"] = (trips.timestamp.dt.dayofweek >= 5).astype(int)
    train, test = trips[trips.timestamp < split], trips[trips.timestamp >= split].copy()

    mean_pred = train.groupby(["zone", "hour", "is_weekend"]).trips.mean().rename("mean_pred")
    test = test.join(mean_pred, on=["zone", "hour", "is_weekend"])

    # Seasonal forecast: the daily and weekly pattern (zone x hour x weekday/weekend) learned from
    # DRY hours only, then multiplied by a rain lift learned from training data. The mean baseline
    # mixes rainy and dry hours into one average, so it is wrong in both directions.
    dry_profile = (train[train.rain == 0].groupby(["zone", "hour", "is_weekend"]).trips.mean()
                   .rename("dry_profile"))
    rain_lift = train[train.rain == 1].trips.mean() / train[train.rain == 0].trips.mean()
    test = test.join(dry_profile, on=["zone", "hour", "is_weekend"])
    test["seasonal_pred"] = test.dry_profile * np.where(test.rain == 1, rain_lift, 1.0)

    rmse_mean = mean_squared_error(test.trips, test.mean_pred) ** 0.5
    rmse_seasonal = mean_squared_error(test.trips, test.seasonal_pred) ** 0.5
    print(f"\nDemand RMSE on held-out days: seasonal + rain {rmse_seasonal:.3f} vs mean baseline {rmse_mean:.3f}")
    print(f"Rain lift learned from training data: x{rain_lift:.2f}")
    return {"rmse_seasonal_rain": round(rmse_seasonal, 3), "rmse_mean_baseline": round(rmse_mean, 3),
            "rain_lift": round(rain_lift, 3)}


def main() -> dict:
    cfg = load_config()
    syn = cfg["paths"]["data_synthetic"]
    for name in ["telemetry.csv", "trips.csv"]:
        if not (syn / name).exists():
            raise FileNotFoundError(f"{syn / name} not found. Run 02_simulation.ipynb first.")
    tele = pd.read_csv(syn / "telemetry.csv", parse_dates=["date"])
    trips = pd.read_csv(syn / "trips.csv", parse_dates=["timestamp"])
    print("=== Part A: hidden health states (HMM) ===")
    results = {"health_hmm": part_a(cfg, tele)}
    print("\n=== Part B: demand forecast ===")
    results["demand"] = part_b(cfg, trips)
    out = cfg["paths"]["artifacts"] / "temporal_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved", out)
    return results


if __name__ == "__main__":
    main()
