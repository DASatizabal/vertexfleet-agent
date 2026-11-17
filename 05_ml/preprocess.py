"""05_ml/preprocess.py: feature engineering shared by the notebooks and the agent.

Every feature uses only the current day/hour and the past, never the future,
so the same code is safe to run live inside the agent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------- failure prediction (one row per vehicle per day) ----------------
RAW_FAILURE_FEATURES = ["vibration", "engine_temp", "battery_health", "days_since_service",
                        "age_years", "rain", "km_today"]
ENGINEERED_FAILURE_FEATURES = ["vib_mean_7d", "vib_std_7d", "vib_change_7d", "temp_mean_7d",
                               "temp_change_7d", "rain_days_7d", "km_7d"]
FAILURE_FEATURES = RAW_FAILURE_FEATURES + ENGINEERED_FAILURE_FEATURES
FAILURE_TARGET = "fails_within_7d"


def build_failure_features(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Add rolling 7-day features per vehicle (past and current day only)."""
    df = telemetry.sort_values(["vehicle_id", "date"]).copy()
    g = df.groupby("vehicle_id", sort=False)
    roll = lambda col, fn: g[col].transform(lambda s: getattr(s.rolling(7, min_periods=1), fn)())
    df["vib_mean_7d"] = roll("vibration", "mean")
    df["vib_std_7d"] = roll("vibration", "std").fillna(0.0)
    df["vib_change_7d"] = df.vibration - g.vibration.shift(7).fillna(df.vibration)
    df["temp_mean_7d"] = roll("engine_temp", "mean")
    df["temp_change_7d"] = df.engine_temp - g.engine_temp.shift(7).fillna(df.engine_temp)
    df["rain_days_7d"] = roll("rain", "sum")
    df["km_7d"] = roll("km_today", "sum")
    return df


# ---------------- demand forecasting (one row per zone per hour) ----------------
DEMAND_NUMERIC = ["hour_sin", "hour_cos", "is_weekend", "rain", "temp_c",
                  "lag_24h", "lag_168h", "mean_24h"]
DEMAND_CATEGORICAL = ["zone"]
DEMAND_TARGET = "trips"


def build_demand_features(trips: pd.DataFrame) -> pd.DataFrame:
    """Calendar encoding plus lagged demand per zone. Drops the first week (no 168h lag yet)."""
    df = trips.sort_values(["zone", "timestamp"]).copy()
    df["hour_sin"] = np.sin(2 * np.pi * df.hour / 24)   # 23:00 and 00:00 end up close together
    df["hour_cos"] = np.cos(2 * np.pi * df.hour / 24)
    df["is_weekend"] = (df.dayofweek >= 5).astype(int)
    g = df.groupby("zone", sort=False).trips
    df["lag_24h"] = g.shift(24)
    df["lag_168h"] = g.shift(168)
    df["mean_24h"] = g.transform(lambda s: s.shift(1).rolling(24, min_periods=1).mean())
    return df.dropna(subset=["lag_24h", "lag_168h", "mean_24h"]).reset_index(drop=True)


# ---------------- time-based split ----------------
def cutoff_date(start: pd.Timestamp, train_days: int) -> pd.Timestamp:
    """The split point: simulation start + train_days (same for every model, so results compare)."""
    return pd.Timestamp(start).normalize() + pd.Timedelta(days=train_days)


def time_split(df: pd.DataFrame, time_col: str, cutoff: pd.Timestamp):
    """Train before the cutoff, test from the cutoff on. Never shuffle time series."""
    return df[df[time_col] < cutoff].copy(), df[df[time_col] >= cutoff].copy()
