
from __future__ import annotations

import argparse
import json
import os
import threading
from typing import Any

import numpy as np
import pandas as pd

from services.data_service import load_timeseries, aggregate_annual


# Model Configuration
MODEL_ID = os.getenv(
    "TIMESFM_MODEL",
    "google/timesfm-2.5-200m-pytorch"
)

_MODEL = None
_MODEL_LOCK = threading.Lock()


# Model Loading
def _model():
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is None:
            try:
                import timesfm

            except ImportError as exc:
                raise RuntimeError(
                    "TimesFM is not installed. "
                    "Run: python -m pip install 'timesfm[torch]'"
                ) from exc

            model = (
                timesfm.TimesFM_2p5_200M_torch
                .from_pretrained(MODEL_ID)
            )

            model.compile(
                timesfm.ForecastConfig(
                    max_context=512,
                    max_horizon=24,
                    per_core_batch_size=1,
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    fix_quantile_crossing=True
                )
            )

            _MODEL = model

    return _MODEL


# Data Preparation
def _prepare(
    df: pd.DataFrame,
    frequency: str
) -> pd.DataFrame:

    if frequency not in {"monthly", "annual"}:
        raise ValueError(
            "frequency must be 'monthly' or 'annual'"
        )

    if not {"date", "value"}.issubset(df.columns):
        raise ValueError(
            "Input requires date and value columns"
        )

    clean = df[["date", "value"]].copy()

    clean["date"] = pd.to_datetime(
        clean["date"],
        errors="coerce"
    )

    clean["value"] = pd.to_numeric(
        clean["value"],
        errors="coerce"
    )

    clean = clean.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna(subset=["date"])

    clean = clean.sort_values("date")

    if clean.empty:
        raise ValueError("No valid observations")

    if clean["date"].duplicated().any():
        raise ValueError("Duplicate timestamps found")

    period_type = (
        "M" if frequency == "monthly" else "Y"
    )

    clean["period"] = (
        clean["date"].dt.to_period(period_type)
    )

    if clean["period"].duplicated().any():
        raise ValueError(
            "Multiple observations in the same period. "
            "Aggregate before forecasting."
        )

    expected = pd.period_range(
        clean["period"].iloc[0],
        clean["period"].iloc[-1],
        freq=period_type
    )

    actual = pd.PeriodIndex(clean["period"])

    if not actual.equals(expected):
        raise ValueError(
            "Missing time periods detected. "
            "Use a complete contiguous series."
        )

    if clean["value"].isna().any():
        raise ValueError(
            "Missing values detected. "
            "Preprocess explicitly before forecasting."
        )

    minimum = 24 if frequency == "monthly" else 8

    if len(clean) < minimum:
        raise ValueError(
            f"At least {minimum} observations required"
        )

    return clean.reset_index(drop=True)



# Monthly Aggregation

def _aggregate_monthly(
    df: pd.DataFrame
) -> pd.DataFrame:

    kind = df.attrs.get("kind", "")
    unit = df.attrs.get("unit", "")

    if kind != "precipitation_daily":
        raise ValueError(
            f"Unsupported daily aggregation: {kind}"
        )

    if unit != "mm/day":
        raise ValueError(
            f"Unexpected precipitation unit: {unit}"
        )

    frame = df[["date", "value"]].copy()

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce"
    )

    frame["value"] = pd.to_numeric(
        frame["value"],
        errors="coerce"
    )

    if frame["date"].isna().any():
        raise ValueError("Invalid observation dates")

    if frame["date"].duplicated().any():
        raise ValueError("Duplicate daily observations")

    frame = frame.sort_values("date")
    frame = frame.set_index("date")

    daily = frame["value"]

    # Convert daily precipitation totals into monthly totals
    monthly = daily.resample("MS").sum(
        min_count=1
    )

    # Count valid daily observations
    valid_days = daily.resample("MS").count()

    # Require all calendar days in each month
    expected_days = monthly.index.days_in_month

    monthly = monthly.where(
        valid_days == expected_days
    )

    result = monthly.reset_index()

    result.attrs = df.attrs.copy()
    result.attrs["unit"] = "mm/month"
    result.attrs["kind"] = "precipitation_monthly"
    result.attrs["temporal_resolution"] = "monthly"

    return result

# Forecasting
def _forecast_array(
    values: np.ndarray,
    horizon: int
) -> tuple[np.ndarray, Any]:

    model = _model()

    point, quantiles = model.forecast(
        horizon=horizon,
        inputs=[values.astype(np.float32)]
    )

    point = np.asarray(point)[0].reshape(-1)

    if len(point) != horizon:
        raise RuntimeError(
            "Unexpected TimesFM output shape"
        )

    return point, quantiles


# Rolling Historical Backtesting

def rolling_backtest(
    values: np.ndarray,
    *,
    frequency: str = "monthly",
    horizon: int = 6,
    windows: int = 4,
    nonnegative: bool = False
) -> dict:

    minimum = 24 if frequency == "monthly" else 8

    results = []

    for i in range(windows, 0, -1):
        end = len(values) - (i - 1) * horizon

        train_end = end - horizon

        if train_end < minimum:
            continue

        train = values[:train_end]
        actual = values[train_end:end]

        predicted, _ = _forecast_array(
            train[-512:],
            horizon
        )

        if nonnegative:
            predicted = np.maximum(predicted, 0)

        if frequency == "monthly" and len(train) >= 12:
            baseline = np.array([
                train[-12 + (j % 12)]
                for j in range(horizon)
            ])
            baseline_name = "seasonal_naive"
        else:
            baseline = np.repeat(train[-1], horizon)
            baseline_name = "last_value"

        mae = float(
            np.mean(np.abs(actual - predicted))
        )

        baseline_mae = float(
            np.mean(np.abs(actual - baseline))
        )

        results.append({
            "window": len(results) + 1,
            "timesfm_mae": mae,
            "baseline_mae": baseline_mae,
            "beats_baseline": mae < baseline_mae
        })

    if not results:
        raise ValueError(
            "Insufficient history for rolling backtesting"
        )

    return {
        "windows": len(results),
        "horizon": horizon,
        "baseline": baseline_name,
        "timesfm_mae": float(
            np.mean([
                r["timesfm_mae"]
                for r in results
            ])
        ),
        "baseline_mae": float(
            np.mean([
                r["baseline_mae"]
                for r in results
            ])
        ),
        "individual_windows": results,
        
        "warnings": [
            "Forecasts are experimental model estimates, not NASA observations.",
            "Forecasts do not represent disaster probabilities.",
            "A single holdout and limited rolling backtests do not guarantee future accuracy.",
            "Further validation across additional periods and datasets is recommended."
        ]

    }

def forecast_timeseries(
    df: pd.DataFrame,
    *,
    frequency: str = "monthly",
    horizon: int = 6,
    nonnegative: bool = False,
    backtest: bool = True,
    unit: str | None = None
) -> dict[str, Any]:

    if not isinstance(horizon, int) or not 1 <= horizon <= 24:
        raise ValueError(
            "horizon must be between 1 and 24"
        )

    clean = _prepare(df, frequency)

    values = clean["value"].to_numpy(
        dtype=np.float32
    )

    minimum = (
        24 if frequency == "monthly" else 8
    )

    if backtest and len(values) < horizon + minimum:
        raise ValueError(
            "Insufficient history for backtesting"
        )

    point, _ = _forecast_array(
        values[-512:],
        horizon
    )

    if nonnegative:
        point = np.maximum(point, 0)

    period_type = (
        "M" if frequency == "monthly" else "Y"
    )

    last_period = clean["period"].iloc[-1]

    future_periods = pd.period_range(
        last_period + 1,
        periods=horizon,
        freq=period_type
    )
    history_count = 12 if frequency == "monthly" else 5

    recent_history = clean.tail(history_count)

    history = [
        {
            "period": str(row.period),
            "value": float(row.value)
        }
        for row in recent_history.itertuples()
    ]
    result = {
        "model": MODEL_ID,
        "frequency": frequency,
        "unit": unit or df.attrs.get("unit", "unknown"),
        "historical_start": str(clean["period"].iloc[0]),
        "historical_end": str(last_period),
        "observations": len(values),
        "horizon": horizon,
        "history": history,
        "forecast": [
            {
                "period": str(period),
                "value": float(value)
            }
            for period, value in zip(
                future_periods,
                point
            )
        ],

        "status": "experimental_forecast",
        "warnings": [
            "Forecasts are experimental model estimates, not NASA observations.",
            "Forecasts do not represent disaster probabilities.",
            "Limited historical backtesting does not guarantee future accuracy.",
            "Additional validation across periods and datasets is recommended."
        ],  
    }

    # Historical Backtest
    if backtest:
        train = values[:-horizon]
        actual = values[-horizon:]

        predicted, _ = _forecast_array(
            train[-512:],
            horizon
        )

        if nonnegative:
            predicted = np.maximum(
                predicted,
                0
            )

        if frequency == "monthly" and len(train) >= 12:
            baseline = np.array([
                train[-12 + (i % 12)]
                for i in range(horizon)
            ])

            baseline_name = "seasonal_naive"

        else:
            baseline = np.repeat(
                train[-1],
                horizon
            )

            baseline_name = "last_value"

        mae = float(
            np.mean(np.abs(actual - predicted))
        )

        baseline_mae = float(
            np.mean(np.abs(actual - baseline))
        )

        result["backtest"] = {
            "holdout_periods": horizon,
            "timesfm_mae": mae,
            "baseline": baseline_name,
            "baseline_mae": baseline_mae,
            "beats_baseline": mae < baseline_mae,
            "note": "Single holdout evaluation"
        }
        result["rolling_backtest"] = rolling_backtest(
            values,
            frequency=frequency,
            horizon=horizon,
            windows=4,
            nonnegative=nonnegative
        )
    return result


# Raw NASA CSV Loading
def load_nasa_forecast_data(
    path: str,
    frequency: str = "monthly"
) -> pd.DataFrame:

    df = load_timeseries(path)

    source_frequency = df.attrs.get(
        "temporal_resolution",
        "unknown"
    )

    if frequency == "annual":
        return aggregate_annual(df)

    if frequency == "monthly":
        if source_frequency == "daily":
            return _aggregate_monthly(df)

        if source_frequency == "monthly":
            return df

        raise ValueError(
            f"Unsupported source frequency: {source_frequency}"
        )

    raise ValueError(
        "frequency must be monthly or annual"
    )


# Command Line Interface
def main():
    parser = argparse.ArgumentParser(
        description="Forecast raw NASA CSV data with TimesFM"
    )

    parser.add_argument(
        "csv",
        help="Path to raw NASA CSV"
    )

    parser.add_argument(
        "--frequency",
        choices=["monthly", "annual"],
        default="monthly"
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=6
    )

    parser.add_argument(
        "--nonnegative",
        action="store_true"
    )

    parser.add_argument(
        "--no-backtest",
        action="store_true"
    )

    args = parser.parse_args()

    df = load_nasa_forecast_data(
        args.csv,
        frequency=args.frequency
    )

    result = forecast_timeseries(
        df,
        frequency=args.frequency,
        horizon=args.horizon,
        nonnegative=args.nonnegative,
        backtest=not args.no_backtest
    )

    print(
        json.dumps(
            result,
            indent=2
        )
    )


if __name__ == "__main__":
    main()
