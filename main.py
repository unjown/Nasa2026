
import asyncio
import math
import threading

from functools import lru_cache
from pathlib import Path

import numpy as np

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from typing import Literal
from pydantic import BaseModel, Field

from services.data_service import (
    list_datasets,
    load_timeseries,
    aggregate_annual,
    DEFAULT_RAW_DIR
)

from analysis.statistics import (
    analyze_trend,
    calculate_correlation
)
from analysis.seasonal import analyze_monthly

from services.scientific_context import (
    build_scientific_context,
    build_qualitative_context
)

from services.predict import (
    load_nasa_forecast_data,
    forecast_timeseries
)
from services.interpret_context import (
    trend_context,
    seasonal_context,
    anomaly_context
)
from services.qwen_service import interpret_statistics

ROOT = Path(__file__).resolve().parent
app = FastAPI(title='EarthTrend PH', version='0.1.0')
app.mount('/dashboard', StaticFiles(directory=ROOT / 'dashboard', html=True), name='dashboard')

@app.get('/')
def home():
    return RedirectResponse('/dashboard/')

@lru_cache(maxsize=1)
def catalog():
    return [d for d in list_datasets(DEFAULT_RAW_DIR) if 'error' not in d and d.get('kind') not in ('unknown',)]


def get_file(dataset_id: str):
    normalized_id = dataset_id.replace("\\", "/")

    entry = next(
        (
            d for d in catalog()
            if d["path"].replace("\\", "/") == normalized_id
        ),
        None
    )

    if entry is None:
        raise HTTPException(
            status_code=404,
            detail="Unknown dataset ID"
        )

    path = (
        DEFAULT_RAW_DIR / entry["path"]
    ).resolve()

    if not path.is_relative_to(DEFAULT_RAW_DIR.resolve()):
        raise HTTPException(
            status_code=400,
            detail="Invalid dataset path"
        )

    return path


@app.get("/api/datasets")
def datasets():
    fields = (
        "path",
        "filename",
        "variable",
        "region",
        "unit",
        "kind",
        "temporal_resolution",
        "first_date",
        "last_date",
        "missing_values",
        "data_bbox",
        "requested_bbox"
    )

    result = []

    for dataset in catalog():
        entry = {
            key: dataset.get(key)
            for key in fields
        }

        entry["path"] = entry["path"].replace("\\", "/")

        result.append(entry)

    return result

def subset_annual(dataset_id: str, period: str):
    if period not in ('all','20','10'):
        raise HTTPException(422, 'Invalid period')
    df = load_timeseries(get_file(dataset_id))
    annual = aggregate_annual(df)
    if period != 'all':
        annual = annual.tail(int(period)).copy()
    if len(annual) == 0:
        raise HTTPException(422, 'No complete years available')
    return df, annual

def finite(value):
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {k: finite(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [finite(v) for v in value]
    return value

def build_analysis(dataset_id: str, period: str):
    df, annual = subset_annual(dataset_id, period)
    usable = annual.dropna(subset=['value'])
    if usable.empty:
        raise HTTPException(422, 'No usable annual observations')
    values = usable['value'].to_numpy(dtype=float)
    peak_i = int(np.argmax(values))
    result = None
    error = None
    try:
        result = analyze_trend(annual)
    except ValueError as exc:
        error = str(exc)
    series = [{'year': int(d.year), 'value': float(v)} for d,v in zip(annual['date'], annual['value']) if np.isfinite(v)]
    meta = df.attrs

    original_unit = meta.get("unit", "unknown")
    aggregation = annual.attrs.get("aggregation_applied")

    unit_conversions = {
        "days/month": "days/year",
        "mm/month": "mm/year",
        "periods/month": "periods/year"
    }

    analyzed_unit = (
        unit_conversions.get(original_unit, original_unit)
        if aggregation == "sum"
        else original_unit
    )

    return finite({
        'dataset_id': dataset_id,
        'region': meta.get('region'),
        'variable': meta.get('variable'),
        'unit': meta.get('unit'),
        'analyzed_unit': analyzed_unit,
        'kind': meta.get('kind'),
        'source_format': meta.get('source_format'),
        'requested_bbox': meta.get('requested_bbox'),
        'data_bbox': meta.get('data_bbox'),
        'source_url': meta.get('reproduce_url'),
        'aggregation': annual.attrs.get('aggregation_applied'),
        'series': series,
        'summary': {'mean': float(np.mean(values)), 'maximum': float(values[peak_i]), 'max_year': int(usable.iloc[peak_i]['date'].year), 'n': len(usable)},
        'statistics': result,
        'analysis_error': error,
        'warnings': [
            'Regional observations may represent coarse NASA grid estimates.',
            'Annual results exclude years below the observation-coverage threshold.',
            'Unadjusted statistical significance may be unreliable when serial autocorrelation is present.',
            'Historical trends do not establish causation or predict environmental hazards.'
        ]
    })

def build_seasonal_analysis(
    dataset_id: str,
    period: str
):
    if period not in {"all", "20", "10"}:
        raise HTTPException(
            status_code=422,
            detail="Invalid period"
        )

    df = load_timeseries(
        get_file(dataset_id)
    )

    return finite(
        analyze_monthly(
            df,
            period=period
        )
    )

@app.get("/api/seasonal")
def seasonal(
    dataset_id: str = Query(...),
    period: str = Query("all")
):
    try:
        return build_seasonal_analysis(
            dataset_id,
            period
        )

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

@app.get('/api/timeseries')
def timeseries(dataset_id: str = Query(...), period: str = Query('all')):
    return build_analysis(dataset_id, period)

class AnalysisRequest(BaseModel):
    dataset_id: str
    period: str = 'all'

@app.post('/api/analyze')
def analyze(request: AnalysisRequest):
    return build_analysis(request.dataset_id, request.period)

# Correlation API

class CorrelationRequest(BaseModel):
    dataset_id: str
    dataset_id2: str
    period: str = "all"


@app.post("/api/correlation")
def correlation(request: CorrelationRequest):
    if request.dataset_id == request.dataset_id2:
        raise HTTPException(
            status_code=422,
            detail="Select two different datasets"
        )

    try:
        _, annual1 = subset_annual(
            request.dataset_id,
            request.period
        )

        _, annual2 = subset_annual(
            request.dataset_id2,
            request.period
        )

        result = calculate_correlation(
            annual1,
            annual2
        )

        return finite(result)

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

# Prediction API

PREDICT_LOCK = threading.Lock()


class PredictRequest(BaseModel):
    dataset_id: str
    frequency: str = "monthly"
    horizon: int = Field(default=6, ge=1, le=24)



@lru_cache(maxsize=32)
def cached_forecast(
    dataset_id: str,
    frequency: str,
    horizon: int,
    modified_ns: int
):
    path = get_file(dataset_id)

    with PREDICT_LOCK:
        df = load_nasa_forecast_data(
            str(path),
            frequency=frequency
        )

        kind = df.attrs.get("kind", "")
        variable = str(
            df.attrs.get("variable", "")
        ).upper()

        nonnegative = (
            kind in {
                "event_count",
                "daily_count",
                "precipitation"
            }
            or any(
                keyword in variable
                for keyword in (
                    "SOIL",
                    "GWET",
                    "RZMC",
                    "SFMC"
                )
            )
        )

        return forecast_timeseries(
            df,
            frequency=frequency,
            horizon=horizon,
            nonnegative=nonnegative,
            backtest=True
        )

class InterpretRequest(BaseModel):
    dataset_id: str
    period: str = "all"
    view: Literal["trend", "seasonal", "anomalies"] = "trend"
    question: str = Field(
        default="Explain the environmental trend.",
        max_length=1000
    )



@app.post("/api/predict")
async def predict(request: PredictRequest):

    try:
        if request.frequency not in {
            "monthly",
            "annual"
        }:
            raise HTTPException(
                status_code=422,
                detail="Unsupported frequency"
            )

        path = get_file(request.dataset_id)

        result = await asyncio.to_thread(
            cached_forecast,
            request.dataset_id,
            request.frequency,
            request.horizon,
            path.stat().st_mtime_ns
        )

        return result

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Prediction service unavailable"
        )


# Interpretation API

@app.post("/api/interpret")
async def interpret(request: InterpretRequest):
    try:
        # Trend Analysis
        if request.view == "trend":
            data = await asyncio.to_thread(
                build_analysis,
                request.dataset_id,
                request.period
            )

            qualitative_context = trend_context(data)
            scientific = build_scientific_context(data)

            verified_statistics = {
                "mean": scientific["mean"],
                "maximum": scientific["maximum"],
                "maximum_year": scientific["maximum_year"],
                "trend": scientific["trend"],
                "statistical_testing": scientific[
                    "statistical_testing"
                ],
                "unit": scientific["analyzed_unit"]
            }

        # Seasonal and Anomaly Analysis
        else:
            data = await asyncio.to_thread(
                build_seasonal_analysis,
                request.dataset_id,
                request.period
            )

            if request.view == "seasonal":
                qualitative_context = seasonal_context(data)

            else:
                qualitative_context = anomaly_context(data)

            # Numerical statistics remain in Python.
            # No annual trend statistics for these views.
            verified_statistics = None

        # Generate explanation using Qwen
        answer = await asyncio.to_thread(
            interpret_statistics,
            request.question,
            qualitative_context
        )

        return {
            "answer": answer,
            "view": request.view,
            "statistics": verified_statistics,
            "dataset_id": request.dataset_id,
            "model": "qwen3:1.7b",
            "status": "ai_generated_interpretation"
        }

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Interpretation service unavailable"
        )
