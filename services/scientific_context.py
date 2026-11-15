
# services/scientific_context.py

from __future__ import annotations

import math


# Indicator Definitions
INDICATORS = {
    "M2SMNXEDI_2_R20mm": {
        "name": "Heavy-rainfall days",
        "description": "Number of days with at least 20 mm of rainfall"
    },
    "M2SMNXEDI_2_R95D": {
        "name": "Very wet days",
        "description": "Number of days exceeding the 95th percentile rainfall threshold"
    },
    "M2SMNXEDI_2_RX1Day": {
        "name": "Maximum 1-day rainfall",
        "description": "Largest daily precipitation amount"
    },
    "M2SMNXEDI_2_RX5Day": {
        "name": "Maximum 5-day rainfall",
        "description": "Largest accumulated rainfall over five days"
    },
    "M2SMNXEDI_2_CDD": {
        "name": "Longest dry spell",
        "description": "Maximum consecutive dry days"
    },
    "M2SMNXEDI_2_drydays": {
        "name": "Dry days",
        "description": "Number of dry days"
    },
    "M2SMNXEDI_2_wetdays": {
        "name": "Wet days",
        "description": "Number of wet days"
    }
}


# Region Definitions
REGIONS = {
    "metro_manila_approximate": "Metro Manila vicinity",
    "benguet_approximate": "Benguet vicinity",
    "angat_vicinity": "Angat vicinity",
    "philippines_bbox": "Philippines bounding box"
}


# Unit Normalization
def analyzed_unit(data: dict) -> str:
    unit = data.get("unit", "unknown")
    aggregation = data.get("aggregation", "")

    if aggregation == "sum":
        conversions = {
            "days/month": "days/year",
            "mm/month": "mm/year"
        }

        return conversions.get(unit, unit)

    return unit


# Statistical Significance
def significance_status(stat: dict | None) -> str:
    if not stat:
        return "unavailable"

    if stat.get("autocorrelation_flag"):
        return "preliminary_due_to_autocorrelation"

    return "unadjusted_test_only"


# Scientific Context
def build_scientific_context(data: dict) -> dict:
    variable = data.get("variable", "")
    region = data.get("region", "")
    stat = data.get("statistics") or {}
    summary = data.get("summary") or {}

    indicator = INDICATORS.get(
        variable,
        {
            "name": variable,
            "description": "Environmental observation"
        }
    )

    unit = analyzed_unit(data)

    slope = stat.get("slope_per_decade")
    confidence_interval = stat.get(
        "slope_ci_per_decade"
    )

    direction = "unknown"

    if slope is not None:
        if slope > 0:
            direction = "increasing"
        elif slope < 0:
            direction = "decreasing"
        else:
            direction = "stable"

    return {
        "indicator": indicator["name"],
        "definition": indicator["description"],
        "region": REGIONS.get(region, region),

        "observation_period": {
            "start": stat.get("start_date"),
            "end": stat.get("end_date")
        },

        "analyzed_frequency": "annual",
        "analyzed_unit": unit,

        "mean": summary.get("mean"),
        "maximum": summary.get("maximum"),
        "maximum_year": summary.get("max_year"),

        "trend": {
            "direction": direction,
            "slope_per_decade": slope,
            "slope_unit": f"{unit} per decade"
        },

        "statistical_testing": {
            "significance_status": significance_status(stat),
            "mann_kendall_p_value_unadjusted":
                stat.get("mann_kendall_p_value"),
            "confidence_interval_95_unadjusted":
                confidence_interval,
            "residual_autocorrelation":
                stat.get("residual_lag1_autocorrelation")
        },

        "scientific_limitations": [
            "The statistical tests are not adjusted for serial autocorrelation.",
            "A historical trend does not establish causation.",
            "Environmental trends do not directly predict disasters.",
            "The source may represent coarse gridded estimates rather than exact administrative boundaries."
        ]
    }

def build_qualitative_context(data: dict) -> dict:
    scientific = build_scientific_context(data)

    trend = scientific["trend"]
    testing = scientific["statistical_testing"]

    return {
        "indicator": scientific["indicator"],
        "definition": scientific["definition"],
        "region": scientific["region"],
        "trend_direction": trend["direction"],
        "significance_status": testing["significance_status"],
        "analyzed_frequency": scientific["analyzed_frequency"],
        "geographic_limitation": (
            "The observations may represent coarse "
            "gridded estimates rather than precise "
            "administrative boundaries."
        ),
        "scientific_limitations": scientific[
            "scientific_limitations"
        ]
    }
