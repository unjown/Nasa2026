from services.scientific_context import (
    build_qualitative_context
)


def trend_context(data: dict) -> dict:
    return build_qualitative_context(data)



def seasonal_context(data: dict) -> dict:
    valid = [
        item for item in data.get("climatology", [])
        if item.get("mean") is not None
    ]

    if not valid:
        return {
            "view": "seasonal",
            "status": "insufficient_data"
        }

    highest = max(valid, key=lambda x: x["mean"])
    lowest = min(valid, key=lambda x: x["mean"])

    return {
        "view": "seasonal",
        "indicator": data.get("variable"),
        "region": data.get("region"),
        "highest_month": highest["label"],
        "lowest_month": lowest["label"],
        "status": "complete_monthly_analysis",
        "limitations": data.get("warnings", [])
    }


def anomaly_context(data: dict) -> dict:
    valid = [
        item for item in data.get("observations", [])
        if item.get("anomaly") is not None
    ]

    if not valid:
        return {
            "view": "anomalies",
            "status": "insufficient_data"
        }

    highest = max(valid, key=lambda x: x["anomaly"])
    lowest = min(valid, key=lambda x: x["anomaly"])

    return {
        "view": "anomalies",
        "indicator": data.get("variable"),
        "region": data.get("region"),
        "highest_positive_anomaly": highest["period"],
        "highest_negative_anomaly": lowest["period"],
        "status": "complete_monthly_analysis",
        "limitations": data.get("warnings", [])
    }
