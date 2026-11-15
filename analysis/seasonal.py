"""Monthly NASA seasonality and anomalies with strict observation coverage.

No missing values are replaced with zero. Daily rainfall is summed only when
all calendar days are present; monthly products retain their native values.
"""
from __future__ import annotations

import calendar
import numpy as np
import pandas as pd

MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']


def prepare_monthly(df: pd.DataFrame) -> pd.DataFrame:
    if not {'date', 'value'}.issubset(df.columns):
        raise ValueError('Expected date and value columns')
    meta = df.attrs
    frame = df[['date', 'value']].copy()
    frame['date'] = pd.to_datetime(frame['date'], errors='coerce')
    frame['value'] = pd.to_numeric(frame['value'], errors='coerce').replace([np.inf, -np.inf], np.nan)
    if frame['date'].isna().any() or frame['date'].duplicated().any():
        raise ValueError('Invalid or duplicated source dates')
    if frame.empty:
        raise ValueError('No observations')
    frame = frame.sort_values('date')
    frequency = meta.get('temporal_resolution')
    if frequency == 'daily':
        if meta.get('kind') != 'precipitation_daily' or meta.get('unit') != 'mm/day':
            raise ValueError('Unsupported daily variable or unit for monthly aggregation')
        # Check every calendar date, not just the number of rows. An omitted date
        # and a duplicate date cannot cancel each other out.
        day = frame['date'].dt.normalize()
        if day.duplicated().any():
            raise ValueError('Multiple measurements for a calendar day')
        frame['period'] = frame['date'].dt.to_period('M')
        rows = []
        for period, group in frame.groupby('period'):
            expected = pd.date_range(period.start_time, period.end_time.normalize(), freq='D')
            actual = pd.DatetimeIndex(group['date'].dt.normalize())
            complete = len(group) == len(expected) and actual.equals(expected) and group['value'].notna().all()
            rows.append((period, float(group['value'].sum()) if complete else np.nan, bool(complete)))
        unit = 'mm/month'
    elif frequency == 'monthly':
        frame['period'] = frame['date'].dt.to_period('M')
        if frame['period'].duplicated().any():
            raise ValueError('Multiple source observations in one month')
        rows = [(r.period, float(r.value) if pd.notna(r.value) else np.nan, bool(pd.notna(r.value))) for r in frame.itertuples()]
        unit = meta.get('unit', 'unknown')
    else:
        raise ValueError(f'Unsupported source resolution: {frequency}')
    result = pd.DataFrame(rows, columns=['period','value','complete']).set_index('period')
    expected_months = pd.period_range(result.index.min(), result.index.max(), freq='M')
    result = result.reindex(expected_months)
    result['complete'] = result['complete'].map(lambda v: bool(v) if pd.notna(v) else False).astype(bool)
    result.loc[~result['complete'], 'value'] = np.nan
    result.index.name = 'period'
    result.attrs = {**meta, 'unit': unit, 'temporal_resolution': 'monthly'}
    return result


def analyze_monthly(df: pd.DataFrame, period: str = 'all', min_samples: int = 5) -> dict:
    if period not in {'all','20','10'}:
        raise ValueError('Invalid time range')
    monthly = prepare_monthly(df)
    if period != 'all':
        last_year = monthly.index.max().year
        monthly = monthly[monthly.index.year >= last_year - int(period) + 1].copy()
    valid = monthly['value'].notna()
    if valid.sum() == 0:
        raise ValueError('No complete months in selected range')
    # Climatology is calculated only from complete months in the selected range.
    groups = monthly.loc[valid].groupby(monthly.index[valid].month)['value']
    means = groups.mean().to_dict()
    stds = groups.std(ddof=1).to_dict()
    counts = groups.count().to_dict()
    climatology = []
    for month in range(1,13):
        n = int(counts.get(month, 0))
        climatology.append({
            'month': month, 'label': MONTHS[month-1], 'count': n,
            'mean': float(means[month]) if n >= min_samples else None,
            'std': float(stds[month]) if n >= min_samples and np.isfinite(stds.get(month, np.nan)) else None,
        })
    baseline = {d['month']: d for d in climatology}
    observations = []
    for index, row in monthly.iterrows():
        b = baseline[index.month]
        value = float(row.value) if pd.notna(row.value) else None
        anomaly = value - b['mean'] if value is not None and b['mean'] is not None else None
        z = anomaly / b['std'] if anomaly is not None and b['std'] is not None and b['std'] > 0 else None
        observations.append({
            'period': str(index), 'year': int(index.year), 'month': int(index.month),
            'value': value, 'complete': bool(row.complete),
            'anomaly': float(anomaly) if anomaly is not None else None,
            'z_score': float(z) if z is not None else None,
            'unusual': bool(abs(z) >= 2) if z is not None else None,
        })
    return {
        'variable': df.attrs.get('variable'), 'region': df.attrs.get('region'),
        'unit': monthly.attrs['unit'], 'period': period,
        'baseline': 'Same-calendar-month mean of complete observations within selected time range',
        'min_samples_per_month': min_samples,
        'start': str(monthly.index.min()), 'end': str(monthly.index.max()),
        'complete_months': int(valid.sum()), 'incomplete_months': int((~valid).sum()),
        'climatology': climatology, 'observations': observations,
        'warnings': [
            'Missing or incomplete months are excluded, never replaced with zero.',
            'A calendar month needs at least five complete historical observations for a baseline.',
            'Anomalies use the selected period as their reference, not an external climate normal.',
            'A z-score of magnitude at least two is a descriptive flag, not a hazard alert or significance test.'
        ]
    }
