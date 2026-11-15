"""Read EarthTrend PH's original NASA Giovanni and Earth Engine CSV exports.

No network calls and no mutation of raw files. Return a consistent pandas frame
with columns date, value and attached source metadata in DataFrame.attrs.
"""
from __future__ import annotations

import csv
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_RAW_DIR = Path(__file__).resolve().parents[1] / 'data' / 'raw'

# Verified against the uploaded NASA exports. Keys are the NASA variable codes.
VARIABLES: dict[str, dict[str, str]] = {
    'GPM_3IMERGDF_07_precipitation': {'unit': 'mm/day', 'kind': 'precipitation_daily', 'aggregation': 'sum'},
    'rainfall_mm_month': {'unit': 'mm/month', 'kind': 'precipitation_monthly', 'aggregation': 'sum'},
    'M2SMNXEDI_2_R20mm': {'unit': 'days/month', 'kind': 'event_count', 'aggregation': 'sum'},
    'M2SMNXEDI_2_R95D': {'unit': 'days/month', 'kind': 'event_count', 'aggregation': 'sum'},
    'M2SMNXEDI_2_RX5Daycount': {'unit': 'periods/month', 'kind': 'event_count', 'aggregation': 'sum'},
    'M2SMNXEDI_2_drydays': {'unit': 'days/month', 'kind': 'event_count', 'aggregation': 'sum'},
    'M2SMNXEDI_2_wetdays': {'unit': 'days/month', 'kind': 'event_count', 'aggregation': 'sum'},
    'M2SMNXEDI_2_CDD': {'unit': 'days', 'kind': 'monthly_max', 'aggregation': 'max'},
    'M2SMNXEDI_2_RX1Day': {'unit': 'mm/day', 'kind': 'monthly_max', 'aggregation': 'max'},
    'M2SMNXEDI_2_RX5Day': {'unit': 'mm/5 days', 'kind': 'monthly_max', 'aggregation': 'max'},
    'M2TMNXLND_5_12_4_GWETROOT': {'unit': 'dimensionless', 'kind': 'soil_wetness', 'aggregation': 'mean'},
    'M2TMNXLND_5_12_4_RZMC': {'unit': 'm3/m3', 'kind': 'soil_moisture', 'aggregation': 'mean'},
    'M2TMNXLND_5_12_4_SFMC': {'unit': 'm3/m3', 'kind': 'soil_moisture', 'aggregation': 'mean'},
    'M2SMNXSLV_5_12_4_T2MMAX': {'unit': 'K', 'kind': 'temperature', 'aggregation': 'mean'},
    'M2SMNXSLV_5_12_4_T2MMEAN': {'unit': 'K', 'kind': 'temperature', 'aggregation': 'mean'},
    'M2SMNXSLV_5_12_4_T2MMIN': {'unit': 'K', 'kind': 'temperature', 'aggregation': 'mean'},
}


def _region_from_filename(name: str) -> str:
    n = name.lower()
    if 'angat' in n:
        return 'angat_vicinity'
    if any(x in n for x in ('benguet', '-bag', 'precip-bag', 'surface-bag', 'wetness-bag')):
        return 'benguet_approximate'
    if n.endswith('-ph.csv'):
        return 'philippines_bbox'
    if any(x in n for x in ('-mm', 'ncr_', 'metro_manila')):
        return 'metro_manila_approximate'
    return 'unspecified'


def _read_header(path: Path) -> tuple[int, dict[str, str]]:
    metadata: dict[str, str] = {}
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        for i, row in enumerate(csv.reader(f)):
            if row and row[0].strip().lower() in ('time', 'system:index'):
                return i, metadata
            if len(row) >= 2 and row[0].strip():
                metadata[row[0].strip().rstrip(':')] = row[1].strip()
    raise ValueError(f'No supported table header in {path.name}')


def _fill_values(metadata: dict[str, str]) -> list[float]:
    result = []
    for key, val in metadata.items():
        if key.lower().startswith('fill value'):
            try:
                result.append(float(val))
            except ValueError:
                pass
    return result


def load_timeseries(path: str | Path, *, convert_temperature: bool = True) -> pd.DataFrame:
    """Load one original CSV. Returns date/value; attrs contain provenance.

    Raises ValueError for unknown data variables rather than silently assuming
    units or aggregation. Missing/fill observations remain NaN.
    """
    path = Path(path)
    header_row, source_meta = _read_header(path)
    raw = pd.read_csv(path, skiprows=header_row, encoding='utf-8-sig', low_memory=False)
    raw.columns = [str(c).strip() for c in raw.columns]
    is_ee = 'system:index' in raw.columns
    if is_ee:
        if 'date' not in raw.columns or 'rainfall_mm_month' not in raw.columns:
            raise ValueError(f'Unsupported Earth Engine export: {path.name}')
        date_col, value_col = 'date', 'rainfall_mm_month'
    else:
        if 'time' not in raw.columns:
            raise ValueError(f'Giovanni time column missing: {path.name}')
        candidates = [c for c in raw.columns if c != 'time']
        if len(candidates) != 1:
            raise ValueError(f'Expected one Giovanni variable, got {candidates}')
        date_col, value_col = 'time', candidates[0]
    variable = re.sub(r'^mean_', '', value_col)
    if variable not in VARIABLES:
        raise ValueError(f'Unknown variable {variable!r} in {path.name}; add metadata explicitly')
    spec = VARIABLES[variable]
    dates = pd.to_datetime(raw[date_col], errors='coerce', utc=True).dt.tz_localize(None)
    values = pd.to_numeric(raw[value_col], errors='coerce').astype(float)
    for fill in _fill_values(source_meta):
        values = values.mask(np.isclose(values, fill, rtol=1e-10, atol=1e-9))
    values = values.replace([np.inf, -np.inf], np.nan)
    values = values.mask(values.abs() > 1e12)
    unit = spec['unit']
    if convert_temperature and unit == 'K':
        values = values - 273.15
        unit = 'degC'
    out = pd.DataFrame({'date': dates, 'value': values})
    if out.date.isna().any():
        raise ValueError(f'Invalid date(s) in {path.name}')
    if out.date.duplicated().any():
        raise ValueError(f'Duplicate observation dates in {path.name}')
    out = out.sort_values('date').reset_index(drop=True)
    out.attrs = {
        'filename': path.name,
        'source_format': 'earth_engine' if is_ee else 'giovanni',
        'variable': variable,
        'unit': unit,
        'original_unit': spec['unit'],
        'kind': spec['kind'],
        'annual_aggregation': spec['aggregation'],
        'temporal_resolution': 'daily' if spec['kind'] == 'precipitation_daily' else 'monthly',
        'region': _region_from_filename(path.name),
        'requested_bbox': source_meta.get('User Bounding Box'),
        'data_bbox': source_meta.get('Data Bounding Box'),
        'reproduce_url': source_meta.get('URL to Reproduce Results'),
        'source_metadata': source_meta,
        'rows': len(out),
        'missing_values': int(out.value.isna().sum()),
        'first_date': out.date.min().strftime('%Y-%m-%d') if len(out) else None,
        'last_date': out.date.max().strftime('%Y-%m-%d') if len(out) else None,
    }
    return out


def list_datasets(raw_dir: str | Path = DEFAULT_RAW_DIR) -> list[dict[str, Any]]:
    """Inspect all CSVs and return metadata (including individual errors)."""
    result = []
    for path in sorted(Path(raw_dir).rglob('*.csv')):
        try:
            df = load_timeseries(path)
            result.append({'path': str(path.relative_to(raw_dir)), **{k: v for k, v in df.attrs.items() if k != 'source_metadata'}})
        except Exception as exc:
            result.append({'path': str(path.relative_to(raw_dir)), 'error': str(exc)})
    return result


def aggregate_annual(df: pd.DataFrame, *, complete_years_only: bool = True,
                     min_coverage: float = 0.95) -> pd.DataFrame:
    """Aggregate according to variable semantics; reject partial calendar years.

    Monthly extrema and maximum consecutive dry days are annual maxima of
    monthly indices, NOT necessarily maxima recomputed across year boundaries.
    """
    if not 0 < min_coverage <= 1:
        raise ValueError('min_coverage must be in (0, 1]')
    meta = df.attrs
    rule = meta.get('annual_aggregation')
    if rule not in ('sum', 'mean', 'max'):
        raise ValueError('Unknown annual aggregation rule')
    series = df.set_index('date')['value'].sort_index()
    grouped = series.resample('YS')
    values = getattr(grouped, rule)()
    observed = grouped.count()
    years = values.index.year
    if meta.get('temporal_resolution') == 'daily':
        expected = pd.Series([366 if pd.Timestamp(y, 12, 31).is_leap_year else 365 for y in years], index=values.index)
    else:
        expected = pd.Series(12, index=values.index)
    coverage = observed / expected
    result = pd.DataFrame({'date': values.index, 'value': values.to_numpy(),
                           'observed': observed.to_numpy(), 'expected': expected.to_numpy(),
                           'coverage': coverage.to_numpy()})
    result.loc[result.coverage < min_coverage, 'value'] = np.nan
    if complete_years_only:
        result = result[result.coverage >= min_coverage].copy()
    result.reset_index(drop=True, inplace=True)
    result.attrs = {**meta, 'temporal_resolution': 'annual', 'aggregation_applied': rule,
                    'coverage_threshold': min_coverage,
                    'note': 'Monthly maximum indices may miss events spanning month boundaries.' if rule == 'max' else None}
    return result
