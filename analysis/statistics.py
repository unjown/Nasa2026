"""Trend estimates and explicit diagnostic warnings (not autocorrelation-corrected)."""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr

def _lag1(y):
    y = np.asarray(y, dtype=float)
    if len(y) < 4 or np.std(y[:-1]) == 0 or np.std(y[1:]) == 0:
        return None
    return float(np.corrcoef(y[:-1], y[1:])[0, 1])

def lag1_autocorrelation(df):
    d = df[['date','value']].dropna().sort_values('date')
    return _lag1(d['value'].to_numpy())

def analyze_trend(df, *, alpha=0.05, min_observations=8):
    if not 0 < alpha < 1:
        raise ValueError('alpha must be between 0 and 1')
    d = df[['date','value']].copy()
    d['date'] = pd.to_datetime(d['date'], errors='coerce')
    d['value'] = pd.to_numeric(d['value'], errors='coerce')
    d = d.replace([np.inf,-np.inf], np.nan).dropna().sort_values('date')
    if d['date'].duplicated().any():
        raise ValueError('Duplicate dates')
    if len(d) < max(3,min_observations):
        raise ValueError('Insufficient observations')
    x = (d['date'] - d['date'].iloc[0]).dt.total_seconds().to_numpy() / (365.2425*86400)
    y = d['value'].to_numpy(dtype=float)
    if np.ptp(y) == 0:
        raise ValueError('Constant time series')
    fit = stats.linregress(x,y)
    tau, p = stats.kendalltau(x,y)
    critical = stats.t.ppf(1-alpha/2, len(d)-2)
    ci = [float(fit.slope-critical*fit.stderr), float(fit.slope+critical*fit.stderr)]
    sen = stats.theilslopes(y,x,alpha=1-alpha)
    residual = y-(fit.intercept+fit.slope*x)
    residual_ac = _lag1(residual)
    threshold = float(1.96/np.sqrt(len(y)))
    flag = residual_ac is not None and abs(residual_ac)>threshold
    warnings = []
    if flag:
        warnings.append('Residual serial correlation detected; unadjusted p-values and confidence intervals may be unreliable.')
    warnings.append('Historical trends do not establish causation or predict environmental hazards.')
    return {
        'n':int(len(d)), 'start_date':d['date'].iloc[0].strftime('%Y-%m-%d'),
        'end_date':d['date'].iloc[-1].strftime('%Y-%m-%d'),
        'unit':df.attrs.get('unit','unknown'),
        'slope_per_year':float(fit.slope), 'slope_per_decade':float(fit.slope*10),
        'intercept':float(fit.intercept), 'r_squared':float(fit.rvalue**2),
        'ols_p_value':float(fit.pvalue), 'slope_ci':ci,
        'slope_ci_per_year':ci, 'slope_ci_per_decade':[v*10 for v in ci],
        'kendall_tau':float(tau), 'mann_kendall_p_value':float(p),
        'significant_mann_kendall':bool(p<alpha),
        'theil_sen_slope_per_year':float(sen.slope),
        'theil_sen_slope_per_decade':float(sen.slope*10),
        'raw_lag1_autocorrelation':_lag1(y),
        'residual_lag1_autocorrelation':residual_ac,
        'autocorrelation_threshold':threshold, 'autocorrelation_flag':bool(flag),
        'status':'preliminary_autocorrelation_warning' if flag else 'preliminary_assumptions_not_fully_verified',
        'alpha':alpha, 'warnings':warnings, 'warning':' '.join(warnings)
    }

# Correlation Analysis

def calculate_correlation(df1, df2):
    merged = df1[["date", "value"]].merge(
        df2[["date", "value"]],
        on="date",
        how="inner",
        suffixes=("_1", "_2")
    ).dropna()

    if len(merged) < 3:
        raise ValueError(
            "At least 3 overlapping observations are required"
        )

    x = merged["value_1"].to_numpy(dtype=float)
    y = merged["value_2"].to_numpy(dtype=float)

    if not (
        np.isfinite(x).all() and
        np.isfinite(y).all()
    ):
        raise ValueError("Non-finite observations detected")

    if np.ptp(x) == 0 or np.ptp(y) == 0:
        raise ValueError(
            "Correlation is undefined for constant data"
        )

    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)

    return {
        "observations": len(merged),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "first_year": int(merged["date"].dt.year.min()),
        "last_year": int(merged["date"].dt.year.max()),
        "status": "exploratory_unadjusted",
        "warnings": [
            "Correlation does not establish causation.",
            "P-values are unadjusted for temporal autocorrelation.",
            "Shared long-term trends may inflate correlations."
        ]
    }
