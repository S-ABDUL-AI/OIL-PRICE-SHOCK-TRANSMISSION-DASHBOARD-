"""
Oil Price Shock Transmission — data generation and econometric models.
MSc Economics Thesis, KNUST 2022.
"""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

warnings.filterwarnings("ignore")

VARIABLE_LABELS = {
    "log_pub": "Public Debt (log)",
    "log_ext": "External Debt (log)",
    "log_dom": "Domestic Debt (log)",
    "log_oil": "Oil Price (log)",
    "log_infl": "Inflation (log)",
    "log_reer": "REER (log)",
    "log_ir": "Interest Rate (log)",
}

LEVEL_COLS = [
    "public_debt",
    "domestic_debt",
    "external_debt",
    "oil_price",
    "inflation",
    "reer",
    "interest_rate",
]

LOG_COLS = [
    "log_pub",
    "log_ext",
    "log_dom",
    "log_oil",
    "log_infl",
    "log_reer",
    "log_ir",
]

DEBT_MODELS = {
    "Public Debt": {
        "debt_col": "log_pub",
        "cols": ["log_pub", "log_oil", "log_infl", "log_reer", "log_ir"],
    },
    "External Debt": {
        "debt_col": "log_ext",
        "cols": ["log_ext", "log_oil", "log_infl", "log_reer", "log_ir"],
    },
    "Domestic Debt": {
        "debt_col": "log_dom",
        "cols": ["log_dom", "log_oil", "log_infl", "log_reer", "log_ir"],
    },
}

VAR_DISPLAY_NAMES = {
    "log_pub": "Public Debt",
    "log_ext": "External Debt",
    "log_dom": "Domestic Debt",
    "log_oil": "Oil Price ★",
    "log_infl": "Inflation",
    "log_reer": "REER",
    "log_ir": "Interest Rate",
}


def generate_ghana_data(seed=13):
    """
    Calibrated synthetic replica of Ghana annual macro data 1983–2019.

    Returns a single DataFrame with level and log columns indexed by year.
    """
    np.random.seed(seed)
    years = np.arange(1983, 2020)
    n = len(years)

    oil_raw = np.array([
        29.5, 28.1, 27.5, 14.4, 18.4, 16.8, 14.9, 17.9, 18.6, 22.3,
        19.5, 18.6, 20.0, 15.9, 12.7, 17.9, 28.5, 24.1, 25.0, 28.9,
        31.0, 38.3, 54.5, 65.1, 72.4, 98.5, 62.0, 79.6, 111.3, 111.9,
        111.6, 108.7, 99.0, 52.4, 44.0, 54.2, 64.3,
    ])
    log_oil = np.log(oil_raw)

    phi = 0.45
    ec_pub = np.zeros(n)
    for t in range(1, n):
        ec_pub[t] = phi * ec_pub[t - 1] + np.random.normal(0, 0.04)
    log_public_debt = -0.35 * log_oil + ec_pub + 5.2

    ec_ext = np.zeros(n)
    for t in range(1, n):
        ec_ext[t] = phi * ec_ext[t - 1] + np.random.normal(0, 0.03)
    log_external_debt = 0.30 * log_oil + ec_ext + 2.3

    log_domestic_debt = np.zeros(n)
    log_domestic_debt[0] = 3.5
    for t in range(1, n):
        log_domestic_debt[t] = log_domestic_debt[t - 1] + np.random.normal(0, 0.04)

    log_inflation = np.zeros(n)
    log_inflation[0] = 4.8
    for t in range(1, n):
        log_inflation[t] = log_inflation[t - 1] + np.random.normal(-0.070, 0.08)

    log_reer = np.zeros(n)
    log_reer[0] = 5.1
    for t in range(1, n):
        log_reer[t] = log_reer[t - 1] + np.random.normal(-0.025, 0.04)

    log_interest = np.zeros(n)
    log_interest[0] = 3.2
    for t in range(1, n):
        log_interest[t] = log_interest[t - 1] + np.random.normal(-0.015, 0.05)

    return pd.DataFrame(
        {
            "public_debt": np.exp(log_public_debt) * 100,
            "domestic_debt": np.exp(log_domestic_debt) * 100,
            "external_debt": np.exp(log_external_debt) * 100,
            "oil_price": oil_raw,
            "inflation": np.exp(log_inflation),
            "reer": np.exp(log_reer),
            "interest_rate": np.exp(log_interest),
            "log_pub": log_public_debt,
            "log_ext": log_external_debt,
            "log_dom": log_domestic_debt,
            "log_oil": log_oil,
            "log_infl": log_inflation,
            "log_reer": log_reer,
            "log_ir": log_interest,
        },
        index=years,
    )


def run_adf_tests(df):
    """ADF unit root tests on levels and first differences for all log variables."""
    rows = []
    for col, label in VARIABLE_LABELS.items():
        ser = df[col].dropna().values
        lvl = adfuller(ser, autolag="AIC")
        dif = adfuller(np.diff(ser), autolag="AIC")
        rows.append({
            "Variable": label,
            "ADF (Level)": round(lvl[0], 3),
            "p (Level)": round(lvl[1], 4),
            "I(0)?": "Yes" if lvl[1] < 0.05 else "No",
            "ADF (Δ)": round(dif[0], 3),
            "p (Δ)": round(dif[1], 4),
            "I(1)?": "Yes" if dif[1] < 0.05 else "No",
        })
    return pd.DataFrame(rows)


def run_johansen(df, cols):
    """Johansen cointegration test for a single debt model."""
    data = df[cols].dropna().values
    joh = coint_johansen(data, det_order=0, k_ar_diff=1)
    n_cv = sum(joh.lr1 > joh.cvt[:, 1])
    trace_rows = []
    for i in range(min(3, len(joh.lr1))):
        trace_rows.append({
            "H₀: rank ≤ r": i,
            "Trace Statistic": round(joh.lr1[i], 3),
            "5% Critical Value": round(joh.cvt[i, 1], 3),
            "Max-Eigen Stat": round(joh.lr2[i], 3),
            "5% CV (Max-Eigen)": round(joh.cvm[i, 1], 3),
            "Reject H₀?": "Yes ✓" if joh.lr1[i] > joh.cvt[i, 1] else "No",
        })
    return {
        "cointegrating_vectors": n_cv,
        "trace_table": pd.DataFrame(trace_rows),
        "eigenvectors": joh.evec,
    }


def extract_irf(var_model, periods, debt_idx=0, oil_idx=1):
    """Orthogonalized IRF of debt response to oil shock with approximate CI."""
    irf_obj = var_model.irf(periods)
    orth_irfs = irf_obj.orth_irfs[:, debt_idx, oil_idx]
    resid_std = np.std(np.asarray(var_model.resid)[:, debt_idx])
    ci_half = resid_std * np.sqrt(np.arange(periods + 1)) * 0.30
    return pd.DataFrame({
        "period": np.arange(periods + 1),
        "orth_irf": orth_irfs,
        "ci_upper": orth_irfs + ci_half,
        "ci_lower": orth_irfs - ci_half,
    })


def extract_fevd(var_model, periods, debt_idx=0, oil_idx=1):
    """Forecast error variance decomposition for debt variable."""
    fevd_obj = var_model.fevd(periods)
    n_vars = fevd_obj.decomp.shape[0]
    rows = []
    for t in range(periods):
        for j in range(n_vars):
            rows.append({
                "period": t + 1,
                "variable_idx": j,
                "pct": fevd_obj.decomp[debt_idx, t, j] * 100,
            })
    return pd.DataFrame(rows)


def fit_var_models(df, max_lags=2, irf_periods=10):
    """Fit VAR(p) for all three debt models; extract IRF and FEVD."""
    out = {}
    for model_name, cfg in DEBT_MODELS.items():
        cols = cfg["cols"]
        df_m = df[cols].dropna()
        var_fit = VAR(df_m).fit(maxlags=max_lags, ic="aic")
        irf_df = extract_irf(var_fit, irf_periods)
        fevd_df = extract_fevd(var_fit, irf_periods)
        fevd_obj = var_fit.fevd(irf_periods)
        oil_pct = fevd_obj.decomp[0, :, 1] * 100
        fevd_final = fevd_obj.decomp[0, -1, :] * 100
        out[model_name] = {
            "var_results": var_fit,
            "lag_order": var_fit.k_ar,
            "aic": round(var_fit.aic, 3),
            "bic": round(var_fit.bic, 3),
            "irf": irf_df,
            "fevd": fevd_df,
            "orth_irfs": irf_df["orth_irf"].values,
            "ci_upper": irf_df["ci_upper"].values,
            "ci_lower": irf_df["ci_lower"].values,
            "fevd_oil": oil_pct,
            "fevd_final": fevd_final,
            "var_names": cols,
        }
    return out


def run_all_johansen(df):
    """Run Johansen tests for all debt models."""
    return {name: run_johansen(df, cfg["cols"]) for name, cfg in DEBT_MODELS.items()}
