# Oil Price Shock Transmission Dashboard
### Interactive Econometric Tool — MSc Economics Thesis, KNUST 2022

**Built by Sherriff Abdul-Hamid**  
Development economist and data scientist — USAID · UNDP · UKAID · Obama Foundation Leaders Award (Top 1.3%)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://oil-shock-transmission.streamlit.app)

---

## The Research Question

> *Do oil price shocks have a statistically significant long-run relationship with Ghana's public debt — and if so, through which channel?*

This dashboard makes my **MSc Economics thesis** (*Oil Price Shocks and Public Debt: Empirical Evidence from Ghana*, KNUST 2022) fully interactive. Users can explore the full econometric pipeline — unit root tests, Johansen cointegration, impulse response functions, and variance decompositions — across three debt models using Ghana annual time-series data 1983–2019.

This is the **only SCE application portfolio** that turns original peer-reviewed academic research into a live, interactive econometric tool.

---

## Thesis Summary

| Item | Detail |
|---|---|
| **Institution** | Kwame Nkrumah University of Science and Technology (KNUST), Ghana |
| **Year** | 2022 |
| **Methods** | Johansen Cointegration, VECM, IRF, FEVD |
| **Data** | Ghana annual macro series 1983–2019 (37 observations) |
| **Variables** | Public debt, domestic debt, external debt, oil price (Brent), inflation, REER, 91-day T-bill rate |

### Key Findings

| Finding | Result |
|---|---|
| All variables integrated | I(1) — confirmed via ADF test |
| Public debt cointegrated with oil | Yes — 1 cointegrating vector (Johansen, 5%) |
| External debt cointegrated with oil | Yes — 1 cointegrating vector (Johansen, 5%) |
| Domestic debt cointegrated with oil | No — consistent with absence of external channel |
| Public debt IRF to oil shock | **Negative in periods 1–3, then recovers** |
| Oil → % of public debt variance (FEVD) | **~1.4%** at 10-period horizon |
| Oil → % of external debt variance (FEVD) | **~4.4%** at 10-period horizon |

### Economic Interpretation

A positive oil price shock **initially reduces** Ghana's public debt — counterintuitive for an oil importer, but explained by a **terms-of-trade mechanism**: rising oil prices are accompanied by commodity price co-movement (cocoa, gold), temporarily improving Ghana's fiscal position and reducing borrowing need. The effect reverses as import costs dominate the medium run.

External debt responds **more persistently** to oil shocks, consistent with current account pressure from rising import bills.

Domestic debt shows **no systematic response**, confirming that the transmission channel is external (balance of payments, exchange rate) rather than domestic.

---

## App Structure — Four Tabs

| Tab | Content |
|---|---|
| **Ghana Macro Overview** | Time series for all 7 variables · Oil price history with annotated shocks (1986 OPEC collapse, 1990 Gulf War, 1998 Asian Crisis, 2008 GFC, 2014 shale surge) · Pairwise correlation heatmap |
| **Unit Root & Cointegration** | ADF test results table (all 7 variables, levels and first differences) · Visual stationarity check · Johansen trace/max-eigenvalue tests for all 3 debt models |
| **Impulse Response Functions** | Orthogonalized IRF (Cholesky) for all 3 debt models · 10-period horizon with 90% approximate confidence bands · Overlay comparison chart · Economic interpretation |
| **Variance Decomposition** | Stacked FEVD bar charts · Oil contribution line charts · Summary table comparing results to thesis benchmarks · Policy interpretation including SCE application |

---

## Econometric Pipeline

```python
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# Step 1: Unit root tests
adf_result = adfuller(series, autolag="AIC")

# Step 2: Johansen cointegration
joh = coint_johansen(df_levels, det_order=0, k_ar_diff=1)
n_cointegrating_vectors = sum(joh.lr1 > joh.cvt[:, 1])  # at 5%

# Step 3: VAR estimation
var_fit = VAR(df_log).fit(maxlags=2, ic="aic")

# Step 4: Orthogonalized IRF (Cholesky)
irf = var_fit.irf(10)
orth_irfs = irf.orth_irfs[:, debt_idx, oil_idx]   # shape (11,)

# Step 5: Forecast Error Variance Decomposition
fevd = var_fit.fevd(10)
oil_contribution = fevd.decomp[debt_idx, :, oil_idx] * 100   # % per period
```

---

## Data Description

| Variable | Source | Unit | Notes |
|---|---|---|---|
| Public debt | World Bank / IMF | % of GDP | General government gross debt |
| Domestic debt | Bank of Ghana | % of GDP | Domestic public debt |
| External debt | World Bank | % of GDP | External public debt |
| Oil price | EIA / IMF | USD/barrel | Brent crude annual average |
| Inflation | IMF IFS | % change | Consumer price inflation |
| REER | BIS / IMF | Index (2010=100) | Real effective exchange rate |
| Interest rate | Bank of Ghana | % | 91-day T-bill rate |

---

## Repository Structure

```
├── app.py                   # Main Streamlit application (all 4 tabs)
├── requirements.txt         # Runtime dependencies
└── README.md                # This file
```

---

## Run Locally

```bash
git clone https://github.com/S-ABDUL-AI/oil-shock-transmission.git
cd oil-shock-transmission
pip install -r requirements.txt
streamlit run app.py
```

---

## Replication Note

The app uses **calibrated synthetic data** that approximates the Ghana macro series used in the original thesis. The data-generating process (DGP) is designed to reproduce:

- I(1) integration order for all variables (ADF-confirmed)
- Cointegrating vectors for public and external debt models (Johansen)
- Negative short-run IRF for public debt (consistent with thesis periods 1–3)
- FEVD magnitudes broadly consistent with thesis benchmarks (~1.4% / ~4.4%)

Full numerical replication of thesis results requires the original IMF/World Bank dataset. Contact the author for the data request procedure.

---

## SCE Relevance

This research framework directly applies to Southern California Edison's analytical challenges:

| Thesis concept | SCE analogue |
|---|---|
| Oil price shock → debt variance | Fuel cost shock → grid investment cost variance |
| Johansen cointegration | Long-run equilibrium between energy prices and capex |
| IRF: short-run vs long-run response | Rate case timing: when do cost shocks feed through to customers? |
| FEVD: oil vs own-shock share | How much of capex variance is commodity-driven vs regulatory? |
| VECM error correction | Cost reversion mechanism after fuel price spikes |

The same methodology — adapted to CAISO nodal pricing, SCE fuel procurement data, and CPUC rate case cycles — directly supports the **Energy Policy and Pricing Senior Advisor** (Job ID 4818) and **Senior Data Science Manager** (Job ID 6279) roles.

---

## About the Author

**Sherriff Abdul-Hamid** is a development economist and data scientist with 10+ years of experience in quantitative research, policy analytics, and decision-support tools for public-sector and energy programs.

- MSc Economics (Econometrics), KNUST
- Harvard Business School · Senior Executive Program
- Former Founder & CEO, Poverty 360 — 25,000+ beneficiaries
- Directed $200M+ in resource allocation for USAID, UNDP, UKAID
- **Obama Foundation Leaders Award** — Top 1.3% globally, 2023
- **Mandela Washington Fellow** — Top 0.3%, U.S. Dept. of State, 2018

**Connect:**  
[LinkedIn](https://www.linkedin.com/in/abdul-hamid-sherriff-08583354/) ·
[GitHub](https://github.com/S-ABDUL-AI) ·
[Portfolio](https://share.streamlit.io/user/s-abdul-ai)

---

## Related Portfolio Tools

| Tool | Description |
|---|---|
| [California Grid Asset Risk Dashboard](https://share.streamlit.io/user/s-abdul-ai) | Wildfire vulnerability and asset risk scoring for SCE grid infrastructure |
| [Grid Investment Prioritization Engine](https://share.streamlit.io/user/s-abdul-ai) | Capital allocation optimization for grid modernization programs |
| [Public Budget Allocation Tool](https://smart-resource-allocation-dashboard-eudzw5r2f9pbu4qyw3psez.streamlit.app) | Need-based government resource distribution with ministerial brief |
| [Medicaid Access Risk Monitor](https://chpghrwawmvddoquvmniwm.streamlit.app) | ML-powered healthcare coverage gap analysis across 50 US states |
| [GovFund Allocation Engine](https://impact-allocation-engine-ahxxrbgwmvyapwmifahk2b.streamlit.app) | Cost-effectiveness decision tool for public health funders |
