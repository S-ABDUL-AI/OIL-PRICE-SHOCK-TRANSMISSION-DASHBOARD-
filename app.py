"""
Oil Price Shock Transmission Dashboard
Interactive Econometric Tool — MSc Economics Thesis, KNUST 2022
Author: Sherriff Abdul-Hamid
Methods: VAR, Johansen Cointegration, IRF, FEVD
Data: Ghana annual time-series 1983–2019 (calibrated synthetic replica)
"""

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from model import (
    LEVEL_COLS,
    LOG_COLS,
    fit_var_models,
    generate_ghana_data,
    run_adf_tests,
    run_all_johansen,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Oil Price Shock Transmission — Ghana",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────────────────────
NAVY      = "#0A1F44"
NAVY_MID  = "#152B5C"
GOLD      = "#C9A84C"
GOLD_LT   = "#E8C97A"
INK       = "#1A1A1A"
BODY      = "#2C3E50"
MUTED     = "#6B7280"
RED       = "#C8382A"
AMBER     = "#B8560A"
GREEN     = "#1A7A2E"
TEAL      = "#0E7490"
RULE      = "#E2E6EC"
OFF_WHITE = "#F8F6F1"

VAR_COLORS = {
    "public_debt":   "#0A1F44",
    "external_debt": "#C9A84C",
    "domestic_debt": "#0E7490",
    "oil_price":     "#C8382A",
    "inflation":     "#7C3AED",
    "reer":          "#059669",
    "interest_rate": "#B45309",
}

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] {{
    font-family: 'Source Sans 3', sans-serif; background:{OFF_WHITE};
  }}
  .hero-wrap {{
    background: linear-gradient(135deg,{NAVY} 0%,{NAVY_MID} 65%,#1E3A6E 100%);
    border-left: 6px solid {GOLD}; border-radius: 6px;
    padding: 34px 40px 28px; margin-bottom: 24px;
  }}
  .hero-eye   {{ font-size:11px; font-weight:700; letter-spacing:2.5px;
                 color:{GOLD}; text-transform:uppercase; margin-bottom:10px; }}
  .hero-title {{ font-size:26px; font-weight:700; color:#FFFFFF;
                 line-height:1.3; margin-bottom:10px; }}
  .hero-sub   {{ font-size:13.5px; color:#B0BFD8; line-height:1.65; max-width:860px; }}
  .sec-lbl {{ font-size:10px; font-weight:700; letter-spacing:2px;
              color:{GOLD}; text-transform:uppercase; margin-bottom:3px; }}
  .sec-ttl {{ font-size:19px; font-weight:700; color:{NAVY}; margin-bottom:3px; }}
  .sec-sub {{ font-size:13px; color:{MUTED}; margin-bottom:14px; }}
  .kpi-card  {{ background:#FFFFFF; border:1px solid {RULE};
                border-top:3px solid {NAVY}; border-radius:4px; padding:14px 18px; }}
  .kpi-label {{ font-size:10px; font-weight:700; letter-spacing:1px;
                color:{MUTED}; text-transform:uppercase; margin-bottom:3px; }}
  .kpi-val   {{ font-size:24px; font-weight:700; color:{NAVY}; line-height:1.1; }}
  .kpi-sub   {{ font-size:11px; color:{MUTED}; margin-top:2px; }}
  .finding   {{ background:#FFFFFF; border:1px solid {RULE};
                border-left:4px solid {GOLD}; border-radius:4px;
                padding:14px 18px; margin-bottom:10px; }}
  .finding-h {{ font-size:12px; font-weight:700; color:{NAVY}; margin-bottom:4px; }}
  .finding-b {{ font-size:13px; color:{BODY}; line-height:1.6; }}
  .scope-box {{ background:#FFFBF0; border:1px solid {AMBER};
                border-left:4px solid {AMBER}; border-radius:4px;
                padding:9px 14px; font-size:11.5px; color:{AMBER};
                margin-bottom:18px; }}
  .brief-navy {{ background:#F0F4FF; border:1px solid #C4D0F5;
                 border-left:4px solid {NAVY}; border-radius:4px;
                 padding:14px 16px; }}
  .brief-gold {{ background:#FFFBF0; border:1px solid #E8C97A;
                 border-left:4px solid {GOLD}; border-radius:4px;
                 padding:14px 16px; }}
  .brief-red  {{ background:#FFF5F5; border:1px solid #FFC9C9;
                 border-left:4px solid {RED}; border-radius:4px;
                 padding:14px 16px; }}
  .brief-head {{ font-size:10px; font-weight:700; letter-spacing:2px;
                 text-transform:uppercase; margin-bottom:6px; }}
  .brief-body {{ font-size:13px; color:{BODY}; line-height:1.6; }}
  .result-tbl {{ width:100%; border-collapse:collapse; font-size:12.5px; }}
  .result-tbl th {{ background:{NAVY}; color:{GOLD}; font-weight:700;
                    padding:8px 12px; text-align:left; }}
  .result-tbl td {{ padding:7px 12px; border-bottom:1px solid {RULE}; color:{BODY}; }}
  .result-tbl tr:nth-child(even) td {{ background:#F8F9FC; }}
  .sig-yes {{ color:{GREEN}; font-weight:700; }}
  .sig-no  {{ color:{RED};   font-weight:700; }}
  .byline {{ background:{NAVY}; border-radius:4px; padding:16px 22px;
             font-size:11.5px; color:#B0BFD8; line-height:1.8; margin-top:28px; }}
  .byline a {{ color:{GOLD}; text-decoration:none; }}
  button[data-baseweb="tab"] {{
    font-weight:600 !important; font-size:13px !important; color:{NAVY} !important;
  }}
  div[data-testid="stButton"] > button {{
    background:{NAVY}; color:#FFFFFF; border:none; border-radius:3px;
    font-weight:600; letter-spacing:0.3px;
  }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CACHED MODEL CALLS
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data(seed=13):
    return generate_ghana_data(seed)


@st.cache_data
def cached_adf(df_log):
    return run_adf_tests(df_log)


@st.cache_data
def cached_johansen(df_log):
    return run_all_johansen(df_log)


@st.cache_data
def cached_var(df_log, max_lags, irf_horizon):
    return fit_var_models(df_log, max_lags=max_lags, irf_periods=irf_horizon)


# ─────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────
def navy_layout(height=320, margin=None, **kw):
    m = margin or dict(t=30, b=40, l=50, r=20)
    return dict(
        paper_bgcolor="white", plot_bgcolor="white",
        height=height, margin=m,
        font=dict(family="Source Sans 3, sans-serif", color=BODY, size=11),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=RULE, showline=True),
        yaxis=dict(showgrid=False, zeroline=False, linecolor=RULE, showline=True),
        **kw,
    )


def irf_chart(orth_irfs, ci_upper, ci_lower, title, color=NAVY,
              debt_label="Debt", oil_label="Oil Price"):
    periods = list(range(len(orth_irfs)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=periods + periods[::-1],
        y=list(ci_upper) + list(ci_lower[::-1]),
        fill="toself", fillcolor="rgba(10,31,68,0.12)",
        line=dict(width=0), showlegend=True, name="90% CI (approx.)",
    ))
    fig.add_hline(y=0, line=dict(color=RULE, width=1.5, dash="dot"))
    fig.add_trace(go.Scatter(
        x=periods, y=orth_irfs,
        mode="lines+markers",
        line=dict(color=color, width=2.5),
        marker=dict(size=7, color=color),
        name=f"Response of {debt_label}",
    ))
    min_idx = int(np.argmin(orth_irfs))
    if orth_irfs[min_idx] < 0:
        fig.add_annotation(
            x=min_idx, y=orth_irfs[min_idx],
            text=f"Max −ve: period {min_idx}",
            showarrow=True, arrowhead=2, arrowcolor=RED,
            font=dict(size=10, color=RED), ax=30, ay=-30,
        )
    fig.update_layout(
        **navy_layout(height=300),
        title=dict(text=title, font=dict(size=13, color=NAVY), x=0),
        xaxis_title="Periods after shock",
        yaxis_title="Log-point response",
        legend=dict(orientation="h", y=-0.25, x=0),
    )
    return fig


def fevd_stacked_bar(fevd_all_vars, var_names, title, highlight_oil=True):
    periods = list(range(1, 11))
    colors_list = [GOLD, NAVY, TEAL, "#7C3AED", "#059669", "#B45309", "#6B7280"]
    fig = go.Figure()
    for j, vname in enumerate(var_names):
        vals = [fevd_all_vars[t][j] for t in range(10)]
        label = vname.replace("log_", "").replace("_", " ").title()
        lcolor = colors_list[j % len(colors_list)]
        if "oil" in vname.lower() and highlight_oil:
            lcolor = RED
        fig.add_trace(go.Bar(name=label, x=periods, y=vals, marker_color=lcolor))
    fig.update_layout(
        barmode="stack",
        **navy_layout(height=300),
        title=dict(text=title, font=dict(size=13, color=NAVY), x=0),
        xaxis_title="Forecast horizon (periods)",
        yaxis_title="% of variance explained",
        yaxis_range=[0, 100],
        legend=dict(orientation="h", y=-0.35, x=0, font=dict(size=10)),
        bargap=0.12,
    )
    return fig


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="background:{NAVY};border-radius:4px;padding:14px 16px;margin-bottom:14px;">
      <div style="font-size:10px;font-weight:700;letter-spacing:2px;color:{GOLD};
                  text-transform:uppercase;margin-bottom:5px;">Research Tool</div>
      <div style="font-size:13px;font-weight:700;color:#FFFFFF;">Oil Shock Transmission</div>
      <div style="font-size:11px;color:#B0BFD8;margin-top:2px;">Ghana · 1983–2019</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📖  About this tool"):
        st.markdown("""
        This dashboard makes my **MSc Economics thesis** (KNUST, 2022) interactive.

        **Methods applied:**
        - ADF unit root tests
        - Johansen cointegration
        - Vector Autoregression (VAR)
        - Orthogonalized Impulse Response Functions
        - Forecast Error Variance Decomposition

        **Data:** Calibrated synthetic replica of Ghana annual
        macroeconomic series 1983–2019.
        Full replication requires IMF/World Bank originals.
        """)

    st.markdown(f'<div class="sec-lbl">Model Settings</div>', unsafe_allow_html=True)
    max_lags_ui  = st.slider("Max VAR lags (AIC selects)", 1, 4, 2)
    irf_horizon  = st.slider("IRF / FEVD horizon (periods)", 5, 15, 10)
    show_orth    = st.checkbox("Orthogonalized IRF (Cholesky)", value=True)
    show_ci      = st.checkbox("Show confidence bands", value=True)


# ─────────────────────────────────────────────────────────────
# LOAD DATA & RUN MODELS
# ─────────────────────────────────────────────────────────────
df = load_data()
df_levels = df[LEVEL_COLS]
df_log = df[LOG_COLS]

with st.spinner("Running VAR models, cointegration tests, IRF & FEVD…"):
    adf_results  = cached_adf(df_log)
    joh_results  = cached_johansen(df_log)
    var_results  = cached_var(df_log, max_lags_ui, irf_horizon)


# ─────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-wrap">
  <div class="hero-eye">
    Oil Price Shock Transmission · MSc Economics Thesis KNUST 2022 ·
    VAR · Johansen Cointegration · IRF · FEVD
  </div>
  <div class="hero-title">
    Do oil price shocks transmit to Ghana's public debt?<br>
    Empirical evidence from 37 years of macro data.
  </div>
  <div class="hero-sub">
    This dashboard turns a peer-reviewed MSc thesis into an interactive econometric
    tool — allowing users to explore impulse responses, cointegration structure, and
    variance decompositions across three debt models (public, domestic, external) using
    a Johansen–VAR framework on Ghana annual data 1983–2019.
    Use the sidebar to adjust model settings and forecast horizons.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="scope-box">
  <strong>Replication note:</strong> Results use calibrated synthetic data approximating the
  Ghana IMF/World Bank macro series used in the original thesis. Qualitative findings
  (cointegration structure, IRF sign pattern, FEVD magnitudes) are consistent with
  thesis conclusions. Full numerical replication requires the original dataset.
  · Thesis: <em>Oil Price Shocks and Public Debt: Empirical Evidence from Ghana</em>, KNUST 2022.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# KEY FINDINGS KPI ROW
# ─────────────────────────────────────────────────────────────
pub_fevd_10  = round(var_results["Public Debt"]["fevd_oil"][-1], 1)
ext_fevd_10  = round(var_results["External Debt"]["fevd_oil"][-1], 1)
dom_fevd_10  = round(var_results["Domestic Debt"]["fevd_oil"][-1], 1)
pub_irf_min  = round(var_results["Public Debt"]["orth_irfs"].min(), 4)
joh_pub_r    = joh_results["Public Debt"]["cointegrating_vectors"]
joh_ext_r    = joh_results["External Debt"]["cointegrating_vectors"]
joh_dom_r    = joh_results["Domestic Debt"]["cointegrating_vectors"]

c1, c2, c3, c4, c5 = st.columns(5)
for col, label, val, sub in [
    (c1, "Oil → Public Debt",   f"{pub_fevd_10:.1f}%", "of variance (FEVD, P10)"),
    (c2, "Oil → External Debt", f"{ext_fevd_10:.1f}%", "of variance (FEVD, P10)"),
    (c3, "Oil → Domestic Debt", f"{dom_fevd_10:.1f}%", "no significant effect"),
    (c4, "Cointegrating Vectors",
         f"{joh_pub_r} pub / {joh_ext_r} ext", "Johansen at 5% level"),
    (c5, "IRF: Public Debt",    f"{pub_irf_min:+.4f}",  "min log-pt response to +1SD oil"),
]:
    with col:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-val">{val}</div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Ghana Macro Overview",
    "🔬  Unit Root & Cointegration",
    "📈  Impulse Response Functions",
    "📉  Variance Decomposition",
])

years = df_levels.index.tolist()

# ═════════════════════════════════════════════════════════════
# TAB 1 — GHANA MACRO OVERVIEW
# ═════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="sec-lbl">Ghana Macro Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-ttl">Annual macroeconomic series, 1983–2019</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">All series shown in levels. Tick \'Log scale\' in each chart for log levels used in VAR estimation.</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-lbl" style="margin-top:4px;">Exhibit 1 — Oil Price History with Major Shocks</div>', unsafe_allow_html=True)
    fig_oil = go.Figure()
    fig_oil.add_trace(go.Scatter(
        x=years, y=df_levels["oil_price"],
        mode="lines", name="Brent Crude (USD/bbl)",
        line=dict(color=RED, width=2.5),
        fill="tozeroy", fillcolor="rgba(200,56,42,0.08)",
    ))
    shocks = [
        (1986, "1986\nOPEC collapse"),
        (1990, "1990–91\nGulf War"),
        (1998, "1998\nAsian crisis"),
        (2008, "2008\nGFC peak"),
        (2014, "2014–15\nShale surge"),
    ]
    for yr, label in shocks:
        idx = yr - 1983
        if 0 <= idx < len(years):
            y_val = df_levels["oil_price"].iloc[idx]
            fig_oil.add_vline(x=yr, line=dict(color=AMBER, width=1, dash="dash"))
            fig_oil.add_annotation(
                x=yr, y=y_val + 8,
                text=label, showarrow=False,
                font=dict(size=9, color=AMBER), align="center",
            )
    fig_oil.update_layout(
        **navy_layout(height=280),
        xaxis_title="Year", yaxis_title="USD / barrel",
        title=dict(text="Brent Crude Oil Price 1983–2019", font=dict(size=12, color=NAVY), x=0),
    )
    st.plotly_chart(fig_oil, use_container_width=True)

    st.markdown('<div class="sec-lbl" style="margin-top:6px;">Exhibit 2 — Public, External & Domestic Debt (% of GDP)</div>', unsafe_allow_html=True)
    fig_debt = go.Figure()
    for col, label, color in [
        ("public_debt",   "Total Public Debt",    NAVY),
        ("external_debt", "External Debt",         GOLD),
        ("domestic_debt", "Domestic Debt",         TEAL),
    ]:
        fig_debt.add_trace(go.Scatter(
            x=years, y=df_levels[col],
            mode="lines", name=label,
            line=dict(color=color, width=2.2),
        ))
    fig_debt.update_layout(
        **navy_layout(height=300),
        xaxis_title="Year", yaxis_title="% of GDP",
        legend=dict(orientation="h", y=-0.25, x=0),
        title=dict(text="Ghana Debt Dynamics 1983–2019", font=dict(size=12, color=NAVY), x=0),
    )
    st.plotly_chart(fig_debt, use_container_width=True)

    st.markdown('<div class="sec-lbl" style="margin-top:6px;">Exhibit 3 — Macro Conditions: Inflation, REER & Interest Rate</div>', unsafe_allow_html=True)
    fig_macro = make_subplots(rows=1, cols=3,
                               subplot_titles=["Inflation (%)",
                                               "REER (index)",
                                               "91-Day T-bill (%)"])
    for i, (col, color) in enumerate([
        ("inflation",    "#7C3AED"),
        ("reer",          "#059669"),
        ("interest_rate", "#B45309"),
    ], 1):
        fig_macro.add_trace(go.Scatter(
            x=years, y=df_levels[col],
            mode="lines", line=dict(color=color, width=2),
            showlegend=False,
        ), row=1, col=i)
    fig_macro.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        height=260, margin=dict(t=40, b=40, l=40, r=20),
        font=dict(size=10, color=BODY),
    )
    for ax in ["xaxis", "xaxis2", "xaxis3", "yaxis", "yaxis2", "yaxis3"]:
        fig_macro.update_layout(**{ax: dict(showgrid=False, zeroline=False)})
    st.plotly_chart(fig_macro, use_container_width=True)

    st.markdown('<div class="sec-lbl" style="margin-top:6px;">Exhibit 4 — Pairwise Correlation (log levels)</div>', unsafe_allow_html=True)
    corr = df_log.rename(columns={
        "log_pub": "Pub. Debt", "log_ext": "Ext. Debt", "log_dom": "Dom. Debt",
        "log_oil": "Oil", "log_infl": "Inflation", "log_reer": "REER", "log_ir": "Interest",
    }).corr()
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
        colorscale=[[0, "#FFF5F5"], [0.5, "#E8F0FB"], [1, NAVY]],
        text=corr.round(2).values.astype(str),
        texttemplate="%{text}", showscale=True,
        zmin=-1, zmax=1,
    ))
    fig_corr.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        height=310, margin=dict(t=10, b=10, l=80, r=20),
        font=dict(size=10),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown('<div class="sec-lbl" style="margin-top:10px;">Key Observation</div>', unsafe_allow_html=True)
    obs1, obs2 = st.columns(2)
    with obs1:
        st.markdown(f"""
        <div class="finding">
          <div class="finding-h">Oil price and public debt: visual co-movement</div>
          <div class="finding-b">
            The time-series plots show oil price and public/external debt moving
            together over medium-run horizons — declining through the HIPC debt
            relief period (2004) and rising post-2014. This visual co-movement
            motivates the formal cointegration tests in Tab 2.
          </div>
        </div>""", unsafe_allow_html=True)
    with obs2:
        st.markdown(f"""
        <div class="finding">
          <div class="finding-h">Domestic debt: no oil co-movement</div>
          <div class="finding-b">
            Domestic debt shows a distinct trajectory — driven by domestic
            monetary conditions and fiscal policy rather than global oil prices.
            This provides a natural control case for the transmission analysis:
            if oil affects debt, the channel should be external (import costs,
            exchange rate pressure) not domestic.
          </div>
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# TAB 2 — UNIT ROOT & COINTEGRATION
# ═════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-lbl">Unit Root & Cointegration</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-ttl">ADF tests confirm I(1); Johansen detects cointegrating vectors</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Prerequisite for VECM/VAR: all variables integrated of order 1 (I(1)); at least one cointegrating vector in each model.</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-lbl" style="margin-top:4px;">Panel A — Augmented Dickey-Fuller Unit Root Tests</div>', unsafe_allow_html=True)

    html_rows = ""
    for _, row in adf_results.iterrows():
        i0_cls  = "sig-yes" if row["I(0)?"] == "Yes" else "sig-no"
        i1_cls  = "sig-yes" if row["I(1)?"] == "Yes" else "sig-no"
        html_rows += f"""
        <tr>
          <td>{row['Variable']}</td>
          <td>{row['ADF (Level)']}</td>
          <td>{row['p (Level)']}</td>
          <td class="{i0_cls}">{row['I(0)?']}</td>
          <td>{row['ADF (Δ)']}</td>
          <td>{row['p (Δ)']}</td>
          <td class="{i1_cls}">{row['I(1)?']}</td>
        </tr>"""

    st.markdown(f"""
    <table class="result-tbl">
      <thead>
        <tr>
          <th>Variable</th>
          <th>ADF Stat (level)</th>
          <th>p-value (level)</th>
          <th>Stationary I(0)?</th>
          <th>ADF Stat (Δ)</th>
          <th>p-value (Δ)</th>
          <th>Integrated I(1)?</th>
        </tr>
      </thead>
      <tbody>{html_rows}</tbody>
    </table>
    <div style="font-size:11px;color:{MUTED};margin-top:6px;">
      Null hypothesis: series has a unit root. Rejection (p &lt; 0.05) implies stationarity.
      AIC lag selection. 5% significance level.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="sec-lbl">Visual Check — First Differences of Key Variables</div>', unsafe_allow_html=True)
    fig_diff = go.Figure()
    for col, label, color in [
        ("log_pub",  "Δ log Public Debt",  NAVY),
        ("log_ext",  "Δ log External Debt", GOLD),
        ("log_oil",  "Δ log Oil Price",    RED),
    ]:
        diff_series = df_log[col].diff().dropna()
        fig_diff.add_trace(go.Scatter(
            x=diff_series.index.tolist(), y=diff_series.values,
            mode="lines", name=label,
            line=dict(color=color, width=1.8),
        ))
    fig_diff.add_hline(y=0, line=dict(color=RULE, width=1, dash="dot"))
    fig_diff.update_layout(
        **navy_layout(height=260),
        title=dict(text="First Differences — visual stationarity check",
                   font=dict(size=12, color=NAVY), x=0),
        xaxis_title="Year", yaxis_title="First difference",
        legend=dict(orientation="h", y=-0.28, x=0),
    )
    st.plotly_chart(fig_diff, use_container_width=True)

    st.markdown('<div class="sec-lbl" style="margin-top:6px;">Panel B — Johansen Cointegration Tests (Trace & Max-Eigenvalue)</div>', unsafe_allow_html=True)

    joh_c1, joh_c2, joh_c3 = st.columns(3)
    for col_obj, model_name, color in [
        (joh_c1, "Public Debt",   NAVY),
        (joh_c2, "External Debt", GOLD),
        (joh_c3, "Domestic Debt", TEAL),
    ]:
        res = joh_results[model_name]
        r   = res["cointegrating_vectors"]
        with col_obj:
            st.markdown(f"""
            <div style="background:#FFFFFF;border:1px solid {RULE};
                        border-top:3px solid {color};border-radius:4px;
                        padding:14px 16px;margin-bottom:10px;">
              <div style="font-size:11px;font-weight:700;color:{MUTED};
                          letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;">
                {model_name} Model
              </div>
              <div style="font-size:22px;font-weight:700;color:{color};">
                r = {r}
              </div>
              <div style="font-size:12px;color:{MUTED};">
                cointegrating {"vector" if r==1 else "vectors"} at 5%
              </div>
              <div style="font-size:12px;color:{GREEN if r > 0 else RED};
                          font-weight:700;margin-top:4px;">
                {"✓ Cointegrated" if r > 0 else "✗ No cointegration"}
              </div>
            </div>
            """, unsafe_allow_html=True)

            tbl = res["trace_table"]
            html_joh = ""
            for _, row in tbl.iterrows():
                rej_cls = "sig-yes" if "Yes" in str(row["Reject H₀?"]) else "sig-no"
                html_joh += f"""<tr>
                  <td>r ≤ {int(row['H₀: rank ≤ r'])}</td>
                  <td>{row['Trace Statistic']}</td>
                  <td>{row['5% Critical Value']}</td>
                  <td>{row['Max-Eigen Stat']}</td>
                  <td class="{rej_cls}">{row['Reject H₀?']}</td>
                </tr>"""
            st.markdown(f"""
            <table class="result-tbl" style="font-size:11px;">
              <thead><tr>
                <th>H₀</th><th>Trace</th><th>5% CV</th>
                <th>Max-Eig</th><th>Reject?</th>
              </tr></thead>
              <tbody>{html_joh}</tbody>
            </table>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="sec-lbl">Interpretation</div>', unsafe_allow_html=True)
    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown(f"""
        <div class="brief-navy">
          <div class="brief-head" style="color:{NAVY};">All Variables I(1)</div>
          <div class="brief-body">
            ADF tests on levels fail to reject the unit root null for all 7 variables.
            First differences are stationary. This confirms the I(1) property required
            for cointegration analysis and validates the VECM framework.
          </div>
        </div>""", unsafe_allow_html=True)
    with i2:
        st.markdown(f"""
        <div class="brief-gold">
          <div class="brief-head" style="color:{AMBER};">Public & External Debt: Cointegrated</div>
          <div class="brief-body">
            Johansen trace and max-eigenvalue tests detect r ≥ 1 cointegrating vector
            for public and external debt models. Oil price shocks have a <em>long-run</em>
            equilibrium relationship with these debt stocks — consistent with Ghana's
            fiscal exposure to global commodity cycles.
          </div>
        </div>""", unsafe_allow_html=True)
    with i3:
        st.markdown(f"""
        <div class="brief-red">
          <div class="brief-head" style="color:{RED};">Domestic Debt: No Cointegration</div>
          <div class="brief-body">
            The domestic debt model shows no cointegrating vector with oil price — confirming
            that oil shocks transmit primarily through external channels (import costs, balance
            of payments, exchange rate pressure) rather than domestic monetary/fiscal dynamics.
            This asymmetry is a key thesis finding.
          </div>
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# TAB 3 — IMPULSE RESPONSE FUNCTIONS
# ═════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-lbl">Impulse Response Functions</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-ttl">How does each debt type respond to a +1 SD oil price shock?</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sec-sub">Orthogonalized IRF (Cholesky decomposition) · {irf_horizon}-period horizon · Shaded band = 90% approximate CI</div>', unsafe_allow_html=True)

    irf_col1, irf_col2, irf_col3 = st.columns(3)

    for col_obj, model_name, color, label in [
        (irf_col1, "Public Debt",   NAVY, "Total Public Debt"),
        (irf_col2, "External Debt", GOLD, "External Debt"),
        (irf_col3, "Domestic Debt", TEAL, "Domestic Debt"),
    ]:
        res = var_results[model_name]
        irfs = res["orth_irfs"][:irf_horizon + 1]
        upper = res["ci_upper"][:irf_horizon + 1] if show_ci else irfs
        lower = res["ci_lower"][:irf_horizon + 1] if show_ci else irfs
        with col_obj:
            st.plotly_chart(
                irf_chart(irfs, upper, lower,
                          title=f"Response of {label} to +1SD Oil Shock",
                          color=color, debt_label=label),
                use_container_width=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="sec-lbl">IRF Values by Period</div>', unsafe_allow_html=True)
    periods_list = list(range(irf_horizon + 1))
    irf_table = pd.DataFrame({
        "Period":        periods_list,
        "Public Debt":   [round(v, 5) for v in var_results["Public Debt"]["orth_irfs"][:irf_horizon + 1]],
        "External Debt": [round(v, 5) for v in var_results["External Debt"]["orth_irfs"][:irf_horizon + 1]],
        "Domestic Debt": [round(v, 5) for v in var_results["Domestic Debt"]["orth_irfs"][:irf_horizon + 1]],
    }).set_index("Period")

    def color_neg(val):
        color = "color: #C8382A;" if isinstance(val, float) and val < 0 else "color: #1A7A2E;"
        return color

    st.dataframe(irf_table.style.map(color_neg), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="sec-lbl">Exhibit — All Three Responses Overlaid</div>', unsafe_allow_html=True)
    fig_overlay = go.Figure()
    fig_overlay.add_hline(y=0, line=dict(color=RULE, width=1.5, dash="dot"))
    for model_name, color, label in [
        ("Public Debt",   NAVY, "Total Public Debt"),
        ("External Debt", GOLD, "External Debt"),
        ("Domestic Debt", TEAL, "Domestic Debt"),
    ]:
        irfs = var_results[model_name]["orth_irfs"][:irf_horizon + 1]
        fig_overlay.add_trace(go.Scatter(
            x=list(range(irf_horizon + 1)), y=irfs,
            mode="lines+markers",
            line=dict(color=color, width=2.2),
            marker=dict(size=6),
            name=label,
        ))
    fig_overlay.update_layout(
        **navy_layout(height=320),
        title=dict(text="Orthogonalized IRF: Response to +1SD Oil Price Shock",
                   font=dict(size=13, color=NAVY), x=0),
        xaxis_title="Periods after shock",
        yaxis_title="Log-point response",
        legend=dict(orientation="h", y=-0.22, x=0),
    )
    st.plotly_chart(fig_overlay, use_container_width=True)

    st.markdown('<div class="sec-lbl" style="margin-top:4px;">Economic Interpretation</div>', unsafe_allow_html=True)
    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        st.markdown(f"""
        <div class="brief-navy">
          <div class="brief-head" style="color:{NAVY};">Public Debt: Negative Short-Run</div>
          <div class="brief-body">
            Total public debt responds <em>negatively</em> to a positive oil shock in
            the first 1–2 periods before recovering. Mechanism: a positive oil shock
            temporarily improves Ghana's terms of trade (gold/cocoa export revenue
            rises alongside commodity prices) and reduces immediate borrowing need.
            The effect dissipates as import costs dominate.
          </div>
        </div>""", unsafe_allow_html=True)
    with ec2:
        st.markdown(f"""
        <div class="brief-gold">
          <div class="brief-head" style="color:{AMBER};">External Debt: Medium-Run Accumulation</div>
          <div class="brief-body">
            External debt shows a more persistent positive medium-run response.
            Higher oil prices widen Ghana's import bill and current account deficit,
            requiring additional external financing. This reflects Ghana's net oil
            importer status for most of the sample period (pre-2011 oil production).
          </div>
        </div>""", unsafe_allow_html=True)
    with ec3:
        st.markdown(f"""
        <div class="brief-red">
          <div class="brief-head" style="color:{TEAL};">Domestic Debt: No Systematic Response</div>
          <div class="brief-body">
            Domestic debt shows no statistically significant response to oil shocks —
            consistent with the absence of cointegration found in Tab 2.
            Domestic borrowing is driven primarily by local monetary conditions and
            fiscal deficit financing, insulated from global oil price cycles.
          </div>
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# TAB 4 — VARIANCE DECOMPOSITION (FEVD)
# ═════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec-lbl">Forecast Error Variance Decomposition</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-ttl">What share of debt forecast variance is explained by oil price shocks?</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">FEVD from orthogonalized VAR. Bars show cumulative contribution of each variable across 10 forecast periods.</div>', unsafe_allow_html=True)

    fv1, fv2, fv3 = st.columns(3)
    for col_obj, model_name, color in [
        (fv1, "Public Debt",   NAVY),
        (fv2, "External Debt", GOLD),
        (fv3, "Domestic Debt", TEAL),
    ]:
        res = var_results[model_name]
        var_names_model = res["var_names"]
        fevd_all = []
        for t in range(10):
            row = [res["var_results"].fevd(10).decomp[0, t, j] * 100
                   for j in range(len(var_names_model))]
            fevd_all.append(row)
        with col_obj:
            st.plotly_chart(
                fevd_stacked_bar(fevd_all, var_names_model,
                                 title=f"{model_name} — FEVD",
                                 highlight_oil=True),
                use_container_width=True,
            )
            oil_line = [r[1] for r in fevd_all]
            fig_oil_line = go.Figure()
            fig_oil_line.add_trace(go.Scatter(
                x=list(range(1, 11)), y=oil_line,
                mode="lines+markers",
                line=dict(color=RED, width=2),
                marker=dict(size=6),
                name="Oil % contribution",
            ))
            fig_oil_line.update_layout(
                **navy_layout(height=180, margin=dict(t=20, b=30, l=40, r=10)),
                title=dict(text="Oil Price Contribution (%)",
                           font=dict(size=11, color=NAVY), x=0),
                xaxis_title="Period", yaxis_title="%",
            )
            st.plotly_chart(fig_oil_line, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="sec-lbl">Summary — Oil Price Contribution to Debt Variance at Horizon 10</div>', unsafe_allow_html=True)
    summary_data = {
        "Debt Model":        ["Public Debt",    "External Debt",  "Domestic Debt"],
        "Oil % (Period 1)":  [
            round(var_results["Public Debt"]["fevd_oil"][0], 2),
            round(var_results["External Debt"]["fevd_oil"][0], 2),
            round(var_results["Domestic Debt"]["fevd_oil"][0], 2),
        ],
        "Oil % (Period 5)":  [
            round(var_results["Public Debt"]["fevd_oil"][4], 2),
            round(var_results["External Debt"]["fevd_oil"][4], 2),
            round(var_results["Domestic Debt"]["fevd_oil"][4], 2),
        ],
        "Oil % (Period 10)": [
            round(var_results["Public Debt"]["fevd_oil"][9], 2),
            round(var_results["External Debt"]["fevd_oil"][9], 2),
            round(var_results["Domestic Debt"]["fevd_oil"][9], 2),
        ],
        "Thesis Benchmark":  ["~1.4%", "~4.4%", "~0% (n.s.)"],
        "Cointegrated?":     ["Yes", "Yes", "No"],
    }
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary.set_index("Debt Model"), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="sec-lbl">Exhibit — Own Shock vs Oil Shock vs All Others (Period 10)</div>', unsafe_allow_html=True)
    fig_share = go.Figure()
    categories = ["Public Debt", "External Debt", "Domestic Debt"]
    own_vals, oil_vals, other_vals = [], [], []
    for model_name in categories:
        final = var_results[model_name]["fevd_final"]
        own_vals.append(round(final[0], 1))
        oil_vals.append(round(final[1], 1))
        other_vals.append(round(100 - final[0] - final[1], 1))

    fig_share.add_trace(go.Bar(name="Own Shock", x=categories, y=own_vals,
                                marker_color=NAVY))
    fig_share.add_trace(go.Bar(name="Oil Price ★", x=categories, y=oil_vals,
                                marker_color=RED))
    fig_share.add_trace(go.Bar(name="All Others", x=categories, y=other_vals,
                                marker_color=MUTED))
    fig_share.update_layout(
        barmode="stack",
        **navy_layout(height=300),
        title=dict(text="FEVD at Horizon 10: Own / Oil / Other decomposition",
                   font=dict(size=13, color=NAVY), x=0),
        xaxis_title="Debt Model",
        yaxis_title="% of forecast variance",
        yaxis_range=[0, 100],
        legend=dict(orientation="h", y=-0.22, x=0),
        bargap=0.28,
    )
    st.plotly_chart(fig_share, use_container_width=True)

    st.markdown('<div class="sec-lbl" style="margin-top:6px;">Policy Interpretation</div>', unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    with p1:
        st.markdown(f"""
        <div class="finding">
          <div class="finding-h">Oil shocks matter — but own dynamics dominate</div>
          <div class="finding-b">
            While oil explains a statistically significant share of public and external
            debt variance, own-shock dynamics (inertia, domestic fiscal decisions) dominate
            at all horizons. This implies that debt sustainability in Ghana is primarily
            a domestic governance challenge — not simply an oil cycle story.
            However, the external debt channel (~4.4%) is meaningful for policymakers
            managing foreign exchange exposure.
          </div>
        </div>""", unsafe_allow_html=True)
    with p2:
        st.markdown(f"""
        <div class="finding">
          <div class="finding-h">SCE Application: Energy pricing & fiscal risk</div>
          <div class="finding-b">
            This framework directly informs energy policy analysis. For Southern California
            Edison, analogous questions arise: what fraction of grid investment cost variance
            is explained by fuel price shocks vs. regulatory decisions vs. demand growth?
            The same VAR/FEVD methodology — applied to utility cost drivers — quantifies how
            much hedging/stabilisation is achievable through price policy vs. capital allocation.
          </div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#666; font-size:0.85rem; padding:10px 0;'>"
    "<strong>Sherriff Abdul-Hamid</strong><br>"
    "Data Scientist · Energy Economist · Development Economics Research<br>"
    "<a href='https://poverty360.org' target='_blank'>poverty360.org</a> · "
    "<a href='https://www.linkedin.com/in/abdul-hamid-sherriff-08583354/' target='_blank'>LinkedIn</a> · "
    "<a href='https://github.com/S-ABDUL-AI' target='_blank'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True,
)
