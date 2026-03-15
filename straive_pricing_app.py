"""
STRAIVE — Dynamic Pricing Optimization Platform
Advanced Analytics: Demand Modeling + Price Elasticity + Revenue Simulation + What-If Engine
Run: streamlit run straive_pricing_app.py
"""
from __future__ import annotations
import io
import json
import logging
import math
import sys
import warnings
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm, pearsonr
from scipy.stats.qmc import Sobol

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("straive_artifacts")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PARAMS = OUTPUT_DIR / "pricing_params.json"
CURRENT_DATE = datetime.now()

# STRAIVE Product Lines / Service Tiers
PRODUCT_CATALOG: Dict[str, Dict] = {
    "Editorial Services – Standard":    {"base_price": 2800, "unit": "project", "segment": "Editorial",      "cost": 1100},
    "Editorial Services – Premium":     {"base_price": 5200, "unit": "project", "segment": "Editorial",      "cost": 1800},
    "Data Annotation – Basic":          {"base_price":  450, "unit": "1k items","segment": "Data Services",  "cost":  180},
    "Data Annotation – Advanced":       {"base_price":  980, "unit": "1k items","segment": "Data Services",  "cost":  320},
    "AI/ML Pipeline – Starter":         {"base_price": 8500, "unit": "month",   "segment": "AI/ML",          "cost": 3200},
    "AI/ML Pipeline – Enterprise":      {"base_price":22000, "unit": "month",   "segment": "AI/ML",          "cost": 7500},
    "Content Transformation – Basic":   {"base_price": 1200, "unit": "project", "segment": "Content",        "cost":  480},
    "Content Transformation – Plus":    {"base_price": 2600, "unit": "project", "segment": "Content",        "cost":  900},
    "Research & Analytics – Standard":  {"base_price": 4800, "unit": "project", "segment": "Analytics",      "cost": 1700},
    "Research & Analytics – Premium":   {"base_price":11500, "unit": "project", "segment": "Analytics",      "cost": 3800},
    "Publishing Tech – SaaS":           {"base_price": 3200, "unit": "month",   "segment": "Technology",     "cost":  950},
    "Publishing Tech – Enterprise":     {"base_price": 9500, "unit": "month",   "segment": "Technology",     "cost": 2600},
}

CUSTOMER_SEGMENTS: Dict[str, Dict] = {
    "Academic Publishers":    {"volume_multiplier": 1.4, "price_sensitivity": 0.72, "loyalty": 0.85, "color": "#4f9eff"},
    "STM Publishers":         {"volume_multiplier": 2.1, "price_sensitivity": 0.55, "loyalty": 0.90, "color": "#36d97b"},
    "Trade Publishers":       {"volume_multiplier": 0.9, "price_sensitivity": 0.88, "loyalty": 0.70, "color": "#ff6b6b"},
    "Government & NGO":       {"volume_multiplier": 1.2, "price_sensitivity": 0.61, "loyalty": 0.82, "color": "#ffd700"},
    "Corporate/Enterprise":   {"volume_multiplier": 1.8, "price_sensitivity": 0.48, "loyalty": 0.78, "color": "#b48eff"},
    "EdTech Platforms":       {"volume_multiplier": 1.3, "price_sensitivity": 0.80, "loyalty": 0.68, "color": "#ff9800"},
}

REGIONS: Dict[str, Dict] = {
    "North America": {"demand_index": 1.35, "competitor_pressure": 0.65, "growth_rate": 0.08},
    "Europe":        {"demand_index": 1.20, "competitor_pressure": 0.72, "growth_rate": 0.06},
    "Asia Pacific":  {"demand_index": 1.55, "competitor_pressure": 0.80, "growth_rate": 0.14},
    "Middle East":   {"demand_index": 0.88, "competitor_pressure": 0.55, "growth_rate": 0.11},
    "Latin America": {"demand_index": 0.72, "competitor_pressure": 0.60, "growth_rate": 0.09},
    "Africa":        {"demand_index": 0.60, "competitor_pressure": 0.40, "growth_rate": 0.18},
}

COMPETITORS = {
    "Aptara":         {"relative_price": 0.88, "quality_score": 7.2},
    "Innodata":       {"relative_price": 0.82, "quality_score": 7.5},
    "MPS Limited":    {"relative_price": 0.91, "quality_score": 7.8},
    "Cenveo":         {"relative_price": 0.94, "quality_score": 7.0},
    "Techbooks":      {"relative_price": 0.78, "quality_score": 6.8},
    "SPi Global":     {"relative_price": 0.86, "quality_score": 7.3},
}

PRODUCT_SEGMENTS = list(set(v["segment"] for v in PRODUCT_CATALOG.values()))
SEGMENT_COLORS = {"Editorial": "#4f9eff", "Data Services": "#36d97b", "AI/ML": "#b48eff",
                  "Content": "#ff9800", "Analytics": "#ffd700", "Technology": "#ff6b6b"}

# ─────────────────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG  = "#05080f"
CARD_BG  = "#0e1320"
BORDER   = "#1e2740"
ACCENT   = "#4f9eff"
ACCENT2  = "#ff6b6b"
ACCENT3  = "#ffd700"
TEXT     = "#e8edf5"
MUTED    = "#7a869a"
GREEN    = "#36d97b"
YELLOW   = "#f5a623"
RED      = "#ff4d6d"
PURPLE   = "#b48eff"

PLOTLY_DARK = dict(
    paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
    font=dict(color=TEXT, family="'Inter','Segoe UI',sans-serif", size=12),
    title_font=dict(color=TEXT, size=15, family="'Oswald',sans-serif"),
    legend=dict(bgcolor="rgba(14,19,32,0.85)", bordercolor=BORDER, borderwidth=1,
                font=dict(color=TEXT, size=11)),
    margin=dict(l=14, r=14, t=48, b=14),
    hoverlabel=dict(bgcolor=CARD_BG, bordercolor=BORDER, font=dict(color=TEXT, size=12)),
)

def style_fig(fig: go.Figure, height: Optional[int] = None) -> go.Figure:
    upd = dict(**PLOTLY_DARK)
    if height: upd["height"] = height
    fig.update_layout(**upd)
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER, tickfont=dict(color=MUTED, size=11))
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER, tickfont=dict(color=MUTED, size=11))
    return fig

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def generate_transaction_data(n: int = 3600, seed: int = 42) -> pd.DataFrame:
    """Generate realistic STRAIVE transaction history."""
    np.random.seed(seed)
    products = list(PRODUCT_CATALOG.keys())
    segments = list(CUSTOMER_SEGMENTS.keys())
    regions  = list(REGIONS.keys())

    rows = []
    start = CURRENT_DATE - timedelta(days=730)
    for _ in range(n):
        prod    = np.random.choice(products)
        seg     = np.random.choice(segments)
        reg     = np.random.choice(regions)
        info    = PRODUCT_CATALOG[prod]
        seg_inf = CUSTOMER_SEGMENTS[seg]
        reg_inf = REGIONS[reg]

        # Price variation around base
        base  = info["base_price"]
        noise = np.random.normal(0, 0.12)
        price = base * (1 + noise) * (0.88 + 0.24 * np.random.random())

        # Demand driven by price elasticity
        elasticity    = -1.2 - 0.5 * (1 - seg_inf["price_sensitivity"])
        demand_base   = seg_inf["volume_multiplier"] * reg_inf["demand_index"] * 8
        price_ratio   = price / base
        demand        = max(1, int(demand_base * (price_ratio ** elasticity) + np.random.normal(0, 1.5)))

        revenue  = price * demand
        cost     = info["cost"] * demand
        margin   = (revenue - cost) / revenue if revenue > 0 else 0
        days_ago = int(np.random.exponential(200))
        date     = start + timedelta(days=min(days_ago, 730))

        # Seasonality bump Q4
        if date.month in [10, 11, 12]:
            demand = int(demand * 1.22)
            revenue = price * demand

        rows.append({
            "date":          date,
            "product":       prod,
            "segment":       info["segment"],
            "customer_type": seg,
            "region":        reg,
            "base_price":    round(base, 2),
            "actual_price":  round(price, 2),
            "discount_pct":  round(max(0, (base - price) / base * 100), 1),
            "volume":        demand,
            "revenue":       round(revenue, 2),
            "cost":          round(cost, 2),
            "margin_pct":    round(margin * 100, 2),
            "deal_won":      1 if np.random.random() < (0.65 - 0.2 * max(0, price_ratio - 1)) else 0,
        })

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# ELASTICITY MODEL
# ─────────────────────────────────────────────────────────────────────────────
def fit_elasticity_model(df: pd.DataFrame) -> Dict[str, Any]:
    """Fit log-log OLS demand model: log(volume) ~ log(price) + controls."""
    results = {}
    for seg in df["segment"].unique():
        sub = df[df["segment"] == seg].copy()
        sub = sub[sub["volume"] > 0]
        if len(sub) < 20:
            continue
        sub["log_volume"] = np.log(sub["volume"])
        sub["log_price"]  = np.log(sub["actual_price"])
        sub["log_disc"]   = np.log1p(sub["discount_pct"])
        try:
            mod = smf.ols("log_volume ~ log_price + log_disc", data=sub).fit()
            results[seg] = {
                "elasticity":   round(float(mod.params.get("log_price", -1.2)), 4),
                "disc_lift":    round(float(mod.params.get("log_disc", 0.15)), 4),
                "intercept":    round(float(mod.params["Intercept"]), 4),
                "r_squared":    round(float(mod.rsquared), 4),
                "n_obs":        len(sub),
                "mean_price":   round(float(sub["actual_price"].mean()), 2),
                "mean_volume":  round(float(sub["volume"].mean()), 2),
                "mean_margin":  round(float(sub["margin_pct"].mean()), 2),
            }
        except Exception as e:
            log.warning(f"Elasticity fit failed for {seg}: {e}")
    return results

def fit_win_probability_model(df: pd.DataFrame) -> Any:
    """Logistic regression: P(deal_won) ~ price_ratio + discount."""
    sub = df.copy()
    sub["price_ratio"] = sub["actual_price"] / sub["base_price"]
    try:
        mod = smf.logit("deal_won ~ price_ratio + discount_pct", data=sub).fit(disp=0)
        return mod
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# OPTIMAL PRICE ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def optimal_price(
    base_price: float,
    cost: float,
    elasticity: float,
    target: str = "revenue",  # "revenue" | "profit" | "margin"
    constraints: Optional[Dict] = None,
) -> Dict[str, float]:
    """Compute revenue/profit-maximising price via 1-D optimisation."""
    lower = base_price * 0.50
    upper = base_price * 2.00
    if constraints:
        lower = max(lower, constraints.get("min_price", lower))
        upper = min(upper, constraints.get("max_price", upper))

    def demand(p: float) -> float:
        return max(0.01, (p / base_price) ** elasticity)

    def neg_obj(p: float) -> float:
        d = demand(p)
        if target == "revenue":
            return -(p * d)
        elif target == "profit":
            return -((p - cost) * d)
        else:  # margin
            rev = p * d
            return -((rev - cost * d) / rev) if rev > 0 else 0

    res = minimize_scalar(neg_obj, bounds=(lower, upper), method="bounded")
    p_opt = float(res.x)
    d_opt = demand(p_opt)
    rev   = p_opt * d_opt
    profit= (p_opt - cost) * d_opt
    return {
        "optimal_price":   round(p_opt, 2),
        "demand_index":    round(d_opt, 4),
        "expected_revenue": round(rev, 2),
        "expected_profit":  round(profit, 2),
        "margin_pct":       round((p_opt - cost) / p_opt * 100, 2),
        "vs_base_pct":      round((p_opt / base_price - 1) * 100, 2),
    }

# ─────────────────────────────────────────────────────────────────────────────
# SOBOL MONTE-CARLO REVENUE SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
def simulate_revenue_scenarios(
    product: str,
    price_range: Tuple[float, float],
    elasticity: float,
    cost: float,
    n_sim: int = 2048,
) -> pd.DataFrame:
    """QMC simulation of revenue across price scenarios with uncertainty."""
    n_sim_pow2 = int(2 ** np.floor(np.log2(max(n_sim, 64))))
    sobol = Sobol(d=2, scramble=True)
    samples = sobol.random(n_sim_pow2)

    p_lo, p_hi = price_range
    prices  = p_lo + (p_hi - p_lo) * samples[:, 0]
    e_noise = elasticity * (1 + 0.15 * (samples[:, 1] * 2 - 1))  # ±15% elasticity uncertainty

    base = PRODUCT_CATALOG[product]["base_price"]
    demand = np.maximum(0.01, (prices / base) ** e_noise)
    revenue = prices * demand
    profit  = (prices - cost) * demand
    margin  = np.where(revenue > 0, (revenue - cost * demand) / revenue * 100, 0)

    return pd.DataFrame({
        "price":   np.round(prices, 2),
        "demand":  np.round(demand, 4),
        "revenue": np.round(revenue, 2),
        "profit":  np.round(profit, 2),
        "margin":  np.round(margin, 2),
    })

# ─────────────────────────────────────────────────────────────────────────────
# COMPETITIVE POSITIONING
# ─────────────────────────────────────────────────────────────────────────────
def competitive_price_score(price: float, base_price: float) -> Dict[str, Any]:
    """Score STRAIVE pricing against known competitors."""
    straive_rel = price / base_price
    comp_data = []
    for name, info in COMPETITORS.items():
        comp_price = base_price * info["relative_price"]
        gap_pct    = (price - comp_price) / comp_price * 100
        comp_data.append({
            "Competitor":     name,
            "Comp Price":     round(comp_price, 2),
            "STRAIVE Price":  round(price, 2),
            "Gap %":          round(gap_pct, 1),
            "Quality Score":  info["quality_score"],
            "Comp Relative":  info["relative_price"],
        })
    df = pd.DataFrame(comp_data)
    cheaper_than = int((df["Gap %"] < 0).sum())
    pct_rank     = cheaper_than / len(df) * 100
    return {"df": df, "pct_rank": round(pct_rank, 1), "straive_relative": round(straive_rel, 4)}

# ─────────────────────────────────────────────────────────────────────────────
# WHAT-IF ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def apply_pricing_scenario(
    df: pd.DataFrame,
    overrides: Dict[str, float],  # {product: new_price}
) -> pd.DataFrame:
    """Return a modified copy of the transaction data with overridden prices."""
    df2 = df.copy()
    for prod, new_price in overrides.items():
        mask = df2["product"] == prod
        old  = PRODUCT_CATALOG[prod]["base_price"]
        elast_raw = -1.2  # fallback
        df2.loc[mask, "actual_price"] = new_price
        df2.loc[mask, "discount_pct"] = max(0, (old - new_price) / old * 100)
        ratio = new_price / old
        df2.loc[mask, "volume"]  = (df2.loc[mask, "volume"] * (ratio ** elast_raw)).round(0).astype(int)
        df2.loc[mask, "revenue"] = df2.loc[mask, "actual_price"] * df2.loc[mask, "volume"]
        df2.loc[mask, "cost"]    = PRODUCT_CATALOG[prod]["cost"] * df2.loc[mask, "volume"]
        df2.loc[mask, "margin_pct"] = ((df2.loc[mask, "revenue"] - df2.loc[mask, "cost"]) /
                                        df2.loc[mask, "revenue"] * 100).round(2)
    return df2

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="STRAIVE · Dynamic Pricing Optimization",
        page_icon="💹",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── CSS ──────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {{
        background-color: {DARK_BG} !important;
        color: {TEXT};
        font-family: 'Inter','Segoe UI',sans-serif;
    }}
    .stApp {{ background: {DARK_BG}; }}
    .block-container {{ padding-top:1.6rem; padding-bottom:2rem; max-width:1440px; }}

    .main-title {{
        font-family:'Oswald',sans-serif;
        font-size: clamp(1.9rem,4.5vw,3rem);
        font-weight:700;
        background: linear-gradient(120deg,{ACCENT} 0%,{GREEN} 45%,{ACCENT3} 80%,{ACCENT} 100%);
        background-size:200% auto;
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
        animation: shimmer 5s linear infinite;
        text-align:center; letter-spacing:4px; padding:.6rem 0 .2rem 0; text-transform:uppercase;
    }}
    @keyframes shimmer {{ 0%{{background-position:0% center}} 100%{{background-position:200% center}} }}

    .hero-underline {{
        height:2px; width:60%; margin:.35rem auto .6rem auto;
        background:linear-gradient(90deg,transparent,{ACCENT},{ACCENT3},{GREEN},transparent);
        border-radius:2px; opacity:.75;
    }}
    .sub-title {{
        text-align:center; color:{MUTED}; font-size:.78rem; letter-spacing:2.5px;
        margin-bottom:2rem; text-transform:uppercase; font-family:'Inter',sans-serif;
    }}

    .section-header {{
        font-family:'Oswald',sans-serif; font-size:1.05rem; font-weight:600;
        letter-spacing:2.5px; color:{ACCENT}; text-transform:uppercase;
        display:flex; align-items:center; gap:.7rem;
        margin:1.6rem 0 .9rem 0; padding-bottom:.45rem; border-bottom:1px solid {BORDER};
    }}
    .section-header::before {{
        content:''; display:block; width:4px; height:1.1em; border-radius:2px;
        background:linear-gradient(180deg,{ACCENT},{GREEN}); flex-shrink:0;
    }}

    .insight-card {{
        background:{CARD_BG}; border:1px solid {BORDER}; border-radius:14px;
        padding:1.3rem 1.2rem; margin:.5rem 0;
        box-shadow:0 2px 16px rgba(0,0,0,.35);
        transition:box-shadow .2s,border-color .2s,transform .15s;
    }}
    .insight-card:hover {{
        box-shadow:0 6px 28px rgba(79,158,255,.18); border-color:{ACCENT}; transform:translateY(-2px);
    }}
    .insight-title  {{ font-family:'Oswald',sans-serif;font-size:.9rem;font-weight:600;
                       color:{ACCENT};text-transform:uppercase;letter-spacing:1px;margin-bottom:.5rem; }}
    .insight-value  {{ font-size:1.5rem;font-weight:700;color:{TEXT};font-family:'Oswald',sans-serif; }}
    .insight-sub    {{ font-size:.75rem;color:{MUTED};margin-top:.3rem; }}

    .kpi-card {{
        background:{CARD_BG}; border:1px solid {BORDER}; border-radius:10px;
        padding:.55rem .5rem; text-align:center; border-top:2px solid {ACCENT};
        margin-bottom:.8rem;
    }}
    .kpi-val  {{ font-family:'Oswald',sans-serif;font-size:1.45rem;font-weight:700;line-height:1.2; }}
    .kpi-lbl  {{ color:{MUTED};font-size:.58rem;letter-spacing:1px;text-transform:uppercase;margin-top:.2rem; }}

    .stButton>button {{
        background:linear-gradient(135deg,{ACCENT} 0%,#1a5fd1 100%);
        color:#fff; border:none; border-radius:8px;
        font-family:'Oswald',sans-serif;font-size:.9rem;font-weight:600;
        letter-spacing:2px; text-transform:uppercase;
        padding:.7rem 2rem; width:100%; cursor:pointer;
        transition:all .2s ease; box-shadow:0 2px 16px rgba(79,158,255,.25);
    }}
    .stButton>button:hover {{
        transform:translateY(-2px); box-shadow:0 8px 28px rgba(79,158,255,.45);
        background:linear-gradient(135deg,#6bb3ff 0%,#2268e0 100%);
    }}

    div[data-testid="metric-container"] {{
        background:{CARD_BG}; border:1px solid {BORDER}; border-radius:10px;
        padding:1rem 1.1rem; border-top:2px solid {ACCENT};
        box-shadow:0 2px 12px rgba(0,0,0,.3);
    }}
    div[data-testid="metric-container"] label {{
        font-size:.7rem !important;letter-spacing:1.5px !important;
        text-transform:uppercase !important;color:{MUTED} !important;
    }}
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {{
        font-family:'Oswald',sans-serif !important;font-size:1.7rem !important;
        color:{TEXT} !important;font-weight:600 !important;
    }}

    [data-testid="stSidebar"] {{
        background:{CARD_BG} !important; border-right:1px solid {BORDER} !important;
    }}
    ::-webkit-scrollbar {{ width:6px; height:6px; }}
    ::-webkit-scrollbar-track {{ background:{DARK_BG}; }}
    ::-webkit-scrollbar-thumb {{ background:{BORDER}; border-radius:3px; }}

    [data-testid="stDataFrame"] {{
        border:1px solid {BORDER} !important; border-radius:10px !important; overflow:hidden;
    }}
    .stSelectbox>div>div {{
        background:{CARD_BG} !important; border-color:{BORDER} !important; border-radius:8px !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    # ── Hero ─────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="padding:1.2rem 0 0 0;">
      <div class="main-title">💹 STRAIVE Dynamic Pricing</div>
      <div class="hero-underline"></div>
      <div style="display:flex;align-items:center;justify-content:center;gap:.5rem;margin:.3rem auto .1rem auto;">
        <span style="width:6px;height:6px;border-radius:50%;background:{ACCENT3};display:inline-block;"></span>
        <span style="font-family:'Inter';font-size:.78rem;font-weight:500;color:{ACCENT3};letter-spacing:2.5px;text-transform:uppercase;">
          Optimization · Elasticity Modeling · Revenue Simulation
        </span>
        <span style="width:6px;height:6px;border-radius:50%;background:{ACCENT3};display:inline-block;"></span>
      </div>
      <div class="sub-title">Price Elasticity · Optimal Pricing · Competitive Intelligence · What-If Scenarios</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"""
        <div style="padding:.6rem 0 .4rem 0;text-align:center;">
          <div style="font-family:'Oswald';font-size:1.15rem;font-weight:700;
                     color:{ACCENT};letter-spacing:3px;text-transform:uppercase;">💹 STRAIVE</div>
          <div style="color:{MUTED};font-size:.6rem;letter-spacing:1.5px;text-transform:uppercase;
                     margin-top:.15rem;">Pricing Intelligence Platform</div>
        </div>
        <hr style="border-color:{BORDER};margin:.5rem 0 .9rem 0;">
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="color:{MUTED};font-size:.62rem;letter-spacing:2px;
                   text-transform:uppercase;margin-bottom:.35rem;padding-left:.1rem;">
          ⚙ Model Control
        </div>
        """, unsafe_allow_html=True)
        train_btn    = st.button("▶ BUILD MODEL", type="primary")
        n_records    = st.slider("Transaction records", 1000, 8000, 3600, 200, key="n_rec")
        optimize_for = st.selectbox("Optimise for", ["revenue", "profit", "margin"], key="opt_obj")

        st.markdown(f"""
        <hr style="border-color:{BORDER};margin:.9rem 0 .7rem 0;">
        <div style="color:{MUTED};font-size:.62rem;letter-spacing:2px;
                   text-transform:uppercase;margin-bottom:.5rem;padding-left:.1rem;">
          📂 Analytics Module
        </div>
        """, unsafe_allow_html=True)

        NAV_OPTIONS = [
            "📊 Executive Dashboard",
            "🔍 Elasticity Analysis",
            "💡 Optimal Pricing",
            "📈 Revenue Simulator",
            "🎯 Price-Volume Curves",
            "⚔️  Competitive Positioning",
            "🌍 Regional Pricing",
            "👥 Segment Intelligence",
            "🔧 What-If Scenarios",
            "🤝 Win-Rate Analysis",
            "📉 Margin Waterfall",
            "📦 Product Portfolio",
            "⚠️  Risk & Sensitivity",
            "🗓️ Seasonality & Trends",
        ]
        active_tab = st.radio("Navigation", NAV_OPTIONS, label_visibility="collapsed")

    # ── Gate on model build ───────────────────────────────────────────────────
    if not train_btn and "df" not in st.session_state:
        st.markdown(f"""
        <div style="margin:4rem auto;max-width:560px;text-align:center;">
          <div style="font-size:3.5rem;margin-bottom:1rem;">💹</div>
          <div style="font-family:'Oswald';font-size:1.6rem;font-weight:600;
                     color:{TEXT};letter-spacing:2px;text-transform:uppercase;margin-bottom:.8rem;">
            Ready to Optimise
          </div>
          <div style="color:{MUTED};font-size:.88rem;line-height:1.7;margin-bottom:1.5rem;">
            Click <b style="color:{ACCENT};">BUILD MODEL</b> to generate transaction data,
            fit demand models, and unlock all analytics modules.
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    if train_btn or "df" not in st.session_state:
        with st.spinner("Generating transaction data · Fitting demand models · Calibrating elasticities..."):
            df       = generate_transaction_data(n=n_records)
            elast    = fit_elasticity_model(df)
            win_mod  = fit_win_probability_model(df)
            st.session_state.update({
                "df": df, "elasticity": elast,
                "win_model": win_mod, "optimize_for": optimize_for,
            })
            st.success(f"✅ Model built — {len(df):,} transactions · {df['product'].nunique()} products · {df['customer_type'].nunique()} segments")

    df       = st.session_state["df"]
    elast    = st.session_state["elasticity"]
    win_mod  = st.session_state.get("win_model")

    # ── Top KPI bar ───────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:12px;
               padding:.45rem 1rem .2rem 1rem;margin:.2rem 0 .6rem 0;">
      <span style="color:{MUTED};font-size:.62rem;letter-spacing:2px;text-transform:uppercase;">
       Platform Summary
      </span>
    </div>
    """, unsafe_allow_html=True)

    _mc = st.columns(8)
    total_rev    = df["revenue"].sum()
    total_profit = (df["revenue"] - df["cost"]).sum()
    avg_margin   = df["margin_pct"].mean()
    avg_price    = df["actual_price"].mean()
    avg_disc     = df["discount_pct"].mean()
    win_rate     = df["deal_won"].mean() * 100
    n_products   = df["product"].nunique()
    n_segments   = df["customer_type"].nunique()

    _kpis = [
        ("Total Revenue",  f"${total_rev/1e6:.1f}M",    ACCENT),
        ("Gross Profit",   f"${total_profit/1e6:.1f}M", GREEN),
        ("Avg Margin",     f"{avg_margin:.1f}%",         ACCENT3),
        ("Avg Price",      f"${avg_price:,.0f}",         ACCENT),
        ("Avg Discount",   f"{avg_disc:.1f}%",           YELLOW),
        ("Win Rate",       f"{win_rate:.1f}%",           GREEN),
        ("Products",       str(n_products),               PURPLE),
        ("Segments",       str(n_segments),               ACCENT2),
    ]
    for col, (lbl, val, clr) in zip(_mc, _kpis):
        col.markdown(f"""
        <div class="kpi-card" style="border-top-color:{clr};">
          <div class="kpi-val" style="color:{clr};">{val}</div>
          <div class="kpi-lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="height:1px;background:linear-gradient(90deg,transparent,{ACCENT},transparent);
               margin:.4rem 0 1.2rem 0;opacity:.4;"></div>
    """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 0 — Executive Dashboard
    # ═══════════════════════════════════════════════════════════════════════════
    if active_tab == "📊 Executive Dashboard":
        st.markdown('<div class="section-header">Revenue by Product Segment</div>', unsafe_allow_html=True)
        seg_rev = df.groupby("segment")["revenue"].sum().reset_index().sort_values("revenue", ascending=False)
        seg_rev["color"] = seg_rev["segment"].map(SEGMENT_COLORS)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(seg_rev, x="revenue", y="segment", orientation="h",
                         color="segment", color_discrete_map=SEGMENT_COLORS,
                         title="Revenue by Service Segment", height=360)
            fig.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
            style_fig(fig); st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure(go.Pie(labels=seg_rev["segment"], values=seg_rev["revenue"],
                                   marker_colors=list(SEGMENT_COLORS.values()), hole=0.45,
                                   textinfo="label+percent"))
            fig.update_layout(title="Revenue Share", height=360)
            style_fig(fig); st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Revenue Trend (Monthly)</div>', unsafe_allow_html=True)
        df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
        monthly = df.groupby(["month", "segment"])["revenue"].sum().reset_index()
        fig = px.area(monthly, x="month", y="revenue", color="segment",
                      color_discrete_map=SEGMENT_COLORS,
                      title="Monthly Revenue by Segment", height=380)
        style_fig(fig); st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Margin Distribution by Product</div>', unsafe_allow_html=True)
        prod_stats = df.groupby("product").agg(
            Revenue=("revenue", "sum"), Margin=("margin_pct", "mean"),
            Volume=("volume", "sum"), AvgPrice=("actual_price", "mean")
        ).reset_index().sort_values("Revenue", ascending=False)
        prod_stats["Segment"] = prod_stats["product"].map(lambda x: PRODUCT_CATALOG.get(x, {}).get("segment", "Other"))
        fig = px.scatter(prod_stats, x="Revenue", y="Margin", size="Volume",
                         hover_name="product", color="Segment",
                         color_discrete_map=SEGMENT_COLORS,
                         title="Revenue vs Margin (Bubble = Volume)", height=480)
        fig.add_hline(y=prod_stats["Margin"].mean(), line_dash="dot", line_color=MUTED, opacity=.6)
        style_fig(fig); st.plotly_chart(fig, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1 — Elasticity Analysis
    # ═══════════════════════════════════════════════════════════════════════════
    elif active_tab == "🔍 Elasticity Analysis":
        st.markdown('<div class="section-header">Price Elasticity of Demand by Segment</div>', unsafe_allow_html=True)

        if elast:
            e_df = pd.DataFrame([
                {"Segment": seg, **vals} for seg, vals in elast.items()
            ]).sort_values("elasticity")
            e_df["Interpretation"] = e_df["elasticity"].apply(
                lambda x: "Highly Elastic" if x < -1.5 else ("Elastic" if x < -1.0 else "Inelastic"))

            ec1, ec2 = st.columns(2)
            with ec1:
                fig = px.bar(e_df, x="elasticity", y="Segment", orientation="h",
                             color="elasticity", color_continuous_scale="RdYlGn_r",
                             title="Price Elasticity Coefficients (OLS Log-Log)", height=360)
                fig.add_vline(x=-1.0, line_dash="dash", line_color=ACCENT3, opacity=.8)
                fig.add_annotation(x=-1.0, y=len(e_df)-0.5, text="Unitary", showarrow=False,
                                   font=dict(color=ACCENT3, size=10))
                fig.update_layout(coloraxis_showscale=False, yaxis={"categoryorder":"total descending"})
                style_fig(fig); st.plotly_chart(fig, use_container_width=True)
            with ec2:
                fig2 = px.scatter(e_df, x="elasticity", y="mean_margin",
                                  size="n_obs", hover_name="Segment",
                                  color="Interpretation",
                                  color_discrete_map={"Highly Elastic": RED, "Elastic": YELLOW, "Inelastic": GREEN},
                                  title="Elasticity vs Average Margin", height=360)
                style_fig(fig2); st.plotly_chart(fig2, use_container_width=True)

            st.markdown('<div class="section-header">Elasticity Model Summary</div>', unsafe_allow_html=True)
            display_cols = ["Segment", "elasticity", "disc_lift", "r_squared", "mean_price", "mean_volume", "mean_margin", "n_obs"]
            st.dataframe(e_df[display_cols].rename(columns={
                "elasticity": "Price Elasticity", "disc_lift": "Discount Lift",
                "r_squared": "R²", "mean_price": "Avg Price ($)", "mean_volume": "Avg Volume",
                "mean_margin": "Avg Margin (%)", "n_obs": "Observations"
            }).style.background_gradient(subset=["Price Elasticity"], cmap="RdYlGn_r"),
                use_container_width=True, height=280)
            st.download_button("⬇ Download Elasticity CSV", to_csv_bytes(e_df), "elasticity.csv", "text/csv")
        else:
            st.warning("No elasticity data — rebuild model.")

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2 — Optimal Pricing
    # ═══════════════════════════════════════════════════════════════════════════
    elif active_tab == "💡 Optimal Pricing":
        st.markdown('<div class="section-header">Revenue & Profit Optimal Price Calculator</div>', unsafe_allow_html=True)

        oc1, oc2, oc3 = st.columns([2,1,1])
        with oc1: sel_prod = st.selectbox("Select Product", list(PRODUCT_CATALOG.keys()), key="opt_prod")
        with oc2: sel_seg  = st.selectbox("Customer Segment", list(CUSTOMER_SEGMENTS.keys()), key="opt_seg")
        with oc3: sel_obj  = st.selectbox("Optimise for", ["revenue", "profit", "margin"], key="opt_obj2")

        prod_info = PRODUCT_CATALOG[sel_prod]
        base_p    = prod_info["base_price"]
        cost_p    = prod_info["cost"]
        seg_elast = elast.get(prod_info["segment"], {})
        elasticity_val = seg_elast.get("elasticity", -1.2) * CUSTOMER_SEGMENTS[sel_seg]["price_sensitivity"]

        min_p = st.slider("Min Price ($)", int(base_p*0.4), int(base_p*0.9), int(base_p*0.6), key="min_p")
        max_p = st.slider("Max Price ($)", int(base_p*0.9), int(base_p*2.2), int(base_p*1.8), key="max_p")

        result = optimal_price(base_p, cost_p, elasticity_val, target=sel_obj,
                               constraints={"min_price": min_p, "max_price": max_p})

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Optimal Price",    f"${result['optimal_price']:,.2f}", f"{result['vs_base_pct']:+.1f}% vs base")
        r2.metric("Expected Revenue", f"${result['expected_revenue']:,.2f}")
        r3.metric("Expected Profit",  f"${result['expected_profit']:,.2f}")
        r4.metric("Margin %",         f"{result['margin_pct']:.1f}%")

        # Price-objective curve
        st.markdown('<div class="section-header">Objective vs Price Curve</div>', unsafe_allow_html=True)
        prices_range = np.linspace(min_p, max_p, 200)
        revenues = []; profits = []; margins = []
        for p in prices_range:
            d = max(0.01, (p / base_p) ** elasticity_val)
            rev = p * d; prof = (p - cost_p) * d
            revenues.append(rev); profits.append(prof)
            margins.append((rev - cost_p * d) / rev * 100 if rev > 0 else 0)

        fig = go.Figure()
        rev_max = max(revenues); prof_max = max(profits) if max(profits) > 0 else 1
        fig.add_trace(go.Scatter(x=prices_range, y=[v/rev_max for v in revenues],
                                 mode="lines", name="Revenue (normalised)", line=dict(color=ACCENT, width=2)))
        fig.add_trace(go.Scatter(x=prices_range, y=[v/prof_max for v in profits],
                                 mode="lines", name="Profit (normalised)", line=dict(color=GREEN, width=2)))
        fig.add_trace(go.Scatter(x=prices_range, y=[v/100 for v in margins],
                                 mode="lines", name="Margin (normalised)", line=dict(color=ACCENT3, width=2, dash="dot")))
        fig.add_vline(x=result["optimal_price"], line_dash="dash", line_color=ACCENT2,
                      annotation_text=f"Optimal ${result['optimal_price']:,.0f}",
                      annotation_font=dict(color=ACCENT2, size=11))
        fig.add_vline(x=base_p, line_dash="dot", line_color=MUTED, opacity=.5,
                      annotation_text="Base Price", annotation_font=dict(color=MUTED, size=10))
        fig.update_layout(title=f"Objective Landscape — {sel_prod}", height=380,
                          yaxis_title="Normalised Objective", xaxis_title="Price ($)")
        style_fig(fig); st.plotly_chart(fig, use_container_width=True)

        # Full product optimisation table
        st.markdown('<div class="section-header">Optimal Prices — All Products</div>', unsafe_allow_html=True)
        all_opt = []
        for prod, info in PRODUCT_CATALOG.items():
            bp  = info["base_price"]; cp = info["cost"]; seg = info["segment"]
            el  = elast.get(seg, {}).get("elasticity", -1.2)
            res = optimal_price(bp, cp, el, target=sel_obj)
            all_opt.append({
                "Product":      prod, "Segment": seg,
                "Base Price":   f"${bp:,}", "Cost":         f"${cp:,}",
                "Optimal Price":f"${res['optimal_price']:,.2f}",
                "Change %":     f"{res['vs_base_pct']:+.1f}%",
                "Exp Revenue":  f"${res['expected_revenue']:,.2f}",
                "Margin %":     f"{res['margin_pct']:.1f}%",
                "Elasticity":   el,
            })
        opt_df = pd.DataFrame(all_opt)
        st.dataframe(opt_df, use_container_width=True, height=400)
        st.download_button("⬇ Download Optimal Prices", to_csv_bytes(opt_df), "optimal_prices.csv", "text/csv")

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 3 — Revenue Simulator
    # ═══════════════════════════════════════════════════════════════════════════
    elif active_tab == "📈 Revenue Simulator":
        st.markdown('<div class="section-header">Monte Carlo Revenue Simulation (Sobol QMC)</div>', unsafe_allow_html=True)

        sc1, sc2, sc3 = st.columns([2,1,1])
        with sc1: sim_prod = st.selectbox("Product", list(PRODUCT_CATALOG.keys()), key="sim_prod")
        with sc2: n_sims   = st.selectbox("Simulations", [512, 1024, 2048, 4096], index=2, key="n_sims")
        with sc3: sim_obj  = st.selectbox("View", ["revenue", "profit", "margin"], key="sim_obj")

        info   = PRODUCT_CATALOG[sim_prod]
        bp     = info["base_price"]; cp = info["cost"]
        el     = elast.get(info["segment"], {}).get("elasticity", -1.2)
        p_lo   = st.slider("Price Range — Min ($)", int(bp*0.4), int(bp*0.95), int(bp*0.6), key="s_lo")
        p_hi   = st.slider("Price Range — Max ($)", int(bp), int(bp*2.5), int(bp*1.8), key="s_hi")

        if st.button("🎲 Run Simulation"):
            with st.spinner("Running Sobol QMC simulation..."):
                sim_df = simulate_revenue_scenarios(sim_prod, (p_lo, p_hi), el, cp, n_sim=n_sims)
            st.session_state["sim_df"] = sim_df

        sim_df = st.session_state.get("sim_df")
        if sim_df is not None:
            sa1, sa2 = st.columns(2)
            with sa1:
                fig = px.histogram(sim_df, x=sim_obj, nbins=60,
                                   title=f"Distribution of {sim_obj.title()} ({len(sim_df):,} simulations)",
                                   color_discrete_sequence=[ACCENT], height=360)
                p5  = np.percentile(sim_df[sim_obj], 5)
                p95 = np.percentile(sim_df[sim_obj], 95)
                fig.add_vline(x=sim_df[sim_obj].median(), line_dash="dash", line_color=ACCENT3,
                              annotation_text=f"Median: {sim_df[sim_obj].median():,.1f}",
                              annotation_font=dict(color=ACCENT3, size=10))
                fig.add_vrect(x0=p5, x1=p95, fillcolor=ACCENT, opacity=.08,
                              annotation_text="90% CI", annotation_font=dict(color=MUTED, size=9))
                style_fig(fig); st.plotly_chart(fig, use_container_width=True)
            with sa2:
                fig2 = px.scatter(sim_df.sample(min(800, len(sim_df))), x="price", y=sim_obj,
                                  color="margin", color_continuous_scale="RdYlGn",
                                  title=f"Price vs {sim_obj.title()} (coloured by margin)", height=360)
                style_fig(fig2); st.plotly_chart(fig2, use_container_width=True)

            st.markdown('<div class="section-header">Simulation Statistics</div>', unsafe_allow_html=True)
            ss1, ss2, ss3, ss4 = st.columns(4)
            ss1.metric("Mean Revenue",   f"${sim_df['revenue'].mean():,.2f}")
            ss2.metric("P10 Revenue",    f"${np.percentile(sim_df['revenue'],10):,.2f}")
            ss3.metric("P90 Revenue",    f"${np.percentile(sim_df['revenue'],90):,.2f}")
            ss4.metric("Best Price",     f"${sim_df.loc[sim_df[sim_obj].idxmax(),'price']:,.2f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 4 — Price-Volume Curves
    # ═══════════════════════════════════════════════════════════════════════════
    elif active_tab == "🎯 Price-Volume Curves":
        st.markdown('<div class="section-header">Price-Volume-Revenue Demand Curves</div>', unsafe_allow_html=True)

        pv_seg  = st.selectbox("Service Segment", PRODUCT_SEGMENTS, key="pv_seg")
        el_val  = elast.get(pv_seg, {}).get("elasticity", -1.2)
        seg_products = [k for k,v in PRODUCT_CATALOG.items() if v["segment"] == pv_seg]
        pv_prod = st.selectbox("Product", seg_products, key="pv_prod") if seg_products else None

        if pv_prod:
            bp   = PRODUCT_CATALOG[pv_prod]["base_price"]
            cp   = PRODUCT_CATALOG[pv_prod]["cost"]
            p_range = np.linspace(bp * 0.4, bp * 2.2, 300)
            demand  = np.maximum(0.01, (p_range / bp) ** el_val)
            rev     = p_range * demand
            profit  = (p_range - cp) * demand
            margin  = np.where(rev > 0, (rev - cp * demand) / rev * 100, 0)

            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=["Price vs Demand Index", "Price vs Revenue & Profit"])
            fig.add_trace(go.Scatter(x=p_range, y=demand, mode="lines",
                                     line=dict(color=ACCENT, width=2.5), name="Demand Index"), row=1, col=1)
            fig.add_trace(go.Scatter(x=p_range, y=rev, mode="lines",
                                     line=dict(color=GREEN, width=2.5), name="Revenue"), row=1, col=2)
            fig.add_trace(go.Scatter(x=p_range, y=profit, mode="lines",
                                     line=dict(color=ACCENT3, width=2, dash="dot"), name="Profit"), row=1, col=2)

            opt_r = optimal_price(bp, cp, el_val, target="revenue")
            opt_p = optimal_price(bp, cp, el_val, target="profit")
            fig.add_vline(x=opt_r["optimal_price"], line_dash="dash", line_color=GREEN,
                          annotation_text="Rev Max", annotation_font=dict(color=GREEN, size=9), row=1, col=2)
            fig.add_vline(x=opt_p["optimal_price"], line_dash="dash", line_color=ACCENT3,
                          annotation_text="Profit Max", annotation_font=dict(color=ACCENT3, size=9), row=1, col=2)
            fig.add_vline(x=bp, line_dash="dot", line_color=MUTED, opacity=.5, row=1, col=1)
            fig.add_vline(x=bp, line_dash="dot", line_color=MUTED, opacity=.5, row=1, col=2)

            fig.update_layout(height=420, title=f"Demand Curves — {pv_prod}")
            style_fig(fig); st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="section-header">Multi-Segment Price Sensitivity</div>', unsafe_allow_html=True)
            fig2 = go.Figure()
            for seg_name, seg_info in CUSTOMER_SEGMENTS.items():
                el_adj = el_val * seg_info["price_sensitivity"]
                d_seg  = np.maximum(0.01, (p_range / bp) ** el_adj)
                fig2.add_trace(go.Scatter(x=p_range, y=d_seg, mode="lines",
                                          name=seg_name, line=dict(width=2)))
            fig2.add_vline(x=bp, line_dash="dot", line_color=MUTED, opacity=.5)
            fig2.update_layout(title="Demand by Customer Segment", height=380,
                               xaxis_title="Price ($)", yaxis_title="Demand Index")
            style_fig(fig2); st.plotly_chart(fig2, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 5 — Competitive Positioning
    # ═══════════════════════════════════════════════════════════════════════════
    elif active_tab == "⚔️  Competitive Positioning":
        st.markdown('<div class="section-header">Competitive Price Benchmarking</div>', unsafe_allow_html=True)

        cp_prod  = st.selectbox("Product", list(PRODUCT_CATALOG.keys()), key="cp_prod")
        cp_price = st.slider(
            "STRAIVE Price ($)",
            int(PRODUCT_CATALOG[cp_prod]["base_price"] * 0.5),
            int(PRODUCT_CATALOG[cp_prod]["base_price"] * 2.0),
            int(PRODUCT_CATALOG[cp_prod]["base_price"]),
            key="cp_price"
        )

        comp = competitive_price_score(cp_price, PRODUCT_CATALOG[cp_prod]["base_price"])
        comp_df = comp["df"]

        ca1, ca2, ca3 = st.columns(3)
        ca1.metric("STRAIVE vs Market Avg", f"{(comp_df['Gap %'].mean()):+.1f}%")
        ca2.metric("Cheaper Than",          f"{int((comp_df['Gap %']<0).sum())} of {len(comp_df)} competitors")
        ca3.metric("Price Percentile",      f"{comp['pct_rank']:.0f}th")

        fig = go.Figure()
        comp_sorted = comp_df.sort_values("Comp Price")
        fig.add_trace(go.Bar(x=comp_sorted["Competitor"], y=comp_sorted["Comp Price"],
                             name="Competitor Price", marker_color=MUTED, opacity=.7))
        fig.add_trace(go.Scatter(x=comp_sorted["Competitor"],
                                 y=[cp_price]*len(comp_sorted),
                                 mode="lines+markers", name="STRAIVE Price",
                                 line=dict(color=ACCENT, width=3), marker=dict(size=8)))
        fig.update_layout(title="Competitive Price Comparison", height=380, barmode="group",
                          yaxis_title="Price ($)")
        style_fig(fig); st.plotly_chart(fig, use_container_width=True)

        # Quality-Price map
        comp_df["STRAIVE_Quality"] = 8.5
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=comp_df["Comp Relative"], y=comp_df["Quality Score"],
                                  mode="markers+text", text=comp_df["Competitor"],
                                  textposition="top center", name="Competitors",
                                  marker=dict(size=14, color=ACCENT2)))
        fig2.add_trace(go.Scatter(x=[comp["straive_relative"]], y=[8.5],
                                  mode="markers+text", text=["STRAIVE"],
                                  textposition="top center", name="STRAIVE",
                                  marker=dict(size=18, color=ACCENT, symbol="star")))
        fig2.add_vline(x=1.0, line_dash="dot", line_color=MUTED, opacity=.5)
        fig2.update_layout(title="Quality vs Relative Price Map", height=420,
                           xaxis_title="Relative Price (1.0 = STRAIVE Base)",
                           yaxis_title="Quality Score")
        style_fig(fig2); st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(comp_df.style.background_gradient(subset=["Gap %"], cmap="RdYlGn_r"),
                     use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 6 — Regional Pricing
    # ═══════════════════════════════════════════════════════════════════════════
    elif active_tab == "🌍 Regional Pricing":
        st.markdown('<div class="section-header">Regional Pricing Strategy</div>', unsafe_allow_html=True)

        rp_prod = st.selectbox("Product", list(PRODUCT_CATALOG.keys()), key="rp_prod")
        bp      = PRODUCT_CATALOG[rp_prod]["base_price"]
        cp      = PRODUCT_CATALOG[rp_prod]["cost"]
        seg_    = PRODUCT_CATALOG[rp_prod]["segment"]
        el_val  = elast.get(seg_, {}).get("elasticity", -1.2)

        reg_rows = []
        for reg, rinfo in REGIONS.items():
            adj_elast = el_val * (0.8 + 0.4 * rinfo["competitor_pressure"])
            adj_price = optimal_price(
                bp * rinfo["demand_index"], cp, adj_elast,
                target="profit",
                constraints={"min_price": bp*0.5, "max_price": bp*2.0}
            )
            reg_rows.append({
                "Region":         reg,
                "Demand Index":   rinfo["demand_index"],
                "Comp Pressure":  rinfo["competitor_pressure"],
                "Growth Rate":    f"{rinfo['growth_rate']*100:.0f}%",
                "Rec. Price ($)": adj_price["optimal_price"],
                "vs Base %":      f"{adj_price['vs_base_pct']:+.1f}%",
                "Est. Margin %":  f"{adj_price['margin_pct']:.1f}%",
            })

        reg_df = pd.DataFrame(reg_rows)

        fig = px.bar(reg_df, x="Region", y="Rec. Price ($)",
                     color="Demand Index", color_continuous_scale="YlOrRd",
                     title=f"Recommended Price by Region — {rp_prod}", height=400)
        fig.add_hline(y=bp, line_dash="dot", line_color=MUTED,
                      annotation_text=f"Base ${bp:,}", annotation_font=dict(color=MUTED))
        style_fig(fig); st.plotly_chart(fig, use_container_width=True)

        # Regional revenue split from data
        reg_rev = df.groupby("region")["revenue"].sum().reset_index()
        fig2 = go.Figure(go.Pie(labels=reg_rev["region"], values=reg_rev["revenue"], hole=0.45,
                                textinfo="label+percent"))
        fig2.update_layout(title="Actual Revenue Split by Region", height=360)
        style_fig(fig2); st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(reg_df.style.background_gradient(subset=["Rec. Price ($)", "Demand Index"], cmap="YlOrRd"),
                     use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 7 — Segment Intelligence
    # ═══════════════════════════════════════════════════════════════════════════
    elif active_tab == "👥 Segment Intelligence":
        st.markdown('<div class="section-header">Customer Segment Revenue & Pricing</div>', unsafe_allow_html=True)

        seg_stats = df.groupby("customer_type").agg(
            Revenue   =("revenue", "sum"),
            Volume    =("volume", "sum"),
            AvgPrice  =("actual_price", "mean"),
            AvgDisc   =("discount_pct", "mean"),
            AvgMargin =("margin_pct", "mean"),
            WinRate   =("deal_won", "mean"),
            Deals     =("deal_won", "count"),
        ).reset_index().rename(columns={"customer_type": "Customer Segment"})
        seg_stats["WinRate"] *= 100

        colors = [CUSTOMER_SEGMENTS[s]["color"] for s in seg_stats["Customer Segment"]]

        sa1, sa2 = st.columns(2)
        with sa1:
            fig = px.bar(seg_stats, x="Revenue", y="Customer Segment", orientation="h",
                         color="Customer Segment",
                         color_discrete_sequence=colors,
                         title="Revenue by Customer Segment", height=380)
            fig.update_layout(showlegend=False, yaxis={"categoryorder":"total ascending"})
            style_fig(fig); st.plotly_chart(fig, use_container_width=True)
        with sa2:
            fig2 = px.scatter(seg_stats, x="AvgPrice", y="AvgMargin",
                              size="Revenue", hover_name="Customer Segment",
                              color="WinRate", color_continuous_scale="RdYlGn",
                              title="Avg Price vs Margin (Bubble = Revenue)", height=380)
            style_fig(fig2); st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-header">Price Sensitivity Profiles</div>', unsafe_allow_html=True)
        sens_df = pd.DataFrame([
            {"Segment": k, "Price Sensitivity": v["price_sensitivity"],
             "Volume Multiplier": v["volume_multiplier"], "Loyalty": v["loyalty"]}
            for k, v in CUSTOMER_SEGMENTS.items()
        ])
        fig3 = go.Figure()
        cats   = ["Price Sensitivity", "Volume Multiplier (÷2)", "Loyalty"]
        colors2= [CUSTOMER_SEGMENTS[s]["color"] for s in sens_df["Segment"]]
        for i, row in sens_df.iterrows():
            vals = [row["Price Sensitivity"], row["Volume Multiplier"]/2, row["Loyalty"], row["Price Sensitivity"]]
            fig3.add_trace(go.Scatterpolar(r=vals, theta=cats+[cats[0]],
                                           fill="toself", name=row["Segment"],
                                           line=dict(color=colors2[i], width=2)))
        fig3.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                           title="Segment Behavioural Profiles", height=500)
        style_fig(fig3); st.plotly_chart(fig3, use_container_width=True)

        st.dataframe(seg_stats.style.background_gradient(subset=["Revenue","AvgMargin","WinRate"], cmap="RdYlGn"),
                     use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 8 — What-If Scenarios
    # ═══════════════════════════════════════════════════════════════════════════
    elif active_tab == "🔧 What-If Scenarios":
        st.markdown('<div class="section-header">Pricing Scenario Planning</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:12px;
                   padding:1rem 1.2rem;margin-bottom:1.2rem;color:{MUTED};font-size:.82rem;line-height:1.7;">
          Adjust pricing for individual products and immediately see the projected impact on
          revenue, volume, and margin vs the current baseline.
        </div>
        """, unsafe_allow_html=True)

        overrides = {}
        st.markdown("**Set new prices for selected products:**")
        for i, (prod, info) in enumerate(list(PRODUCT_CATALOG.items())[:6]):
            bp = info["base_price"]
            new_p = st.slider(
                f"{prod}  (base: ${bp:,})",
                int(bp * 0.5), int(bp * 2.0), bp,
                step=max(50, int(bp * 0.05)), key=f"wi_{i}"
            )
            if new_p != bp:
                overrides[prod] = new_p

        if overrides and st.button("▶ Run Scenario"):
            df2      = apply_pricing_scenario(df, overrides)
            baseline = df["revenue"].sum()
            scenario = df2["revenue"].sum()
            bl_prof  = (df["revenue"] - df["cost"]).sum()
            sc_prof  = (df2["revenue"] - df2["cost"]).sum()

            wic1, wic2, wic3, wic4 = st.columns(4)
            delta_rev  = (scenario - baseline) / baseline * 100
            delta_prof = (sc_prof - bl_prof) / bl_prof * 100
            wic1.metric("Baseline Revenue",  f"${baseline/1e6:.2f}M")
            wic2.metric("Scenario Revenue",  f"${scenario/1e6:.2f}M", f"{delta_rev:+.1f}%")
            wic3.metric("Baseline Profit",   f"${bl_prof/1e6:.2f}M")
            wic4.metric("Scenario Profit",   f"${sc_prof/1e6:.2f}M", f"{delta_prof:+.1f}%")

            # Segment comparison
            bl_seg = df.groupby("segment")["revenue"].sum().rename("Baseline")
            sc_seg = df2.groupby("segment")["revenue"].sum().rename("Scenario")
            cmp_df = pd.concat([bl_seg, sc_seg], axis=1).fillna(0).reset_index()
            cmp_df["Delta %"] = ((cmp_df["Scenario"] - cmp_df["Baseline"]) / cmp_df["Baseline"] * 100).round(1)

            fig = go.Figure()
            fig.add_trace(go.Bar(x=cmp_df["segment"], y=cmp_df["Baseline"], name="Baseline",
                                 marker_color=MUTED, opacity=.8))
            fig.add_trace(go.Bar(x=cmp_df["segment"], y=cmp_df["Scenario"], name="Scenario",
                                 marker_color=ACCENT))
            fig.update_layout(barmode="group", title="Revenue Impact by Segment", height=400)
            style_fig(fig); st.plotly_chart(fig, use_container_width=True)
        elif not overrides:
            st.info("Adjust at least one product price above, then click Run Scenario.")

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 9 — Win-Rate Analysis
    # ═══════════════════════════════════════════════════════════════════════════
    elif active_tab == "🤝 Win-Rate Analysis":
        st.markdown('<div class="section-header">Deal Win-Rate vs Pricing</div>', unsafe_allow_html=True)

        df["price_ratio"] = df["actual_price"] / df["base_price"]
        bins = np.arange(0.5, 2.1, 0.1)
        df["price_bin"] = pd.cut(df["price_ratio"], bins=bins)
        wr_df = df.groupby("price_bin", observed=False).agg(
            WinRate=("deal_won", "mean"), Count=("deal_won", "count")
        ).reset_index()
        wr_df["price_mid"] = wr_df["price_bin"].apply(lambda x: (x.left + x.right) / 2 if hasattr(x,'left') else 1.0)
        wr_df = wr_df.dropna(subset=["price_mid"])

        fig = go.Figure()
        fig.add_trace(go.Bar(x=wr_df["price_mid"], y=wr_df["WinRate"]*100,
                             name="Win Rate %", marker_color=ACCENT,
                             text=[f"{v:.0f}%" for v in wr_df["WinRate"]*100],
                             textposition="outside"))
        fig.add_hline(y=df["deal_won"].mean()*100, line_dash="dash",
                      line_color=ACCENT3, annotation_text="Overall Win Rate",
                      annotation_font=dict(color=ACCENT3, size=10))
        fig.update_layout(title="Win Rate by Price Ratio Bucket", height=420,
                          xaxis_title="Price / Base Price Ratio",
                          yaxis_title="Win Rate %")
        style_fig(fig); st.plotly_chart(fig, use_container_width=True)

        # By segment
        seg_wr = df.groupby("customer_type")["deal_won"].mean().reset_index()
        seg_wr.columns = ["Segment", "Win Rate"]
        seg_wr["Win Rate %"] = (seg_wr["Win Rate"] * 100).round(1)
        fig2 = px.bar(seg_wr, x="Win Rate %", y="Segment", orientation="h",
                      color="Win Rate %", color_continuous_scale="RdYlGn",
                      title="Win Rate by Customer Segment", height=360)
        fig2.update_layout(coloraxis_showscale=False, yaxis={"categoryorder":"total ascending"})
        style_fig(fig2); st.plotly_chart(fig2, use_container_width=True)

        disc_wr = df.groupby(pd.cut(df["discount_pct"], bins=[0,5,10,15,20,30,50], include_lowest=True),
                              observed=False)["deal_won"].mean().reset_index()
        disc_wr.columns = ["Discount Band", "Win Rate"]
        disc_wr["Win Rate %"] = (disc_wr["Win Rate"] * 100).round(1)
        disc_wr["Discount Band"] = disc_wr["Discount Band"].astype(str)
        fig3 = px.bar(disc_wr, x="Discount Band", y="Win Rate %",
                      color="Win Rate %", color_continuous_scale="RdYlGn",
                      title="Win Rate by Discount Level", height=360)
        style_fig(fig3); st.plotly_chart(fig3, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 10 — Margin Waterfall
    # ═══════════════════════════════════════════════════════════════════════════
    elif active_tab == "📉 Margin Waterfall":
        st.markdown('<div class="section-header">Margin Waterfall Analysis</div>', unsafe_allow_html=True)

        mw_prod = st.selectbox("Product", list(PRODUCT_CATALOG.keys()), key="mw_prod")
        sub = df[df["product"] == mw_prod]
        if len(sub) < 5:
            st.warning("Not enough data for this product.")
        else:
            bp    = PRODUCT_CATALOG[mw_prod]["base_price"]
            cp    = PRODUCT_CATALOG[mw_prod]["cost"]
            avg_p = sub["actual_price"].mean()
            avg_v = sub["volume"].mean()

            # Waterfall components
            components = {
                "Base Revenue":         bp * avg_v,
                "Price Premium/Discount": (avg_p - bp) * avg_v,
                "Gross Revenue":        avg_p * avg_v,
                "Cost of Delivery":     -cp * avg_v,
                "Gross Profit":         (avg_p - cp) * avg_v,
            }
            labels = list(components.keys())
            values = list(components.values())
            measures = ["absolute", "relative", "total", "relative", "total"]
            colors_wf = [GREEN, RED if values[1] < 0 else GREEN, ACCENT3, RED, ACCENT]

            fig = go.Figure(go.Waterfall(
                name="Margin Waterfall",
                orientation="v",
                measure=measures,
                x=labels,
                y=values,
                connector=dict(line=dict(color=BORDER, width=1.5)),
                decreasing=dict(marker=dict(color=RED)),
                increasing=dict(marker=dict(color=GREEN)),
                totals=dict(marker=dict(color=ACCENT3)),
                text=[f"${v:,.0f}" for v in values],
                textposition="outside",
            ))
            fig.update_layout(title=f"Margin Waterfall — {mw_prod} (per avg deal)", height=480)
            style_fig(fig); st.plotly_chart(fig, use_container_width=True)

            # Margin by segment heatmap
            st.markdown('<div class="section-header">Margin Heatmap — Product × Region</div>', unsafe_allow_html=True)
            pivot = df.pivot_table(values="margin_pct", index="segment", columns="region",
                                   aggfunc="mean").fillna(0)
            fig2 = go.Figure(go.Heatmap(
                z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
                colorscale="RdYlGn", text=np.round(pivot.values,1),
                texttemplate="%{text}%", textfont=dict(size=9)
            ))
            fig2.update_layout(title="Average Margin % (Segment × Region)", height=360)
            style_fig(fig2); st.plotly_chart(fig2, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 11 — Product Portfolio
    # ═══════════════════════════════════════════════════════════════════════════
    elif active_tab == "📦 Product Portfolio":
        st.markdown('<div class="section-header">Product Portfolio Matrix (BCG-style)</div>', unsafe_allow_html=True)

        prod_pf = df.groupby("product").agg(
            Revenue   =("revenue",   "sum"),
            Margin    =("margin_pct","mean"),
            Volume    =("volume",    "sum"),
            Growth    =("date",      lambda x: 1.0),  # placeholder
        ).reset_index()
        prod_pf["Segment"] = prod_pf["product"].map(lambda x: PRODUCT_CATALOG.get(x,{}).get("segment",""))
        prod_pf["Rev_Share"] = prod_pf["Revenue"] / prod_pf["Revenue"].sum() * 100
        med_margin = prod_pf["Margin"].median()
        med_rev    = prod_pf["Revenue"].median()
        prod_pf["Quadrant"] = prod_pf.apply(
            lambda r: "Star ⭐" if r["Revenue"] >= med_rev and r["Margin"] >= med_margin
            else ("Cash Cow 🐄" if r["Revenue"] >= med_rev else
                  ("Question Mark ❓" if r["Margin"] >= med_margin else "Dog 🐕")), axis=1
        )

        fig = px.scatter(prod_pf, x="Revenue", y="Margin",
                         size="Volume", hover_name="product",
                         color="Quadrant",
                         color_discrete_map={"Star ⭐": ACCENT3, "Cash Cow 🐄": GREEN,
                                             "Question Mark ❓": ACCENT, "Dog 🐕": RED},
                         title="Product Portfolio Matrix", height=540)
        fig.add_hline(y=med_margin, line_dash="dot", line_color=MUTED, opacity=.5)
        fig.add_vline(x=med_rev,    line_dash="dot", line_color=MUTED, opacity=.5)
        for _, row in prod_pf.nlargest(8, "Revenue").iterrows():
            fig.add_annotation(x=row["Revenue"], y=row["Margin"], text=row["product"].split("–")[0].strip(),
                               showarrow=False, font=dict(size=8, color=TEXT), yshift=12)
        style_fig(fig); st.plotly_chart(fig, use_container_width=True)

        st.dataframe(prod_pf[["product","Segment","Revenue","Margin","Volume","Rev_Share","Quadrant"]]
                     .rename(columns={"product":"Product","Rev_Share":"Rev Share %"})
                     .sort_values("Revenue", ascending=False)
                     .style.background_gradient(subset=["Revenue","Margin","Rev Share %"], cmap="YlOrRd"),
                     use_container_width=True, height=400)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 12 — Risk & Sensitivity
    # ═══════════════════════════════════════════════════════════════════════════
    elif active_tab == "⚠️  Risk & Sensitivity":
        st.markdown('<div class="section-header">Sensitivity Analysis & Revenue at Risk</div>', unsafe_allow_html=True)

        rsk_prod = st.selectbox("Product", list(PRODUCT_CATALOG.keys()), key="rsk_prod")
        bp       = PRODUCT_CATALOG[rsk_prod]["base_price"]
        cp       = PRODUCT_CATALOG[rsk_prod]["cost"]
        seg_     = PRODUCT_CATALOG[rsk_prod]["segment"]

        # Tornado chart: which parameter drives revenue most?
        base_el    = elast.get(seg_, {}).get("elasticity", -1.2)
        base_price = bp

        params_var = {
            "Price +10%":         (bp * 1.1,  base_el),
            "Price -10%":         (bp * 0.9,  base_el),
            "Elasticity +25%":    (base_price, base_el * 1.25),
            "Elasticity -25%":    (base_price, base_el * 0.75),
            "Cost +20%":          (base_price, base_el),
            "Cost -20%":          (base_price, base_el),
        }
        base_rev = bp * 1.0  # normalised
        tornado_rows = []
        for label, (p, e) in params_var.items():
            d   = max(0.01, (p / bp) ** e)
            c_v = cp * 1.2 if "Cost +20" in label else (cp * 0.8 if "Cost -20" in label else cp)
            rev = p * d; profit = (p - c_v) * d
            tornado_rows.append({"Scenario": label, "Revenue": rev, "Profit": profit})

        t_df  = pd.DataFrame(tornado_rows)
        t_base = bp * 1.0
        t_df["Rev Delta %"] = ((t_df["Revenue"] - base_rev) / base_rev * 100).round(1)

        fig = px.bar(t_df.sort_values("Rev Delta %"), x="Rev Delta %", y="Scenario",
                     orientation="h", color="Rev Delta %",
                     color_continuous_scale="RdYlGn",
                     title="Tornado: Revenue Sensitivity to Key Parameters", height=380)
        fig.add_vline(x=0, line_color=MUTED, line_width=1.5)
        fig.update_layout(coloraxis_showscale=False)
        style_fig(fig); st.plotly_chart(fig, use_container_width=True)

        # Monte Carlo VaR
        st.markdown('<div class="section-header">Revenue at Risk (Monte Carlo)</div>', unsafe_allow_html=True)
        n_mc = 5000
        np.random.seed(99)
        noise_prices = np.random.normal(bp, bp * 0.12, n_mc)
        noise_elast  = np.random.normal(base_el, abs(base_el) * 0.18, n_mc)
        mc_rev = noise_prices * np.maximum(0.01, (noise_prices / bp) ** noise_elast)

        var95 = np.percentile(mc_rev, 5)
        var99 = np.percentile(mc_rev, 1)

        fig2 = px.histogram(mc_rev, nbins=80, title="Revenue Distribution (Monte Carlo, 5,000 draws)",
                            color_discrete_sequence=[ACCENT], height=360)
        fig2.add_vline(x=var95, line_dash="dash", line_color=YELLOW,
                       annotation_text=f"VaR 95%: ${var95:,.0f}", annotation_font=dict(color=YELLOW, size=10))
        fig2.add_vline(x=var99, line_dash="dash", line_color=RED,
                       annotation_text=f"VaR 99%: ${var99:,.0f}", annotation_font=dict(color=RED, size=10))
        fig2.add_vline(x=np.mean(mc_rev), line_dash="solid", line_color=GREEN,
                       annotation_text=f"Mean: ${np.mean(mc_rev):,.0f}", annotation_font=dict(color=GREEN, size=10))
        style_fig(fig2); st.plotly_chart(fig2, use_container_width=True)

        rv1, rv2, rv3 = st.columns(3)
        rv1.metric("VaR (95%)",  f"${var95:,.2f}", f"{(var95/np.mean(mc_rev)-1)*100:+.1f}% vs Mean")
        rv2.metric("VaR (99%)",  f"${var99:,.2f}", f"{(var99/np.mean(mc_rev)-1)*100:+.1f}% vs Mean")
        rv3.metric("Prob > Base",f"{(mc_rev > bp).mean()*100:.0f}%")

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 13 — Seasonality & Trends
    # ═══════════════════════════════════════════════════════════════════════════
    elif active_tab == "🗓️ Seasonality & Trends":
        st.markdown('<div class="section-header">Revenue Seasonality & Pricing Trends</div>', unsafe_allow_html=True)

        df["month_num"] = df["date"].dt.month
        df["quarter"]   = df["date"].dt.quarter
        df["year"]      = df["date"].dt.year

        monthly_agg = df.groupby("month_num").agg(
            Revenue     =("revenue",      "mean"),
            AvgPrice    =("actual_price", "mean"),
            AvgDiscount =("discount_pct", "mean"),
            Volume      =("volume",       "mean"),
        ).reset_index()
        monthly_agg["Month"] = monthly_agg["month_num"].map({
            1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
            7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"
        })

        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=["Avg Revenue by Month","Avg Price by Month",
                                            "Avg Discount % by Month","Avg Volume by Month"])
        for i, (col, clr) in enumerate([("Revenue",GREEN),("AvgPrice",ACCENT),
                                         ("AvgDiscount",YELLOW),("Volume",ACCENT2)]):
            r, c = divmod(i, 2)
            fig.add_trace(go.Bar(x=monthly_agg["Month"], y=monthly_agg[col],
                                 marker_color=clr, name=col), row=r+1, col=c+1)
        fig.update_layout(title="Monthly Seasonality Patterns", height=560, showlegend=False)
        style_fig(fig); st.plotly_chart(fig, use_container_width=True)

        # Price drift over time
        st.markdown('<div class="section-header">Pricing Trend Over Time</div>', unsafe_allow_html=True)
        trend = df.groupby("month").agg(
            AvgPrice    =("actual_price","mean"),
            AvgDiscount =("discount_pct","mean"),
            Revenue     =("revenue","sum"),
        ).reset_index()
        fig2 = make_subplots(rows=1, cols=2,
                             subplot_titles=["Average Price Trend", "Monthly Revenue Trend"])
        fig2.add_trace(go.Scatter(x=trend["month"], y=trend["AvgPrice"],
                                  mode="lines+markers", line=dict(color=ACCENT, width=2), name="Avg Price"), row=1, col=1)
        fig2.add_trace(go.Bar(x=trend["month"], y=trend["Revenue"],
                              marker_color=GREEN, name="Revenue"), row=1, col=2)
        fig2.update_layout(title="Price & Revenue Over Time", height=400, showlegend=False)
        style_fig(fig2); st.plotly_chart(fig2, use_container_width=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns([5,4,5])
    with fc1:
        st.markdown(f"""
        <div style="padding:.8rem;">
          <div style="font-family:'Oswald';font-size:.95rem;font-weight:700;color:{ACCENT};">
           💹 STRAIVE · Dynamic Pricing Platform
          </div>
          <div style="color:{MUTED};font-size:.68rem;">
           Demand Elasticity · Optimal Pricing · Competitive Intelligence
          </div>
        </div>
        """, unsafe_allow_html=True)
    with fc2:
        st.markdown(f"""
        <div style="padding:.8rem;text-align:center;border-left:1px solid {BORDER};border-right:1px solid {BORDER};">
          <div style="color:{MUTED};font-size:.6rem;">Platform</div>
          <div style="font-family:'Oswald';font-size:1.15rem;font-weight:700;color:{ACCENT3};">
           STRAIVE Analytics
          </div>
        </div>
        """, unsafe_allow_html=True)
    with fc3:
        st.markdown(f"""
        <div style="padding:.8rem;text-align:right;">
          <div style="color:{MUTED};font-size:.68rem;">Built: {CURRENT_DATE.strftime('%Y-%m-%d')}</div>
          <div style="color:{ACCENT};font-size:.65rem;">straive_artifacts/pricing_params.json</div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
