"""
STRAIVE — Dynamic Pricing Optimization Platform
Advanced Analytics: Demand Modeling + Price Elasticity + Revenue Simulation + What-If Engine
Run: streamlit run straive_pricing_app.py

NOTE: This app does NOT require matplotlib. All table styling uses pandas .bar()
which is pure CSS and works on any Python/pandas version without extra dependencies.
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

# ── Optional matplotlib guard ────────────────────────────────────────────────
# pandas .background_gradient() requires matplotlib; we intentionally avoid it
# so the app runs on environments where matplotlib is not installed.
# All gradient-style table colouring uses pandas .bar() instead (pure CSS).
try:
    import matplotlib  # noqa: F401
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

# ─────────────────────────────────────────────────────────────────────────────
# SAFE DATAFRAME STYLER  (no matplotlib required)
# ─────────────────────────────────────────────────────────────────────────────
def style_df(
    df: pd.DataFrame,
    bar_cols: Optional[List[str]] = None,
    pos_color: str = "#36d97b",
    neg_color: str = "#ff4d6d",
    mid_align: bool = False,
) -> "pd.io.formats.style.Styler":
    """
    Apply bar-chart column highlighting without requiring matplotlib.
    Falls back gracefully if columns don't exist or are non-numeric.

    Parameters
    ----------
    bar_cols   : numeric column names to render as inline bar charts
    pos_color  : colour for high / positive values
    neg_color  : colour for low / negative values (used when mid_align=True)
    mid_align  : centre bars at zero — ideal for ± delta columns
    """
    styler = df.style
    if bar_cols:
        valid = [
            c for c in bar_cols
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]
        if valid:
            if mid_align:
                styler = styler.bar(subset=valid, align="mid",
                                    color=[neg_color, pos_color])
            else:
                styler = styler.bar(subset=valid, color=pos_color, width=90)
    return styler


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
DARK_BG  = "#030712"
CARD_BG  = "#0d1117"
CARD_BG2 = "#111827"
BORDER   = "#1f2d45"
BORDER2  = "#243448"
ACCENT   = "#3b82f6"
ACCENT2  = "#f43f5e"
ACCENT3  = "#eab308"
TEXT     = "#f1f5f9"
TEXT2    = "#cbd5e1"
MUTED    = "#64748b"
GREEN    = "#22c55e"
YELLOW   = "#f59e0b"
RED      = "#ef4444"
PURPLE   = "#a78bfa"
TEAL     = "#14b8a6"

PLOTLY_DARK = dict(
    paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
    font=dict(color=TEXT2, family="'Plus Jakarta Sans','Segoe UI',sans-serif", size=12),
    title_font=dict(color=TEXT, size=14, family="'Space Grotesk',sans-serif", weight=700),
    legend=dict(bgcolor="rgba(13,17,23,0.9)", bordercolor=BORDER2, borderwidth=1,
                font=dict(color=TEXT2, size=11)),
    margin=dict(l=16, r=16, t=52, b=16),
    hoverlabel=dict(bgcolor=CARD_BG2, bordercolor=BORDER2, font=dict(color=TEXT, size=12)),
)

def style_fig(fig: go.Figure, height: Optional[int] = None) -> go.Figure:
    upd = dict(**PLOTLY_DARK)
    if height: upd["height"] = height
    fig.update_layout(**upd)
    fig.update_xaxes(
        gridcolor="rgba(31,45,69,.6)", zerolinecolor=BORDER2,
        linecolor=BORDER2, tickfont=dict(color=MUTED, size=11),
        showgrid=True, gridwidth=1,
    )
    fig.update_yaxes(
        gridcolor="rgba(31,45,69,.6)", zerolinecolor=BORDER2,
        linecolor=BORDER2, tickfont=dict(color=MUTED, size=11),
        showgrid=True, gridwidth=1,
    )
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
        page_title="STRAIVE · Dynamic Pricing Platform",
        page_icon="💹",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": None,
            "Report a bug": None,
            "About": "STRAIVE Dynamic Pricing Optimization Platform v2.0",
        },
    )

    # ── CSS ──────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Reset & Base ─────────────────────────────────── */
    html, body, [class*="css"] {{
        background-color: {DARK_BG} !important;
        color: {TEXT};
        font-family: 'Plus Jakarta Sans', 'Segoe UI', sans-serif;
        font-size: 14px;
    }}
    .stApp {{ background: {DARK_BG}; }}
    .block-container {{
        padding: 1.5rem 2.5rem 3rem 2.5rem !important;
        max-width: 1600px !important;
    }}

    /* ── Hero Banner ───────────────────────────────────── */
    .hero-wrapper {{
        background: linear-gradient(135deg, #0d1117 0%, #0f172a 40%, #111827 100%);
        border: 1px solid {BORDER2};
        border-radius: 20px;
        padding: 2.4rem 2.8rem 2rem 2.8rem;
        margin-bottom: 1.8rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 40px rgba(59,130,246,.08), inset 0 1px 0 rgba(255,255,255,.04);
    }}
    .hero-wrapper::before {{
        content:'';
        position:absolute; top:-80px; right:-80px;
        width:320px; height:320px; border-radius:50%;
        background: radial-gradient(circle, rgba(59,130,246,.12) 0%, transparent 70%);
        pointer-events:none;
    }}
    .hero-wrapper::after {{
        content:'';
        position:absolute; bottom:-60px; left:20%;
        width:220px; height:220px; border-radius:50%;
        background: radial-gradient(circle, rgba(34,197,94,.07) 0%, transparent 70%);
        pointer-events:none;
    }}
    .hero-badge {{
        display:inline-flex; align-items:center; gap:.45rem;
        background:rgba(59,130,246,.12); border:1px solid rgba(59,130,246,.28);
        border-radius:100px; padding:.28rem .85rem;
        font-size:.65rem; font-weight:600; letter-spacing:2.5px;
        text-transform:uppercase; color:{ACCENT}; margin-bottom:1rem;
    }}
    .hero-badge-dot {{
        width:6px; height:6px; border-radius:50%;
        background:{GREEN}; box-shadow:0 0 8px {GREEN};
        animation: pulse-dot 2s ease-in-out infinite;
    }}
    @keyframes pulse-dot {{
        0%,100% {{ opacity:1; transform:scale(1); }}
        50% {{ opacity:.6; transform:scale(.8); }}
    }}
    .main-title {{
        font-family:'Space Grotesk', sans-serif;
        font-size: clamp(2rem, 3.5vw, 3.2rem);
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, {ACCENT} 40%, {GREEN} 70%, {ACCENT3} 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        letter-spacing: -1px;
        line-height: 1.1;
        margin-bottom: .5rem;
    }}
    .hero-subtitle {{
        color: {MUTED};
        font-size: .82rem;
        font-weight: 400;
        letter-spacing: .5px;
        line-height: 1.6;
        max-width: 580px;
        margin-top: .4rem;
    }}
    .hero-pills {{
        display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1.2rem;
    }}
    .hero-pill {{
        background: rgba(255,255,255,.04); border: 1px solid {BORDER2};
        border-radius: 100px; padding: .25rem .75rem;
        font-size: .65rem; font-weight: 500; color: {TEXT2};
        letter-spacing: 1px; text-transform: uppercase;
    }}

    /* ── Section Headers ───────────────────────────────── */
    .section-header {{
        font-family:'Space Grotesk', sans-serif;
        font-size: .85rem; font-weight: 700;
        letter-spacing: 2px; color: {TEXT2};
        text-transform: uppercase;
        display: flex; align-items: center; gap: .6rem;
        margin: 2rem 0 1rem 0;
        padding-bottom: .6rem;
        border-bottom: 1px solid {BORDER};
    }}
    .section-header::before {{
        content:''; display:block; width:3px; height:1rem; border-radius:3px;
        background: linear-gradient(180deg, {ACCENT}, {GREEN}); flex-shrink:0;
    }}

    /* ── KPI Strip ─────────────────────────────────────── */
    .kpi-strip {{
        display: grid; grid-template-columns: repeat(8, 1fr); gap: .75rem;
        margin: .5rem 0 1.5rem 0;
    }}
    .kpi-card {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 1rem .9rem .85rem .9rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: border-color .2s, box-shadow .2s, transform .15s;
    }}
    .kpi-card:hover {{
        border-color: {BORDER2};
        box-shadow: 0 4px 24px rgba(0,0,0,.4);
        transform: translateY(-1px);
    }}
    .kpi-card::after {{
        content:''; position:absolute; top:0; left:0; right:0; height:2px;
        border-radius:14px 14px 0 0;
        background: var(--kpi-accent);
    }}
    .kpi-val {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem; font-weight: 700;
        line-height: 1.15; color: var(--kpi-accent);
        letter-spacing: -0.5px;
    }}
    .kpi-lbl {{
        color: {MUTED}; font-size: .6rem;
        letter-spacing: 1.5px; text-transform: uppercase;
        margin-top: .35rem; font-weight: 500;
    }}

    /* ── Insight Cards ─────────────────────────────────── */
    .insight-card {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 1.5rem 1.4rem;
        margin: .5rem 0;
        transition: box-shadow .2s, border-color .2s, transform .15s;
    }}
    .insight-card:hover {{
        box-shadow: 0 8px 32px rgba(59,130,246,.12);
        border-color: rgba(59,130,246,.35);
        transform: translateY(-2px);
    }}
    .insight-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: .7rem; font-weight: 600;
        color: {ACCENT}; text-transform: uppercase; letter-spacing: 2px;
        margin-bottom: .6rem;
    }}
    .insight-value {{
        font-size: 2rem; font-weight: 700;
        color: {TEXT}; font-family: 'Space Grotesk', sans-serif;
        letter-spacing: -1px; line-height: 1;
    }}
    .insight-sub {{ font-size: .72rem; color: {MUTED}; margin-top: .5rem; line-height: 1.5; }}

    /* ── Buttons ───────────────────────────────────────── */
    .stButton > button {{
        background: linear-gradient(135deg, {ACCENT} 0%, #1d4ed8 100%) !important;
        color: #fff !important; border: none !important; border-radius: 10px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: .8rem !important; font-weight: 700 !important;
        letter-spacing: 2px !important; text-transform: uppercase !important;
        padding: .75rem 1.8rem !important; width: 100% !important;
        cursor: pointer !important; transition: all .2s ease !important;
        box-shadow: 0 2px 20px rgba(59,130,246,.3) !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 32px rgba(59,130,246,.5) !important;
        background: linear-gradient(135deg, #60a5fa 0%, #2563eb 100%) !important;
    }}

    /* ── Metric Components ─────────────────────────────── */
    div[data-testid="metric-container"] {{
        background: {CARD_BG} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 14px !important;
        padding: 1.2rem 1.3rem !important;
        border-left: 3px solid {ACCENT} !important;
        box-shadow: 0 2px 16px rgba(0,0,0,.25) !important;
        transition: box-shadow .2s !important;
    }}
    div[data-testid="metric-container"]:hover {{
        box-shadow: 0 6px 28px rgba(59,130,246,.15) !important;
    }}
    div[data-testid="metric-container"] label {{
        font-size: .65rem !important; letter-spacing: 2px !important;
        text-transform: uppercase !important; color: {MUTED} !important;
        font-weight: 600 !important;
    }}
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {{
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.8rem !important; color: {TEXT} !important;
        font-weight: 700 !important; letter-spacing: -1px !important;
    }}
    div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {{
        font-size: .72rem !important; font-weight: 600 !important;
    }}

    /* ── Sidebar ───────────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background: {CARD_BG} !important;
        border-right: 1px solid {BORDER} !important;
        padding-top: 0 !important;
    }}
    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 0 !important;
    }}
    .sidebar-logo-area {{
        background: linear-gradient(135deg, #0f172a, #111827);
        border-bottom: 1px solid {BORDER};
        padding: 1.6rem 1.2rem 1.2rem 1.2rem;
        margin: -1px -1px 0 -1px;
    }}
    .sidebar-logo-name {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.25rem; font-weight: 800;
        color: {TEXT}; letter-spacing: -0.5px;
    }}
    .sidebar-logo-name span {{ color: {ACCENT}; }}
    .sidebar-logo-sub {{
        font-size: .62rem; color: {MUTED};
        letter-spacing: 2px; text-transform: uppercase;
        margin-top: .2rem; font-weight: 500;
    }}
    .sidebar-section-label {{
        font-size: .6rem; color: {MUTED};
        letter-spacing: 2.5px; text-transform: uppercase;
        margin: 1.2rem 0 .5rem .2rem; font-weight: 600;
        display: flex; align-items: center; gap: .4rem;
    }}
    .sidebar-section-label::before {{
        content:''; display:inline-block; width:14px; height:1px;
        background: {BORDER2}; flex-shrink:0;
    }}

    /* ── Form Elements ─────────────────────────────────── */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {{
        background: {CARD_BG2} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 10px !important;
        color: {TEXT} !important;
    }}
    .stSelectbox > div > div:focus-within,
    .stMultiSelect > div > div:focus-within {{
        border-color: {ACCENT} !important;
        box-shadow: 0 0 0 2px rgba(59,130,246,.15) !important;
    }}
    .stSlider > div {{ margin-top: .2rem; }}
    div[data-baseweb="slider"] div[data-testid="stThumbValue"] {{
        background: {ACCENT} !important; border-radius: 6px !important;
    }}

    /* ── DataFrames ────────────────────────────────────── */
    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER} !important;
        border-radius: 12px !important; overflow: hidden !important;
        box-shadow: 0 2px 16px rgba(0,0,0,.2) !important;
    }}

    /* ── Tabs (for any st.tabs) ────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {{
        background: {CARD_BG} !important;
        border-bottom: 1px solid {BORDER} !important;
        border-radius: 12px 12px 0 0 !important;
        gap: 0 !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent !important;
        border-radius: 0 !important;
        color: {MUTED} !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important; font-size: .78rem !important;
        padding: .75rem 1.2rem !important;
        border-bottom: 2px solid transparent !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: {ACCENT} !important;
        border-bottom-color: {ACCENT} !important;
    }}

    /* ── Alerts & Info ─────────────────────────────────── */
    .stAlert {{
        border-radius: 12px !important;
        border-left: 3px solid {ACCENT} !important;
        background: rgba(59,130,246,.06) !important;
    }}

    /* ── Divider ───────────────────────────────────────── */
    .styled-divider {{
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, {BORDER2} 20%, {BORDER2} 80%, transparent 100%);
        margin: 1.5rem 0;
        border: none;
    }}

    /* ── Scrollbars ────────────────────────────────────── */
    ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
    ::-webkit-scrollbar-track {{ background: {DARK_BG}; }}
    ::-webkit-scrollbar-thumb {{ background: {BORDER2}; border-radius: 4px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {MUTED}; }}

    /* ── Radio nav in sidebar ──────────────────────────── */
    [data-testid="stSidebar"] [data-testid="stRadio"] label {{
        border-radius: 8px !important;
        padding: .45rem .75rem !important;
        font-size: .78rem !important;
        font-weight: 500 !important;
        color: {TEXT2} !important;
        transition: background .15s, color .15s !important;
        cursor: pointer !important;
    }}
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
        background: rgba(59,130,246,.08) !important;
        color: {TEXT} !important;
    }}
    [data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] + label,
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] {{
        background: rgba(59,130,246,.12) !important;
        color: {ACCENT} !important;
    }}

    /* ── Download buttons ──────────────────────────────── */
    .stDownloadButton > button {{
        background: rgba(59,130,246,.1) !important;
        color: {ACCENT} !important;
        border: 1px solid rgba(59,130,246,.3) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: .75rem !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        padding: .5rem 1.2rem !important;
        width: auto !important;
        box-shadow: none !important;
    }}
    .stDownloadButton > button:hover {{
        background: rgba(59,130,246,.2) !important;
        border-color: {ACCENT} !important;
        transform: none !important;
        box-shadow: 0 2px 12px rgba(59,130,246,.2) !important;
    }}

    /* ── Spinner ───────────────────────────────────────── */
    .stSpinner > div {{
        border-top-color: {ACCENT} !important;
    }}

    /* ── Success/warning messages ──────────────────────── */
    .element-container .stAlert [data-testid="stMarkdownContainer"] p {{
        font-size: .82rem !important; font-weight: 500 !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    # ── Hero ─────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="hero-wrapper">
      <div class="hero-badge">
        <span class="hero-badge-dot"></span>
        Live Analytics Platform
      </div>
      <div class="main-title">STRAIVE Dynamic Pricing</div>
      <div class="hero-subtitle">
        Enterprise-grade demand modeling, price elasticity analysis, and revenue simulation
        built for data-driven pricing decisions across all service lines.
      </div>
      <div class="hero-pills">
        <span class="hero-pill">⚡ Price Elasticity</span>
        <span class="hero-pill">📈 Revenue Simulation</span>
        <span class="hero-pill">🎯 Optimal Pricing</span>
        <span class="hero-pill">🌍 Regional Intelligence</span>
        <span class="hero-pill">⚔️ Competitive Positioning</span>
        <span class="hero-pill">⚠️ Risk Analysis</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"""
        <div class="sidebar-logo-area">
          <div style="display:flex;align-items:center;gap:.7rem;margin-bottom:.5rem;">
            <div style="width:36px;height:36px;border-radius:10px;
                       background:linear-gradient(135deg,{ACCENT},{GREEN});
                       display:flex;align-items:center;justify-content:center;
                       font-size:1.1rem;flex-shrink:0;">💹</div>
            <div>
              <div class="sidebar-logo-name"><span>STRAIVE</span></div>
              <div class="sidebar-logo-sub">Pricing Platform</div>
            </div>
          </div>
          <div style="display:flex;gap:.5rem;margin-top:.8rem;">
            <div style="background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.25);
                       border-radius:6px;padding:.2rem .55rem;font-size:.6rem;
                       color:{GREEN};font-weight:600;letter-spacing:1px;">● LIVE</div>
            <div style="background:rgba(255,255,255,.04);border:1px solid {BORDER};
                       border-radius:6px;padding:.2rem .55rem;font-size:.6rem;
                       color:{MUTED};font-weight:500;">v2.0</div>
          </div>
        </div>
        <div style="height:1px;background:{BORDER};margin:0;"></div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="sidebar-section-label">Model Control</div>', unsafe_allow_html=True)
        train_btn    = st.button("▶  BUILD MODEL", type="primary")
        n_records    = st.slider("Transaction records", 1000, 8000, 3600, 200, key="n_rec")
        optimize_for = st.selectbox("Optimise for", ["revenue", "profit", "margin"], key="opt_obj")

        st.markdown(f"""
        <div style="height:1px;background:{BORDER};margin:1.2rem 0 0 0;"></div>
        <div class="sidebar-section-label">Analytics Modules</div>
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
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                   min-height:52vh;text-align:center;padding:3rem 1rem;">
          <div style="width:80px;height:80px;border-radius:20px;
                     background:linear-gradient(135deg,rgba(59,130,246,.15),rgba(34,197,94,.1));
                     border:1px solid rgba(59,130,246,.25);
                     display:flex;align-items:center;justify-content:center;
                     font-size:2.4rem;margin-bottom:1.8rem;
                     box-shadow:0 8px 40px rgba(59,130,246,.1);">💹</div>
          <div style="font-family:'Space Grotesk',sans-serif;font-size:1.9rem;font-weight:800;
                     color:{TEXT};letter-spacing:-1px;margin-bottom:.8rem;line-height:1.1;">
            Ready to Optimise
          </div>
          <div style="color:{MUTED};font-size:.9rem;line-height:1.75;max-width:460px;margin-bottom:2rem;">
            Click <strong style="color:{ACCENT};font-weight:700;">▶ BUILD MODEL</strong> in the sidebar to generate
            transaction data, fit demand elasticity models, and unlock all 14 analytics modules.
          </div>
          <div style="display:flex;flex-wrap:wrap;justify-content:center;gap:.6rem;max-width:500px;">
            {"".join(f'<span style="background:rgba(255,255,255,.04);border:1px solid {BORDER};border-radius:8px;padding:.35rem .8rem;font-size:.68rem;color:{TEXT2};font-weight:500;">{m}</span>'
              for m in ["Elasticity Modeling","Optimal Pricing","Revenue Simulation",
                        "What-If Scenarios","Competitive Intel","Risk Analysis"])}
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
        ("Avg Price",      f"${avg_price:,.0f}",         TEAL),
        ("Avg Discount",   f"{avg_disc:.1f}%",           YELLOW),
        ("Win Rate",       f"{win_rate:.1f}%",           GREEN),
        ("Products",       str(n_products),               PURPLE),
        ("Segments",       str(n_segments),               ACCENT2),
    ]
    for col, (lbl, val, clr) in zip(_mc, _kpis):
        col.markdown(f"""
        <div class="kpi-card" style="--kpi-accent:{clr};">
          <div class="kpi-val">{val}</div>
          <div class="kpi-lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

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
            renamed = e_df[display_cols].rename(columns={
                "elasticity": "Price Elasticity", "disc_lift": "Discount Lift",
                "r_squared": "R²", "mean_price": "Avg Price ($)", "mean_volume": "Avg Volume",
                "mean_margin": "Avg Margin (%)", "n_obs": "Observations"
            })
            st.dataframe(
                style_df(renamed, bar_cols=["Price Elasticity"], pos_color=ACCENT, mid_align=True),
                use_container_width=True, height=280,
            )
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

        st.dataframe(
            style_df(comp_df, bar_cols=["Gap %"], mid_align=True,
                     pos_color=GREEN, neg_color=RED),
            use_container_width=True,
        )

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

        st.dataframe(
            style_df(reg_df, bar_cols=["Rec. Price ($)", "Demand Index"], pos_color=ACCENT3),
            use_container_width=True,
        )

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

        st.dataframe(
            style_df(seg_stats, bar_cols=["Revenue", "AvgMargin", "WinRate"], pos_color=GREEN),
            use_container_width=True,
        )

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

        pf_display = (
            prod_pf[["product", "Segment", "Revenue", "Margin", "Volume", "Rev_Share", "Quadrant"]]
            .rename(columns={"product": "Product", "Rev_Share": "Rev Share %"})
            .sort_values("Revenue", ascending=False)
        )
        st.dataframe(
            style_df(pf_display, bar_cols=["Revenue", "Margin", "Rev Share %"], pos_color=YELLOW),
            use_container_width=True, height=400,
        )

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
    st.markdown(f"""
    <div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:16px;
               padding:1.4rem 2rem;display:flex;align-items:center;justify-content:space-between;
               flex-wrap:wrap;gap:1rem;margin-top:1rem;">
      <div style="display:flex;align-items:center;gap:.9rem;">
        <div style="width:34px;height:34px;border-radius:9px;flex-shrink:0;
                   background:linear-gradient(135deg,{ACCENT},{GREEN});
                   display:flex;align-items:center;justify-content:center;font-size:1rem;">💹</div>
        <div>
          <div style="font-family:'Space Grotesk',sans-serif;font-size:.88rem;font-weight:700;
                     color:{TEXT};letter-spacing:-0.3px;">STRAIVE Dynamic Pricing Platform</div>
          <div style="color:{MUTED};font-size:.62rem;letter-spacing:1.5px;
                     text-transform:uppercase;margin-top:.1rem;">
            Demand Elasticity · Optimal Pricing · Competitive Intelligence
          </div>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:2rem;">
        <div style="text-align:center;">
          <div style="color:{MUTED};font-size:.58rem;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:.2rem;">Platform</div>
          <div style="font-family:'Space Grotesk',sans-serif;font-size:.82rem;font-weight:700;color:{ACCENT3};">STRAIVE Analytics</div>
        </div>
        <div style="width:1px;height:28px;background:{BORDER};"></div>
        <div style="text-align:right;">
          <div style="color:{MUTED};font-size:.62rem;">Built: {CURRENT_DATE.strftime('%Y-%m-%d')}</div>
          <div style="color:{MUTED};font-size:.58rem;margin-top:.15rem;font-family:'JetBrains Mono',monospace;">
            straive_artifacts/pricing_params.json
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
