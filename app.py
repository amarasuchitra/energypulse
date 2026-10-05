"""
EnergyPulse — Home Electricity Dashboard
==================================================
Smart home energy monitoring, predictions, cost tracking,
appliance breakdown, optimization tips, and usage trends.

Launch:
    streamlit run app.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import copy
import calendar as cal
import io
import json
import pickle
import re
import hashlib
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from replay import ReplaySimulator
from appliances import get_appliance_breakdown
from cost import daily_cost, weekly_cost, next_month_cost
from optimize import detect_anomalies, generate_anomaly_tips, calendar_tips, whatif_simulator
from model import (
    load_data, predict_next_period,
    FEATURE_COLS, TARGET, WINDOW_SIZE,
)
from data import (
    CLEAN_CSV, remap_to_current_dates, shift_forecast_to_current_dates,
    SAMPLE_META_PATH,
)
from data_source import (
    MAX_UPLOAD_BYTES, POWER_UNITS, DataSource, normalize_uploaded,
    source_from_sample, source_from_upload,
)
from i18n import T, TLIST, LANGS, HOME_TYPES, home_type_label, localized_appliance_category, localized_appliance_label, localized_appliance_type, format_localized_month, format_localized_date
from db import get_db
from features_ui import render_family_tab, render_notifications_tab, render_chat_tab, ensure_login
from appliance_ui import (
    render_appliance_tab, render_analysis_tab, render_save_tab, render_meter_settings, report_body,
)
from meter_source import owned_appliances, SOURCE_NOTE, detected_history, today_str
from appliance_ui import meter_settings
from auth import Accounts, PASSWORD_RULES, new_guest_email, is_guest_email, session_expired
from motion import notify, runtime as motion_runtime, skeleton
from integrity import Ledger, store_bill_file, read_bill_file, bill_path, sign_report, verify_report
from home_ui import render_home_tab, _members as household_people
from daily_brief import build_brief, send_daily_brief
from meter_source import last_days
from tariff import Tariff
from notifications import get_notification_service
from devices import household_devices
from sessions import extract_sessions, flag_long_runs
import toolkit_ui
from toolkit import send_left_on
from shell_ui import tc, apply_theme, theme_toggle, current_theme, inject_skin, page_header, sidebar_nav, login_hero, step_list, wordmark
from clock import local_now, browser_tz, options as tz_options, set_tz, user_tz


def init_language():
    """Ensure the selected language exists in session state (persists across tabs)."""
    if "lang" not in st.session_state:
        st.session_state.lang = "en"


def render_language_selector(key="lang_select", centered=False):
    """Language dropdown bound to st.session_state['lang'] so it survives reruns."""
    init_language()
    codes = list(LANGS.keys())
    current = st.session_state.lang if st.session_state.lang in codes else "en"
    names = [LANGS[c] for c in codes]
    container = st.container()
    if centered:
        c1, c2, c3 = container.columns([1, 1.4, 1])
        container = c2
    with container:
        chosen_name = st.selectbox(
            T("lang_label"),
            names,
            index=codes.index(current),
            key=key,
        )
    chosen = codes[names.index(chosen_name)] if chosen_name in names else current
    if chosen != st.session_state.lang:
        st.session_state.lang = chosen
        st.rerun()

# ── Paths ───────────────────────────────────────────────────
DATA_DIR = "data"
MODEL_DIR = "models"
XGB_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")
LSTM_PATH = os.path.join(MODEL_DIR, "lstm_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "lstm_scaler.pkl")
META_PATH = os.path.join(MODEL_DIR, "model_meta.pkl")

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="EnergyPulse",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── City tariff defaults ───────────────────────────────────────
CITY_TARIFF_DEFAULTS = {
    "Mumbai": 9.5, "Delhi": 8.5, "Bangalore": 6.5,
    "Chennai": 7.0, "Kolkata": 7.5, "Hyderabad": 6.0,
    "Pune": 8.0, "Ahmedabad": 5.5,
}

APPLIANCE_PRESETS = [
    ("Air Conditioner", ""), ("Refrigerator", ""), ("Washing Machine", ""), ("Water Heater", ""),
    ("Water Pump", ""), ("Television", ""), ("Microwave", ""), ("Lights & Fans", ""), ("Other", ""),
]

ACCENT = "#F5A83C"
ACCENT2 = "#6FA8DC"
LIGHT = "#6F7994"

st.markdown("""
<style>
    .stApp, section[data-testid="stSidebar"] {
        font-family: 'Hanken Grotesk', 'Segoe UI', system-ui, sans-serif !important;
    }
    .stDeployButton { display: none; }
    .section-header {
        display: flex; align-items: center; gap: 12px;
        margin: 1.4rem 0 0.6rem 0;
    }
    .section-header h2 {
        color: var(--plaster); font-size: 1.05rem; font-weight: 700;
        margin: 0; white-space: nowrap; display: flex; align-items: center; gap: 8px;
    }
    .section-line {
        flex: 1; height: 1px;
        background: linear-gradient(90deg, rgba(245,168,60,0.25), transparent 80%);
    }
    .metric-card {
        background: var(--slate); border: 1px solid var(--line);
        border-radius: 16px; padding: 1.2rem 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); position: relative;
        min-height: 120px; box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03);
    }
    .metric-card::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    }
    .mc-green::before { background: linear-gradient(90deg, var(--current), #6FA8DC); }
    .mc-teal::before { background: linear-gradient(90deg, #6FA8DC, var(--current)); }
    .mc-orange::before { background: linear-gradient(90deg, #F08A5D, #F7C27A); }
    .mc-purple::before { background: linear-gradient(90deg, #B39DDB, #CDBDEB); }
    .mc-rose::before { background: linear-gradient(90deg, #F28FB1, #F6B3C9); }
    .metric-card:hover {
        border-color: rgba(245,168,60,0.2); transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.07), 0 2px 6px rgba(0,0,0,0.04);
    }
    .mc-label {
        color: var(--mist); font-size: 0.68rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.7px;
        margin-bottom: 0.4rem;
    }
    .mc-value {
        color: var(--plaster); font-size: 1.65rem; font-weight: 800;
        line-height: 1.15; white-space: normal; display: flex; align-items: baseline; gap: 6px;
    }
    .mc-unit { font-size: 0.85rem; font-weight: 500; color: var(--faint); }
    .mc-trend {
        display: inline-flex; align-items: center; gap: 3px;
        font-size: 0.7rem; font-weight: 600; padding: 2px 8px;
        border-radius: 6px; margin-top: 0.3rem;
    }
    .mc-trend.up { background: #1F3A33; color: var(--ok); }
    .mc-trend.down { background: #3F2526; color: var(--alert); }
    .mc-trend.neutral { background: var(--slate-2); color: var(--current); }
    .mc-sub {
        color: var(--faint); font-size: 0.68rem; margin-top: 0.3rem;
        white-space: normal;
    }
    [data-testid="stMetricValue"] {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        font-size: 1.4rem !important;
    }
    [data-testid="stMetricDelta"] {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        font-size: 0.85rem !important;
    }
    [data-testid="stMetricLabel"] {
        white-space: normal !important;
    }
    .live-bar {
        display: flex; align-items: center; gap: 10px;
        background: var(--slate); border: 1px solid var(--line);
        border-radius: 14px; padding: 0.65rem 1.3rem; margin-bottom: 1rem;
        font-size: 0.82rem; color: var(--mist);
        box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03);
    }
    .live-dot {
        width: 8px; height: 8px; background: #22C55E; border-radius: 50%;
        animation: gp 2s ease-in-out infinite; flex-shrink: 0;
        box-shadow: 0 0 6px rgba(34,197,94,0.4);
    }
    @keyframes gp {
        0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(34,197,94,0.4); }
        50% { opacity: 0.6; box-shadow: 0 0 0 5px rgba(34,197,94,0); }
    }
    .live-label { color: #22C55E; font-weight: 700; font-size: 0.78rem; letter-spacing: 0.3px; }
    .disc {
        background: var(--slate); border: 1px solid var(--line);
        border-left: 3px solid var(--current); border-radius: 0 14px 14px 0;
        padding: 1.1rem 1.5rem; margin-top: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03);
    }
    .disc p { color: var(--mist); font-size: 0.8rem; margin: 0; line-height: 1.65; }
    .disc strong { color: var(--current); }
    .tip-box {
        background: var(--slate); border: 1px solid var(--line);
        border-radius: 14px; padding: 1rem 1.3rem; margin-bottom: 0.7rem;
        font-size: 0.85rem; color: var(--mist); line-height: 1.6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03);
        transition: all 0.2s ease;
    }
    .tip-box:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.06), 0 1px 4px rgba(0,0,0,0.04); }
    .tip-box strong { color: var(--current); }
    .cal-grid { display: grid; grid-template-columns: repeat(7, 1fr); gap: 4px; }
    .cal-header {
        text-align: center; font-size: 0.7rem; font-weight: 700;
        color: var(--mist); padding: 4px 0; text-transform: uppercase;
    }
    .cal-day {
        text-align: center; border-radius: 8px; padding: 8px 4px;
        font-size: 0.75rem; font-weight: 600; cursor: pointer;
        transition: all 0.2s; border: 1px solid transparent;
    }
    .cal-day:hover { border-color: rgba(245,168,60,0.3); transform: scale(1.05); }
    .cal-day.low { background: rgba(245,168,60,0.1); color: var(--current); }
    .cal-day.medium { background: rgba(196,148,74,0.12); color: #F08A5D; }
    .cal-day.high { background: rgba(180,80,80,0.1); color: var(--alert); }
    .cal-day.empty { background: transparent; cursor: default; }
    .cal-day.future { background: rgba(245,168,60,0.06); color: #6FA8DC; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--slate) 0%, var(--ink) 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.06) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown h4 { color: #FFFFFF !important; }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span { color: rgba(255,255,255,0.75) !important; }
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] small { color: rgba(255,255,255,0.45) !important; }
    .sb-section-label {
        color: rgba(255,255,255,0.5); font-size: 0.62rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 1.2px; margin: 0.9rem 0 0.35rem 0;
        display: flex; align-items: center; gap: 6px;
    }
    .sb-card {
        background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px; padding: 0.65rem 0.8rem; margin-bottom: 0.4rem;
        transition: background 0.2s;
    }
    .sb-card:hover { background: rgba(255,255,255,0.1); }
    .sb-card-title {
        color: rgba(255,255,255,0.95); font-size: 0.82rem; font-weight: 600;
        display: flex; align-items: center; justify-content: space-between;
    }
    .sb-card-sub {
        color: rgba(255,255,255,0.55); font-size: 0.7rem; margin-top: 2px;
    }
    .sb-appliance-tag {
        display: inline-flex; align-items: center; gap: 4px;
        background: rgba(245,168,60,0.2); border: 1px solid rgba(245,168,60,0.3);
        color: rgba(255,255,255,0.85); font-size: 0.68rem; font-weight: 500;
        padding: 3px 8px; border-radius: 6px; margin: 2px;
    }
    .login-container {
        max-width: 420px; margin: 0 auto; padding: 3rem 2.5rem;
        background: var(--slate); border: 1px solid var(--line); border-radius: 20px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.06), 0 1px 4px rgba(0,0,0,0.04);
    }
    .login-title {
        font-size: 1.6rem; font-weight: 800; color: var(--plaster);
        text-align: center; margin-bottom: 0.3rem;
    }
    .login-subtitle {
        font-size: 0.85rem; color: var(--mist); text-align: center;
        margin-bottom: 1.8rem; line-height: 1.5;
    }
    .login-divider {
        display: flex; align-items: center; gap: 12px;
        margin: 1.2rem 0; color: var(--faint); font-size: 0.75rem;
    }
    .login-divider::before, .login-divider::after {
        content: ''; flex: 1; height: 1px; background: var(--line);
    }
    .onboard-container { max-width: 680px; margin: 0 auto; }
    .data-progress {
        height: 4px; background: rgba(245,168,60,0.1); border-radius: 4px;
        overflow: hidden; margin-top: 6px;
    }
    .data-progress-fill {
        height: 100%; background: linear-gradient(90deg, var(--current), #6FA8DC);
        border-radius: 4px; transition: width 0.5s ease;
    }
    [data-testid="stHorizontalBlock"] { gap: 0.8rem !important; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; background: var(--slate); border-radius: 12px;
        padding: 4px; border: 1px solid var(--line);
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px; padding: 10px 20px; font-weight: 600;
        font-size: 0.85rem; color: var(--mist); border: none; background: transparent;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover { color: var(--current); }
    .stTabs [aria-selected="true"] {
        background: var(--current) !important; color: #FFFFFF !important; border-radius: 8px;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }
    .stButton > button {
        border-radius: 10px !important; font-weight: 600 !important;
        font-size: 0.82rem !important; transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .error-card {
        background: var(--slate); border: 1px solid var(--line);
        border-left: 4px solid #F08A5D; border-radius: 0 16px 16px 0;
        padding: 2rem 2.5rem; max-width: 520px; margin: 3rem auto;
        box-shadow: 0 4px 24px rgba(0,0,0,0.06);
        text-align: center;
    }
    .error-card-icon { font-size: 2.2rem; margin-bottom: 0.8rem; }
    .error-card-title { font-size: 1.1rem; font-weight: 700; color: var(--plaster); margin-bottom: 0.4rem; }
    .error-card-msg { font-size: 0.85rem; color: var(--mist); line-height: 1.6; margin-bottom: 1.2rem; }
    .skeleton {
        background: linear-gradient(90deg, var(--line) 25%, var(--slate) 50%, var(--line) 75%);
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
        border-radius: 10px;
    }
    .skeleton-chart { height: 340px; width: 100%; }
    .skeleton-row { height: 16px; width: 80%; margin-bottom: 8px; }
    .skeleton-card { height: 120px; width: 100%; }
    @keyframes shimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    .empty-state {
        text-align: center; padding: 3rem 2rem;
        color: var(--mist); font-size: 0.9rem;
    }
    .empty-state-icon { font-size: 2.5rem; margin-bottom: 0.8rem; opacity: 0.6; }
    .empty-state-title { font-size: 1rem; font-weight: 700; color: var(--plaster); margin-bottom: 0.3rem; }
    .hero {
        position: relative; padding: 1.8rem 2.5rem; margin-bottom: 1rem;
        border-radius: 20px; overflow: hidden;
        background: linear-gradient(135deg,var(--slate) 0%,var(--slate-2) 50%,var(--current) 100%);
        border: 1px solid rgba(245,168,60,0.2);
    }
    .hero-glow {
        position: absolute; top: 0; right: 0; width: 300px; height: 300px;
        background: radial-gradient(circle,rgba(107,143,94,0.15) 0%,transparent 70%);
        transform: translate(30%,-30%);
    }
    .hero-badge {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.25);
        color: #FFFFFF; padding: 5px 14px; border-radius: 20px;
        font-size: 0.7rem; font-weight: 700; letter-spacing: 1px;
        text-transform: uppercase; margin-bottom: 0.8rem;
    }
    .hero-title {
        color: #FFFFFF; font-size: 1.9rem; font-weight: 800;
        margin: 0 0 0.25rem 0; letter-spacing: -0.8px;
    }
    .hero-title span {
        background: linear-gradient(135deg,#FBE0B5,#F7C27A);
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-sub { color: rgba(255,255,255,0.7); font-size: 0.88rem; margin: 0; }
    .bill-card {
        background: var(--slate); border: 1px solid var(--line); border-radius: 14px;
        padding: 1rem 1.1rem; margin-bottom: 0.8rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03);
    }
    /* ── Responsive tweaks: laptop → tablet → mobile ── */
    @media (max-width: 1024px) {
        .hero { padding: 1.4rem 1.6rem; }
        .hero-title { font-size: 1.55rem; }
    }
    @media (max-width: 820px) {
        .metric-card { min-height: auto; padding: 1rem 0.85rem; }
        .mc-value { font-size: 1.35rem; flex-wrap: wrap; }
        .section-header h2 { font-size: 0.95rem; white-space: normal; }
        .live-bar { flex-wrap: wrap; padding: 0.6rem 0.9rem; row-gap: 6px; }
        .live-bar .data-progress { margin-left: auto; }
        .disc { padding: 1rem 1.1rem; }
    }
    @media (max-width: 640px) {
        [data-testid="stMainBlockContainer"] { padding-left: 0.9rem; padding-right: 0.9rem; }
        .hero { padding: 1.15rem 1.05rem; border-radius: 16px; }
        .hero-title { font-size: 1.3rem; letter-spacing: -0.4px; }
        .hero-sub { font-size: 0.78rem; }
        .metric-card { padding: 0.85rem 0.75rem; border-radius: 13px; }
        .mc-label { font-size: 0.62rem; }
        .mc-value { font-size: 1.22rem; }
        .stTabs [data-baseweb="tab-list"] {
            overflow-x: auto; -webkit-overflow-scrolling: touch; scrollbar-width: none;
        }
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
        .stTabs [data-baseweb="tab"] { padding: 10px 14px; white-space: nowrap; }
        .cal-day { padding: 7px 2px; font-size: 0.68rem; }
        .cal-header { font-size: 0.62rem; }
        .stButton > button { min-height: 44px; }
        section[data-testid="stSidebar"] .stButton > button { min-height: 40px; }
    }
</style>
""", unsafe_allow_html=True)

def PLOTLY_LAYOUT(**overrides):
    dark = current_theme() == "dark"
    layout = dict(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Hanken Grotesk, Segoe UI, sans-serif", color=tc("mist")),
        margin=dict(l=40, r=20, t=40, b=40),
        transition=dict(duration=450, easing="cubic-in-out"),
        xaxis=dict(gridcolor=tc("grid"), zerolinecolor=tc("line")),
        yaxis=dict(gridcolor=tc("grid"), zerolinecolor=tc("line")),
    )
    for k, v in overrides.items():
        if k in ("xaxis", "yaxis") and isinstance(v, dict):
            layout[k] = {**layout[k], **v}
        else:
            layout[k] = v
    return layout


def section(title, icon=""):
    icon_html = ""   # headings are plain text; `icon` is kept so existing calls still work
    st.markdown(f"""
    <div class="section-header">
        <h2>{icon_html} {title}</h2>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)


def metric_card(col, label, value, unit="", sub="", trend_text="", trend_dir="neutral",
                css_class="mc-green"):
    parts = [
        f'<div class="metric-card {css_class}">',
        f'<div class="mc-label">{label}</div>',
        f'<div class="mc-value">{value}<span class="mc-unit"> {unit}</span></div>',
    ]
    if trend_text:
        parts.append(f'<div class="mc-trend {trend_dir}">{trend_text}</div>')
    if sub:
        parts.append(f'<div class="mc-sub">{sub}</div>')
    parts.append('</div>')
    html = "".join(parts)
    with col:
        st.markdown(html, unsafe_allow_html=True)


def fmt_rs(val):
    return f"{val:,.2f}"


def fmt_kwh(val):
    return f"{val:.2f}"


@st.cache_data
def load_data_cached():
    return load_data()

@st.cache_data
def load_sample_meta():
    """Description of the bundled dataset, written by data.py."""
    if os.path.exists(SAMPLE_META_PATH):
        try:
            with open(SAMPLE_META_PATH, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {}
    return {}


@st.cache_data(show_spinner=False)
def build_sample_source(days=90):
    """
    The bundled demo dataset, with its real measurement window preserved.

    The timestamps are shifted forward so the demo looks current; the shift is
    recorded on the source so the UI can say plainly that these are sample
    recordings being replayed, not a live meter.
    """
    raw = load_data_cached()
    meta = load_sample_meta()
    original = (meta.get("original_start"), meta.get("original_end"))
    shifted = remap_to_current_dates(raw, last_n_days=days)
    return source_from_sample(shifted, filename=os.path.basename(CLEAN_CSV),
                              original_range=original, shifted=True)


def get_data_source():
    """The active dataset: the user's CSV when uploaded, otherwise the sample."""
    upload = st.session_state.get("data_upload")
    if upload is not None and upload.get("source") is not None:
        return upload["source"]
    return build_sample_source()


def active_dataframe():
    """A defensive copy of the active source, safe for the tabs to mutate."""
    return get_data_source().df.copy()


def source_summary(source):
    """Human readable provenance for the active dataset, in the active language."""
    df = source.df
    if df.empty:
        return {"kind": source.kind, "title": T("src_upload_title"),
                "body": T("src_upload_fail", reason=T("src_need_power")),
                "rows": 0, "caps": source.capabilities}
    rows = len(df)
    if source.is_sample:
        body = T("src_sample_body",
                 start=format_localized_date(df["datetime"].min()),
                 end=format_localized_date(df["datetime"].max()))
        original = source.original_range
        if original and all(original):
            body += " (" + T("src_recorded",
                             start=format_localized_date(original[0]),
                             end=format_localized_date(original[1])) + ")"
    else:
        body = T("src_upload_body",
                 name=source.label,
                 rows=f"{rows:,}",
                 start=format_localized_date(df["datetime"].min()),
                 end=format_localized_date(df["datetime"].max()))
    return {
        "kind": source.kind,
        "title": T("src_sample_title") if source.is_sample else T("src_upload_title"),
        "body": body,
        "rows": rows,
        "caps": source.capabilities,
        "original": source.original_range,
        "age_hours": source.age_hours(),
        "report": source.report,
    }


def format_data_age(hours):
    if hours is None:
        return ""
    if hours < 1:
        return T("src_age_now")
    if hours < 48:
        return T("src_age_hours", n=int(round(hours)))
    return T("src_age_days", n=int(round(hours / 24)))


def read_uploaded_csv(uploaded):
    """Read an uploaded CSV without assuming delimiter, index column or encoding."""
    payload = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded
    if isinstance(payload, str):
        payload = payload.encode("utf-8", errors="replace")
    if payload is None or len(payload) == 0:
        return None, T("src_upload_fail", reason=T("src_need_power"))
    if len(payload) > MAX_UPLOAD_BYTES:
        return None, T("src_upload_fail",
                       reason=f"{len(payload) / 1024 / 1024:.1f} MB > "
                              f"{MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB")
    last_error = None
    for kwargs in (
        {"sep": None, "engine": "python"},
        {"sep": ",", "encoding": "utf-8-sig"},
        {"sep": ";", "encoding": "utf-8-sig"},
        {"sep": "\t", "encoding": "utf-8-sig"},
        {"sep": ",", "encoding": "latin-1"},
    ):
        try:
            # pandas needs a file-like object; raw bytes raise TypeError.
            buffer = io.BytesIO(payload)
            frame = pd.read_csv(buffer, **kwargs)
        except (ValueError, TypeError, UnicodeDecodeError, pd.errors.ParserError,
                pd.errors.EmptyDataError) as exc:
            last_error = exc
            continue
        if frame is not None and not frame.empty:
            return frame, None
    return None, T("src_upload_fail", reason=T("src_unreadable"))


def handle_upload(uploaded, power_unit, signature=None):
    """Normalize an uploaded CSV and store it as the active data source."""
    if uploaded is None:
        return
    frame, error = read_uploaded_csv(uploaded)
    if frame is None:
        st.session_state.data_upload = None
        st.session_state.data_upload_sig = None
        st.session_state.data_upload_error = error
        return
    canonical, report = normalize_uploaded(frame, power_unit=power_unit)
    if not report.get("ok") or canonical.empty:
        reason = T("src_note_no_power")
        st.session_state.data_upload = None
        st.session_state.data_upload_sig = None
        st.session_state.data_upload_error = T("src_upload_fail", reason=reason)
        return
    st.session_state.data_upload = {
        "df": canonical,
        "source": source_from_upload(canonical, uploaded.name, report=report),
    }
    st.session_state.data_upload_sig = signature
    st.session_state.data_upload_error = None
    if "simulator" in st.session_state:
        st.session_state.simulator.reset()


def render_data_source_panel():
    """Settings block: upload a CSV, or fall back to the clearly labelled sample."""
    uploaded = st.file_uploader(T("src_choose_file"), type=["csv", "txt", "tsv"],
                                key="data_uploader")
    unit_options = ["auto"] + list(POWER_UNITS.keys())
    unit_choice = st.selectbox(T("src_power_unit"), unit_options, index=0,
                               key="data_power_unit")
    power_unit = None if unit_choice == "auto" else unit_choice

    # Only re-read the file when it (or the chosen unit) actually changed.
    # Re-parsing a large CSV on every widget interaction is what made the
    # dashboard crawl once a file was uploaded.
    signature = None
    if uploaded is not None:
        signature = (getattr(uploaded, "name", "upload"),
                     len(uploaded.getvalue()), power_unit)

    # An empty uploader no longer means "go back to the sample": the uploader
    # is empty every time this page is reopened. The button below does that.
    if signature is not None and st.session_state.get("data_upload_sig") != signature:
        handle_upload(uploaded, power_unit, signature)

    if st.session_state.get("data_upload_error"):
        st.error(st.session_state.data_upload_error)
    elif st.session_state.get("data_upload") is not None:
        if st.button(T("src_use_sample"), width="stretch", key="sb_back_to_sample"):
            st.session_state.data_upload = None
            st.session_state.data_upload_sig = None
            if "simulator" in st.session_state:
                st.session_state.simulator.reset()
            st.rerun()

    source = get_data_source()
    info = source_summary(source)
    is_upload = not source.is_sample
    st.markdown(f"""
    <div class="sb-card" style="margin-top:0.5rem;">
        <div class="sb-card-title">{info['title']}</div>
        <div class="sb-card-sub">{info['body']}</div>
    </div>
    """, unsafe_allow_html=True)
    if info.get("original") and all(info["original"]):
        st.markdown(
            f'<div style="color:rgba(255,255,255,0.35);font-size:0.66rem;margin-top:-0.3rem;">'
            f'{format_localized_date(info["original"][0])} – '
            f'{format_localized_date(info["original"][1])}</div>',
            unsafe_allow_html=True)
    st.markdown(
        f'<div style="color:rgba(255,255,255,0.35);font-size:0.66rem;">'
        f'{T("src_replaying")}</div>', unsafe_allow_html=True)

    report = info.get("report") or {}
    if not source.is_sample and report.get("applied_unit"):
        st.markdown(
            f'<div style="color:rgba(255,255,255,0.35);font-size:0.66rem;">'
            f'{T("src_unit_detected", unit=report["applied_unit"])}</div>',
            unsafe_allow_html=True)
    if not source.is_sample and report.get("unit_confident") is False:
        st.warning(T("src_note_unit_unsure"))
    mapped = len([v for v in report.get("columns", {}).values() if v != "datetime"])
    if is_upload and mapped:
        st.markdown(
            f'<div style="color:rgba(255,255,255,0.35);font-size:0.66rem;">'
            f'{T("src_rows_mapped", n=mapped)}</div>', unsafe_allow_html=True)
    ignored = report.get("ignored") or []
    if is_upload and ignored:
        st.markdown(
            f'<div style="color:rgba(255,255,255,0.3);font-size:0.64rem;">'
            f'{T("src_rows_ignored", cols=", ".join(map(str, ignored[:6])))}</div>',
            unsafe_allow_html=True)
    for code, params in (report.get("note_codes") or [])[:3]:
        params = dict(params)
        if "how" in params:
            params["how"] = T("src_agg_sum" if params["how"] == "sum"
                              else "src_agg_mean")
        st.markdown(
            f'<div style="color:rgba(255,255,255,0.32);font-size:0.64rem;">'
            f'{T(code, **params)}</div>', unsafe_allow_html=True)


@st.cache_resource
def load_xgb():
    with open(XGB_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_lstm():
    from tensorflow import keras
    return keras.models.load_model(LSTM_PATH)

@st.cache_resource
def load_scaler():
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_meta():
    with open(META_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_forecast():
    path = os.path.join(DATA_DIR, "next_month_forecast.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["datetime"])
    return pd.DataFrame()


def compute_scaling_factor(home_details):
    occupants = home_details.get("occupants", 4)
    appliances = home_details.get("appliances", [])
    num_appliances = len(appliances) if appliances else home_details.get("num_appliances", 3)
    baseline_occupants = 4
    baseline_appliances = 3
    scale = occupants / baseline_occupants
    scale *= 1 + (num_appliances - baseline_appliances) * 0.05
    return max(0.3, min(scale, 3.0))


def refresh_replay(sim: ReplaySimulator) -> None:
    """
    Redraw on a timer while the replay is still moving.

    streamlit-autorefresh is listed in requirements.txt and was never imported,
    so the replay position advanced in the background without the screen ever
    following it. Refresh is limited to a running, unfinished replay so an idle
    dashboard does not rerun every few seconds for nothing.
    """
    if not (sim.is_running and not sim.is_finished):
        return
    try:
        from streamlit_autorefresh import st_autorefresh
    except ImportError:
        return
    # Every refresh re-runs the whole page, so it must be slower than one full
    # run takes; a 2 s tick could restart the page before it finished drawing.
    st_autorefresh(interval=int(max(6, sim.delay)) * 1000, key="replay_autorefresh")


def get_simulator():
    """
    The replay simulator, retargeted whenever the active data source changes.

    The simulator replays whichever frame is active, so an upload replaces the
    bundled sample instead of the dashboard quietly showing sample rows.
    """
    if "simulator" not in st.session_state:
        sim = ReplaySimulator(delay=2)
        sim.start()
        st.session_state.simulator = sim
        st.session_state.simulator_source = None
    sim = st.session_state.simulator

    source = get_data_source()
    signature = (source.kind, source.label, len(source.df),
                 str(source.df["datetime"].max()) if not source.df.empty else "")
    if st.session_state.get("simulator_source") != signature:
        sim.set_data(source.df)
        st.session_state.simulator_source = signature
    return sim


@st.cache_resource
def get_accounts() -> Accounts:
    return Accounts()


def sign_out():
    """Forget everything about the signed-in account in this browser session."""
    if "simulator" in st.session_state:
        st.session_state.simulator.stop()
    for k in ["auth", "home_details", "simulator", "onboard_data", "onboard_step", "sidebar_settings",
              "page", "bills", "bill_seen", "home_state", "nilm_dismissed", "data_upload",
              "data_upload_sig", "data_upload_error"]:
        st.session_state.pop(k, None)


@st.dialog("Log out?")
def confirm_sign_out():
    name = st.session_state.get("auth", {}).get("name") or "this account"
    guest = st.session_state.get("auth", {}).get("guest")
    st.write(f"You are signed in as **{name}**. "
             + ("A guest home is not kept: logging out removes it." if guest
                else "Your home and settings are saved and will be here when you sign in again."))
    stay, leave = st.columns(2)
    if stay.button("Stay signed in", key="logout_stay", width="stretch"):
        st.rerun()
    if leave.button("Log out", key="logout_go", type="primary", width="stretch"):
        sign_out()
        notify("info", "You have logged out.")
        st.rerun()


def render_login_screen():
    if "auth" not in st.session_state:
        st.session_state.auth = {"logged_in": False, "email": None, "mode": "login"}
    auth = st.session_state.auth
    if auth.get("logged_in"):
        if session_expired(auth.get("signed_in_at")):
            sign_out()
            st.rerun()
        return True

    mode = auth.get("mode", "login")
    is_signup = (mode == "signup")
    title = T("login_create_title") if is_signup else T("login_welcome_back")
    subtitle = T("login_sub_signup") if is_signup else T("login_sub_login")

    hero_col, form_col = st.columns([1.25, 1], gap="large")
    with hero_col:
        st.markdown(login_hero(T("hero_subtitle")), unsafe_allow_html=True)
    with form_col:
      with st.container(key="login_panel"):
        st.markdown(f'<div class="ep-form-title">{title}</div>'
                    f'<div class="ep-form-sub">{subtitle}</div>', unsafe_allow_html=True)
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input(T("email_label"), placeholder=T("ph_email"), key="login_email")
            password = st.text_input(T("password_label"), type="password",
                                     placeholder=T("ph_password"), key="login_pw")
            if is_signup:
                confirm = st.text_input("Confirm password", type="password",
                                        placeholder="Type the password again", key="login_pw2")
                st.markdown(
                    "<div class='ep-pw-rules'><b>Your password must have</b><ul>"
                    + "".join(f"<li>{label}</li>" for _, label in PASSWORD_RULES)
                    + "</ul></div>", unsafe_allow_html=True)
                name = st.text_input(T("name_label"), placeholder=T("ph_name"), key="login_name")
            btn_label = T("btn_create_account") if is_signup else T("btn_sign_in")
            submitted = st.form_submit_button(btn_label, width="stretch", type="primary")
            if submitted:
                if not email or not password:
                    st.error(T("err_missing"))
                elif is_signup and password != confirm:
                    st.error("The two passwords do not match. Type the same password in both boxes.")
                else:
                    accounts = get_accounts()
                    ok, result = (accounts.sign_up(email, password, name) if is_signup
                                  else accounts.sign_in(email, password))
                    if not ok:
                        st.error(result)
                        return False
                    email = accounts.normalise(email)
                    auth.update({"logged_in": True, "email": email, "name": result,
                                 "guest": False, "signed_in_at": time.time()})
                    household = ensure_login(email, name=result)
                    saved_home = None if is_signup else get_db().get_home_details(household)
                    if saved_home:
                        st.session_state.home_details = saved_home      # straight to their own home
                    notify("success", f"Account created. Welcome, {result}." if is_signup else f"Signed in as {result}.")
                    st.rerun()

        st.markdown(f'<div class="login-divider">{T("divider_or")}</div>', unsafe_allow_html=True)
        if is_signup:
            if st.button(T("switch_to_login"), width="stretch", key="switch_login"):
                auth["mode"] = "login"
                st.rerun()
        else:
            if st.button(T("switch_to_signup"), width="stretch", key="switch_signup"):
                auth["mode"] = "signup"
                st.rerun()
        st.markdown("")
        if st.button(T("btn_guest"), width="stretch", key="guest_btn"):
            # Every guest gets a private household, so guests never see each other's data.
            guest_email = new_guest_email()
            auth.update({"logged_in": True, "email": guest_email, "name": T("guest_name"),
                         "guest": True, "signed_in_at": time.time()})
            ensure_login(guest_email, name=T("guest_name"), is_guest=True)
            notify("info", "You are browsing as a guest. Your data is private to this session.")
            st.rerun()
        render_language_selector(key="lang_select_login")

    return False


def render_onboarding():
    if "home_details" in st.session_state and st.session_state.home_details is not None:
        return True
    init_language()
    if "onboard_step" not in st.session_state:
        st.session_state.onboard_step = 1
    if "onboard_data" not in st.session_state:
        st.session_state.onboard_data = {
            "home_type": "Apartment", "occupants": 4, "appliances": [],
            "tariff_rate": 8.0, "city": "", "home_size": 0,
        }
    step = st.session_state.onboard_step
    od = st.session_state.onboard_data
    user_name = st.session_state.auth.get("name", "there")

    step_labels = [T("step_home_info"), T("step_appliances"), T("step_rate")]
    side, main_col = st.columns([1, 1.75], gap="large")
    with side:
        st.markdown(
            wordmark()
            + f"<div class='ep-head' style='margin-top:2.6rem'><h1>{T('ob_welcome', name=user_name)}</h1>"
              f"<p>{T('ob_setup_msg')}</p></div>"
            + step_list(step_labels, step), unsafe_allow_html=True)
        st.markdown("")
        render_language_selector(key="lang_select_onboard")
    with main_col:
      with st.container(key="onboard_panel"):
        if step == 1:
            _render_onboard_step1(od)
        elif step == 2:
            _render_onboard_step2(od)
        elif step == 3:
            _render_onboard_step3(od)
    return False


def _render_onboard_step1(od):
    section(T("ob_about_home"))
    c1, c2 = st.columns(2)
    with c1:
        _ht = TLIST("home_types")
        home_type_name = st.selectbox(T("home_type_label"),
            _ht,
            index=_ht.index(home_type_label(od["home_type"])) if home_type_label(od["home_type"]) in _ht else 0,
            key="ob_home_type")
        od["home_type"] = HOME_TYPES[_ht.index(home_type_name)]
    with c2:
        occupants = st.number_input(T("occupants_label"),
            min_value=1, max_value=20, value=od["occupants"], step=1, key="ob_occupants")
        od["occupants"] = occupants
    c3, c4 = st.columns(2)
    with c3:
        home_size = st.number_input(T("size_label"),
            min_value=0, max_value=10000, value=od.get("home_size", 0), step=50, key="ob_size")
        od["home_size"] = home_size
    with c4:
        city = st.text_input(T("city_label"),
            value=od.get("city", ""), key="ob_city", placeholder=T("ph_city"))
        od["city"] = city
    detected = browser_tz()
    zones = tz_options(od.get("timezone") or detected)
    zone = st.selectbox("Time zone", zones, index=zones.index(od["timezone"]) if od.get("timezone") in zones else 0,
                        key="ob_tz")
    od["timezone"] = zone
    set_tz(zone)
    st.caption((f"Your browser reports {detected}. " if detected else "")
               + f"With your agreement the app uses {zone} for today's date, the calendar and the times it shows. "
                 f"It is now {local_now():%a %d %b, %H:%M} there. Pick another zone if that is wrong.")
    st.markdown("")
    c_back, c_next = st.columns([1, 1])
    with c_next:
        if st.button(T("btn_next_appliances"), width="stretch", type="primary", key="ob_next1"):
            st.session_state.onboard_step = 2
            st.rerun()
    with c_back:
        st.button(T("btn_back"), width="stretch", disabled=True, key="ob_back1")


def _render_onboard_step2(od):
    section(T("ob_your_appliances"))
    st.markdown(f'<div style="color:var(--mist);font-size:0.82rem;margin-bottom:0.8rem;">'
                f'{T("ob_app_hint")}</div>',
                unsafe_allow_html=True)
    appliances = od.get("appliances", [])
    if appliances:
        for i, app in enumerate(appliances):
            if not isinstance(app, dict):
                app = {"name": str(app), "type": "Other", "usage": "Medium", "icon": ""}
            app_name = localized_appliance_label(app.get("name", ""), st.session_state.get("lang", "en"))
            app_type = localized_appliance_type(app.get("type", ""), st.session_state.get("lang", "en"))
            cols = st.columns([2, 2, 0.8], vertical_alignment="center")
            with cols[0]:
                st.markdown(f'<div class="ep-row-name">{app_name}</div>'
                            f'<div class="ep-row-kind">{app_type}</div>', unsafe_allow_html=True)
            with cols[1]:
                usage = st.select_slider(T("typical_usage"), options=["Low", "Medium", "High"],
                    value=app.get("usage", "Medium"), key=f"app_usage_{i}")
                if isinstance(app, dict) and i < len(od["appliances"]):
                    od["appliances"][i]["usage"] = usage
            with cols[2]:
                if st.button("Remove", key=f"app_del_{i}"):
                    od["appliances"].pop(i)
                    st.rerun()
        st.markdown("---")
    with st.expander(T("add_appliance_expander"), expanded=not appliances):
        add_cols = st.columns([2, 2, 1])
        with add_cols[0]:
            preset_pairs = [
                (localized_appliance_label(p_name), p_name) for p_name, _ in APPLIANCE_PRESETS
            ]
            preset_label_to_name = dict(preset_pairs)
            options = [label for label, _ in preset_pairs] + [T("custom_option")]
            chosen = st.selectbox(T("appliance_label"), options, key="ob_app_preset")
            custom_name = ""
            if chosen == T("custom_option"):
                custom_name = st.text_input(T("name_label"), placeholder=T("name_ph"), key="ob_app_custom")
        with add_cols[1]:
            type_options = ["Cooling", "Heating", "Kitchen", "Laundry", "Electronics", "Lighting", "Other"]
            # The category follows the appliance picked; a custom item starts as "Other".
            usual = {"Air Conditioner": "Cooling", "Refrigerator": "Kitchen", "Washing Machine": "Laundry",
                     "Water Heater": "Heating", "Television": "Electronics", "Microwave": "Kitchen",
                     "Lights & Fans": "Lighting"}.get(preset_label_to_name.get(chosen, ""), "Other")
            app_type = st.selectbox(
                T("category_label"),
                [localized_appliance_type(t) for t in type_options],
                index=type_options.index(usual), key=f"ob_app_type_{type_options.index(usual)}_{options.index(chosen)}")
            type_label_to_value = {
                localized_appliance_type(t): t for t in type_options
            }
        with add_cols[2]:
            st.markdown('<div style="margin-top:1.8rem;"></div>', unsafe_allow_html=True)
            if st.button(T("btn_add"), key="ob_add_app", type="primary"):
                if chosen == T("custom_option"):
                    name = custom_name if custom_name else T("custom_option")
                else:
                    name = preset_label_to_name.get(chosen, chosen)
                canonical_type = type_label_to_value.get(app_type, "Other")
                icon = ""
                for p_name, p_icon in APPLIANCE_PRESETS:
                    if p_name == name:
                        icon = p_icon
                        break
                od["appliances"].append({"name": name, "type": canonical_type, "usage": "Medium", "icon": icon})
                st.rerun()
    st.markdown("")
    if not od.get("appliances"):
        st.caption("Add at least one appliance to continue. Your 3D home shows only what you add here.")
    c_back, c_next = st.columns([1, 1])
    with c_back:
        if st.button(T("btn_back"), width="stretch", key="ob_back2"):
            st.session_state.onboard_step = 1
            st.rerun()
    with c_next:
        if st.button(T("btn_next_rate"), width="stretch", type="primary", key="ob_next2",
                     disabled=not od.get("appliances")):
            st.session_state.onboard_step = 3
            st.rerun()


def _render_onboard_step3(od):
    section(T("rate_section_title"))
    st.markdown(f'<div style="color:var(--mist);font-size:0.82rem;margin-bottom:0.8rem;">'
                f'{T("rate_hint")}</div>',
                unsafe_allow_html=True)
    city = od.get("city", "")
    tariff_val = od.get("tariff_rate", 8.0)
    if city and tariff_val == 8.0:
        suggested = CITY_TARIFF_DEFAULTS.get(city.strip().title(), None)
        if suggested:
            st.info(T("avg_rate_for", city=city.strip().title(), rate=suggested))
            if st.button(T("use_city_rate", rate=suggested), key="ob_use_city_rate", type="secondary"):
                od["tariff_rate"] = suggested
                st.rerun()
    tariff = st.number_input(T("your_rate_label"),
        min_value=0.5, max_value=30.0, value=float(od.get("tariff_rate", 8.0)),
        step=0.5, key="ob_tariff")
    od["tariff_rate"] = tariff
    st.markdown(f"""
    <div style="background:var(--ink);border:1px solid var(--line);border-radius:10px;padding:0.8rem 1rem;margin-top:0.3rem;">
        <div style="font-size:0.78rem;color:var(--mist);">
            {T("rate_hint_avg")}
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")
    c_back, c_next = st.columns([1, 1])
    with c_back:
        if st.button(T("btn_back"), width="stretch", key="ob_back3"):
            st.session_state.onboard_step = 2
            st.rerun()
    with c_next:
        if st.button(T("btn_start_dashboard"), width="stretch", type="primary", key="ob_finish"):
            st.session_state.home_details = {
                "home_type": od["home_type"],
                "occupants": od["occupants"],
                "appliances": od["appliances"],
                "num_appliances": len(od["appliances"]),
                "tariff_rate": od["tariff_rate"],
                "city": od.get("city", ""),
                "timezone": od.get("timezone") or user_tz(),
                "home_size": od.get("home_size", 0) if od.get("home_size", 0) > 0 else None,
                "peak_morning": (6, 10),
                "peak_evening": (18, 22),
            }
            try:
                get_db().save_home_details(current_household_id(), st.session_state.home_details)
            except Exception:
                pass            # the session copy still works; it is just not remembered
            st.session_state.pop("home_state", None)        # rebuild the 3D home from the new list
            for k in ["onboard_data", "onboard_step"]:
                if k in st.session_state:
                    del st.session_state[k]
            notify("success", "Home saved. Every appliance you listed is now in the 3D home.")
            st.rerun()


@st.cache_data(show_spinner=False)
def _brief_cached(date, scenario, owned, rate, tod):
    """Today's per-appliance forecast and savings plan (see daily_brief.py)."""
    try:
        return build_brief(detected_history(date, scenario, owned), owned, Tariff(rate=rate, tod_enabled=tod))
    except Exception:
        return None


def todays_brief(tariff_rate, owned, household_id, user_name, user_email):
    """Build today's forecast and, the first time it is seen today, notify the household."""
    scenario, tod = meter_settings()
    brief = _brief_cached(today_str(), scenario, tuple(owned), float(tariff_rate), tod)
    if not brief:
        return None
    mark = (household_id, brief["date"])
    if st.session_state.get("_brief_mark") != mark:
        st.session_state["_brief_mark"] = mark
        try:
            db = get_db()
            sent = send_daily_brief(db, household_id, household_people(db, household_id, user_name, user_email),
                                    brief, st.session_state.get("lang", "en"), get_notification_service())
        except Exception:
            sent = None
        if sent:
            notify("info", sent["subject"] + ". The plan is on the Overview page"
                   + (" and in your email." if sent["emailed"] else "."))
    return brief


def render_brief_strip(brief):
    """One line above the 3D home."""
    if not brief:
        return
    top = brief["actions"][0] if brief["actions"] else None
    st.markdown(
        f"<div class='ep-brief'><span><b>Today's forecast</b> &nbsp;<strong>{brief['total_kwh']:.1f}</strong> units, "
        f"about <strong>Rs. {brief['total_cost']:.0f}</strong></span>"
        f"<span>off by {brief['error_pct']:.1f}% on recent days</span>"
        + (f"<span><b>Best change today:</b> {top['title']} (about Rs. {top['saving_month']:.0f} a month)</span>" if top else "")
        + "</div>", unsafe_allow_html=True)


def render_brief_section(brief):
    """Today's prediction and the savings plan, at the top of Overview."""
    if not brief:
        return
    section(f"Today's prediction, {brief['date_label']}")
    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, label="Expected today", value=f"{brief['total_kwh']:.1f}", unit="units",
                sub=f"likely {brief['low_kwh']:.1f} to {brief['high_kwh']:.1f}")
    metric_card(c2, label="Expected cost today", value=f"Rs. {brief['total_cost']:.0f}",
                sub=f"Rs. {brief['low_cost']:.0f} to {brief['high_cost']:.0f}")
    metric_card(c3, label="Forecast error", value=f"{brief['error_pct']:.1f}", unit="%",
                sub=f"measured over the last {brief['backtest_days']} days")
    metric_card(c4, label="Savings found", value=f"Rs. {brief['saving_month']:.0f}", unit="/month",
                sub=f"out of about Rs. {brief['month_cost']:.0f} a month")
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    left, right = st.columns([1, 1.15], gap="medium")
    with left:
        with st.container(border=True):
            section("Expected use by appliance")
            biggest = max([a["kwh"] for a in brief["appliances"]] + [brief["other_kwh"], 0.01])
            rows = "".join(
                f"<tr><td class='n'>{a['name']}<small>{a['usual'] or 'runs through the day'}</small></td>"
                f"<td><div class='bar'><span style='width:{100 * a['kwh'] / biggest:.0f}%'></span></div></td>"
                f"<td class='v'>{a['kwh']:.2f} units<small>Rs. {a['cost']:.1f}</small></td></tr>"
                for a in brief["appliances"])
            rows += (f"<tr><td class='n'>Everything else<small>lights, fans, TV, standby</small></td>"
                     f"<td><div class='bar'><span style='width:{100 * brief['other_kwh'] / biggest:.0f}%'></span></div></td>"
                     f"<td class='v'>{brief['other_kwh']:.2f} units<small>Rs. {brief['other_cost']:.1f}</small></td></tr>")
            st.markdown(f"<table class='ep-rows'>{rows}</table>", unsafe_allow_html=True)
            if brief["yesterday"]:
                st.caption(f"Yesterday the forecast was {brief['yesterday']['forecast']:.1f} units and the "
                           f"meter recorded {brief['yesterday']['actual']:.1f}.")
    with right:
        with st.container(border=True):
            section("What to change, biggest saving first")
            if brief["actions"]:
                items = "".join(
                    f"<div class='ep-plan-item'><i>{n}</i><b>{a['title']}</b>"
                    f"<em>Rs. {a['saving_month']:.0f}/month</em>"
                    f"<p>{a['detail']}</p><code>{a['calculation']}. About Rs. {a['saving_day']:.1f} a day.</code></div>"
                    for n, a in enumerate(brief["actions"], 1))
                st.markdown(f"<div class='ep-plan'>{items}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color:var(--mist);font-size:.86rem'>No change is worth making today. "
                            "Nothing would save more than Rs. 30 a month.</p>", unsafe_allow_html=True)
            for note in brief["notes"][:2]:
                st.caption(note)
            st.caption("Each figure shows how it was worked out.")
    st.caption(f"Forecast rule: the average of the last 7 days and of the same weekday in the last 4 weeks, from "
               f"{brief['history_days']} days of main-meter history. This notification is sent once a day and is "
               f"kept on the Notifications page.")
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)


PAGES = [("home", "tab_home"), ("overview", "tab_overview"), ("appliances", "tab_appliances"),
         ("analysis", "tab_analysis"), ("save", "tab_save"), ("goals", "tab_goals"),
         ("safety", "tab_safety"), ("bills", "tab_bills"), ("upgrades", "tab_upgrades"),
         ("trends", "tab_trends"), ("family", "tab_family"), ("alerts", "tab_notifications"),
         ("assistant", "tab_chat"), ("help", "tab_help"), None, ("settings", "tab_settings")]

# Page titles are written for English; other languages use the page's own name.
PAGE_TITLES = {
    "home": ("Your home, right now", "Appliances light up as the meter reading gives them away."),
    "overview": ("The next hour, the next month", "A forecast of how much you will use and what it will cost."),
    "appliances": ("What the meter gives away", "Every appliance found in the main-meter signal, and how long it ran."),
    "analysis": ("Where the power goes", "Units and rupees, by appliance and by time of day."),
    "save": ("Changes worth making", "Only suggestions a household could follow, each with its working shown."),
    "goals": ("Stay on target", "A monthly goal, your planned hours, similar homes and your carbon."),
    "safety": ("Faults and overload", "What ran too long, what is wearing out, and how close you come to tripping the breaker."),
    "upgrades": ("Is it worth buying?", "A new appliance, rooftop solar or a backup, costed from your own use."),
    "help": ("Get it sorted", "Log power cuts and draft a complaint with your own figures."),
    "bills": ("Your bills", "Keep past bills in one place and compare them with the forecast."),
    "trends": ("How your use moves", "Recent readings and a month at a glance."),
    "family": ("The household", "Who gets which alerts, and in which language."),
    "alerts": ("Alerts", "Bill warnings and tips, sent by email or text."),
    "assistant": ("Ask about your energy", "Answers come from your own readings."),
    "settings": ("Settings", "Your home, your tariff and the demo data."),
}


def _default_settings(hd):
    now = local_now()
    return {"tariff_rate": float(hd.get("tariff_rate", 8.0)), "peak_morning": (6, 10),
            "peak_evening": (18, 22), "display_count": 100,
            "cal_month": int(now.month), "cal_year": int(now.year)}


def render_sidebar():
    """Left navigation.  Controls that used to live here are on the Settings page."""
    init_language()
    hd = st.session_state.home_details
    st.session_state.setdefault("page", "home")
    if "sidebar_settings" not in st.session_state:
        st.session_state.sidebar_settings = _default_settings(hd)
    with st.sidebar:
        st.markdown(wordmark() + "<div style='height:1.6rem'></div>", unsafe_allow_html=True)
        sidebar_nav([None if p is None else (p[0], T(p[1])) for p in PAGES], st.session_state.page)
        user_name = st.session_state.auth.get("name") or T("guest_name")
        size = f", {T('sqft_suffix', n=hd.get('home_size'))}" if hd.get("home_size") else ""
        st.markdown(
            f"<div class='ep-side-home'><b>{home_type_label(hd.get('home_type', 'Home'))}</b>"
            f"<span>{user_name} &middot; {T('people_suffix', n=hd.get('occupants', 0))}{size}</span></div>",
            unsafe_allow_html=True)
        theme_toggle()
        if st.button("Log out", key="side_logout", width="stretch"):
            confirm_sign_out()
    return st.session_state.sidebar_settings


def render_settings_page():
    hd = st.session_state.home_details
    cur = st.session_state.sidebar_settings
    appliances = hd.get("appliances", [])
    lang = st.session_state.get("lang", "en")
    left, right = st.columns(2, gap="large")

    with left:
        with st.container(border=True):
            section(T("sb_section_home"))
            size = f", {T('sqft_suffix', n=hd.get('home_size'))}" if hd.get("home_size") else ""
            st.markdown(f"**{home_type_label(hd.get('home_type', 'Home'))}**  \n"
                        f"{T('people_suffix', n=hd.get('occupants', 0))}{size}")
            if appliances:
                names = [localized_appliance_label(a.get("name", "") if isinstance(a, dict) else str(a), lang)
                         for a in appliances]
                st.caption(", ".join(names))
            else:
                st.caption(T("sb_no_appliances"))
            if st.button(T("sb_edit_home"), key="sb_edit_home"):
                st.session_state.home_details = None
                st.session_state.onboard_step = 1
                st.session_state.onboard_data = {
                    "home_type": hd.get("home_type", "Apartment"), "occupants": hd.get("occupants", 1),
                    "appliances": appliances, "tariff_rate": hd.get("tariff_rate", 8.0),
                    "city": hd.get("city", ""), "home_size": hd.get("home_size", 0) or 0,
                    "timezone": hd.get("timezone") or user_tz(),
                }
                st.rerun()
        with st.container(border=True):
            section(T("sb_section_rate"))
            tariff_rate = st.number_input(T("sb_rate_label"), min_value=0.0,
                                          value=float(hd.get("tariff_rate", 8.0)), step=0.5, key="sb_tariff")
            hd["tariff_rate"] = tariff_rate
            st.caption(T("sb_rate_hint"))
            peak_morning = st.slider(T("sb_peak_am"), 0, 23, tuple(cur["peak_morning"]), key="sb_peak_am")
            peak_evening = st.slider(T("sb_peak_pm"), 0, 23, tuple(cur["peak_evening"]), key="sb_peak_pm")
        with st.container(border=True):
            render_language_selector(key="lang_select")
            zones = tz_options(hd.get("timezone") or user_tz())
            zone = st.selectbox("Time zone", zones, index=zones.index(user_tz()) if user_tz() in zones else 0,
                                key="sb_tz", help="Used for today's date, the calendar and the times shown.")
            if zone != hd.get("timezone"):
                hd["timezone"] = zone
                set_tz(zone)
                try:
                    get_db().save_home_details(current_household_id(), hd)
                except Exception:
                    pass
            st.caption(f"It is now {local_now():%a %d %b, %H:%M} in {user_tz()}.")
            if st.button(T("btn_sign_out"), key="sb_signout"):
                confirm_sign_out()

    with right:
        with st.container(border=True):
            section("Meter demo")
            render_meter_settings()
        with st.container(border=True):
            section(T("sb_section_display"))
            display_count = st.slider(T("sb_trend_window"), 20, 500, int(cur["display_count"]), key="sb_display")
            st.caption(T("sb_viewing_data"))
            cal_cols = st.columns(2)
            month_names = TLIST("months")
            with cal_cols[0]:
                cal_month_name = st.selectbox(T("month_label"), month_names,
                                              index=int(cur["cal_month"]) - 1, key="cal_month_sb")
                cal_month = month_names.index(cal_month_name) + 1
            with cal_cols[1]:
                cal_year = st.number_input(T("year_label"), 2006, 2030, value=int(cur["cal_year"]), key="cal_year_sb")
        with st.container(border=True):
            section(T("sb_section_data"))
            render_data_source_panel()
            if st.button(T("btn_restart_demo"), key="sb_restart"):
                if "simulator" in st.session_state:
                    st.session_state.simulator.reset()
                    st.session_state.simulator.start()
                st.rerun()
            st.caption(T("sb_restart_hint"))

    st.session_state.sidebar_settings = {
        "tariff_rate": tariff_rate, "peak_morning": peak_morning, "peak_evening": peak_evening,
        "display_count": display_count, "cal_month": cal_month, "cal_year": cal_year}


def render_calendar(df, forecast_df, year, month, tariff_rate):
    cal_obj = cal.Calendar(firstweekday=0)
    month_days = cal_obj.monthdayscalendar(year, month)
    daily_usage = {}
    if not df.empty:
        month_mask = (df["datetime"].dt.year == year) & (df["datetime"].dt.month == month)
        month_data = df.loc[month_mask].copy()
        if not month_data.empty:
            month_data["date"] = month_data["datetime"].dt.date
            daily_kwh = month_data.groupby("date")[TARGET].sum()
            for d, kwh in daily_kwh.items():
                daily_usage[d] = kwh * tariff_rate
    daily_forecast = {}
    if not forecast_df.empty:
        fc_month = forecast_df[
            (forecast_df["datetime"].dt.year == year) &
            (forecast_df["datetime"].dt.month == month)
        ]
        if not fc_month.empty:
            fc_daily = fc_month.groupby(fc_month["datetime"].dt.date)["predicted_kwh"].sum()
            for d, kwh in fc_daily.items():
                daily_forecast[d] = kwh * tariff_rate
    all_costs = list(daily_usage.values()) + list(daily_forecast.values())
    p33, p66 = (np.percentile(all_costs, 33), np.percentile(all_costs, 66)) if all_costs else (10, 30)

    def classify(cost):
        return "low" if cost <= p33 else ("medium" if cost <= p66 else "high")

    html = '<div style="background:var(--slate);border:1px solid var(--line);border-radius:16px;padding:1.2rem;box-shadow:0 1px 3px rgba(0,0,0,0.04),0 4px 12px rgba(0,0,0,0.03);">'
    html += f'<div style="color:var(--plaster);font-size:1rem;font-weight:700;margin-bottom:0.8rem;text-align:center;">{TLIST("months")[month - 1]} {year}</div>'
    if not daily_usage and not daily_forecast:
        html += f'<div style="text-align:center;padding:2rem 1rem;color:var(--faint);font-size:0.85rem;">{T("cal_no_data")}</div>'
    else:
        html += '<div class="cal-grid">'
        for d in TLIST("weekdays"):
            html += f'<div class="cal-header">{d}</div>'
        for week in month_days:
            for day in week:
                if day == 0:
                    html += '<div class="cal-day empty"></div>'
                    continue
                dt = pd.Timestamp(year, month, day).date()
                if dt in daily_usage:
                    cost = daily_usage[dt]
                    html += f'<div class="cal-day {classify(cost)}" title="{dt}: Rs. {cost:.2f}">{day}<br><span style="font-size:0.6rem">Rs.{cost:.0f}</span></div>'
                elif dt in daily_forecast:
                    cost = daily_forecast[dt]
                    html += f'<div class="cal-day future" title="{dt}: Rs. {cost:.2f} (est.)">{day}<br><span style="font-size:0.6rem">Rs.{cost:.0f}</span></div>'
                else:
                    html += f'<div class="cal-day empty">{day}</div>'
        html += '</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ── Bill upload (photo / file) ────────────────────────────────
BILL_FILE_TYPES = ["jpg", "jpeg", "png", "webp", "heic", "heif", "pdf"]
IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "heic", "heif"}

UNIT_RE = re.compile(r"(\d+(?:[.,]\d+)?)[ \t]*(?:kwh|units?)\b", re.IGNORECASE)
UNIT_PREFIX_RE = re.compile(
    r"(?:total[ \t]*units?|units?[ \t]*consumed|consumption|units?\b)[^0-9\n]{0,15}?(\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)
AMOUNT_RE = re.compile(r"(?:rs\.?|inr|\u20b9)\s*[:\-]?\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE)
PERIOD_RE = re.compile(
    r"(\d{1,2}[-/][A-Za-z0-9]{2,7}[-/]\d{2,4})\s*(?:to|-|\u2013|until)\s*(\d{1,2}[-/][A-Za-z0-9]{2,7}[-/]\d{2,4})"
    r"|([A-Za-z]{3,9}[\s'-]\d{2,4})\s*(?:to|-|\u2013)\s*([A-Za-z]{3,9}[\s'-]\d{2,4})"
)


def ocr_image_bytes(data: bytes):
    """
    Best-effort OCR on an image. Returns (text, status):
      status True  -> OCR ran successfully
      status False -> the OCR engine is unavailable on this machine
      status None  -> the engine ran but could not read this image

    The two failure modes are kept apart. Importing pytesseract succeeds as soon
    as the Python package is installed, so a machine without the Tesseract
    binary used to raise TesseractNotFoundError inside the call, get swallowed
    by the broad handler below, and be reported as a bad image.
    """
    try:
        import io
        import pytesseract
        from PIL import Image
    except ImportError:
        return "", False
    try:
        # pytesseract imports fine without the binary; this is what detects it.
        pytesseract.get_tesseract_version()
    except Exception:
        return "", False
    try:
        img = Image.open(io.BytesIO(data))
        if img.mode not in ("L", "RGB"):
            img = img.convert("RGB")
        return pytesseract.image_to_string(img), True
    except Exception:
        return "", None


def extract_bill_details(text: str) -> dict:
    """Pull units consumed, amount and billing period out of raw bill text."""
    details = {}
    if not text:
        return details
    m = UNIT_RE.search(text)
    if m:
        details["units"] = m.group(1)
    else:
        m = UNIT_PREFIX_RE.search(text)
        if m:
            details["units"] = m.group(1)
    amounts = AMOUNT_RE.findall(text)
    if amounts:
        biggest = max(amounts, key=lambda a: float(a.replace(",", "") or 0))
        details["amount"] = biggest
    m = PERIOD_RE.search(text)
    if m:
        groups = [g for g in m.groups() if g]
        if len(groups) >= 2:
            details["period"] = f"{groups[0]} - {groups[1]}"
    return details


def month_options():
    """Last 12 months as localized labels, newest first."""
    now = local_now()
    opts = []
    for i in range(12):
        d = now - pd.DateOffset(months=i)
        opts.append(f"{TLIST('months')[d.month - 1]} {d.year}")
    return opts


def save_uploaded_bill(data: bytes, filename: str, ext: str, month_label: str):
    """OCR (images only), store against the chosen month, report what happened."""
    text, ocr_status = (("", False) if ext == "pdf" else ocr_image_bytes(data))
    details = extract_bill_details(text)
    household = current_household_id()
    payload = {"name": filename, "ext": ext, "month": month_label, "details": details}
    path = store_bill_file(household, data, ext)
    entry = get_ledger().record(household, "bill", filename, data, payload)
    st.session_state.setdefault("bills", []).append({
        "name": filename,
        "ext": ext,
        "data": data,
        "month": month_label,
        "details": details,
        "saved_at": entry["created_at"][:16],
        "ledger_id": entry["id"],
        "path": path,
    })

    notify("success", T("bill_saved_ok", month=month_label) + " It has been fingerprinted and sealed.")
    if ext != "pdf" and ocr_status is False:
        st.info(T("ocr_unavailable"))
    elif ext != "pdf" and ocr_status is None:
        st.warning(T("ocr_failed"))
    elif ext == "pdf":
        st.caption(T("pdf_stored_note"))
    if details:
        chips = "".join([
            f'<span><strong style="color:var(--current);">{T("units_consumed")}:</strong> {details.get("units", "-")}</span>',
            f'<span><strong style="color:var(--current);">{T("amount_paid")}:</strong> '
            + (f'Rs. {details.get("amount")}' if details.get("amount") else "-") + '</span>',
            f'<span><strong style="color:var(--current);">{T("billing_period")}:</strong> {details.get("period", "-")}</span>',
        ])
        st.markdown(f"""
        <div class="bill-card">
            <div style="font-weight:700;font-size:0.85rem;color:var(--plaster);margin-bottom:0.35rem;">
                {T("extracted_details")}
            </div>
            <div style="display:flex;flex-wrap:wrap;gap:6px 18px;font-size:0.82rem;color:var(--soft);">
                {chips}
            </div>
        </div>
        """, unsafe_allow_html=True)


@st.cache_resource
def get_ledger() -> Ledger:
    return Ledger()


def current_household_id() -> str:
    auth = st.session_state.get("auth", {})
    if not auth.get("logged_in") or not auth.get("email"):
        raise PermissionError("Sign in to see this household's data.")
    email = auth["email"]
    user = get_db().get_user(email)
    return user["household_id"] if user else hashlib.md5(email.strip().lower().encode()).hexdigest()


def load_saved_bills():
    """Bring back bills saved in earlier sessions, leaving out ones the user removed."""
    if "bills" in st.session_state:
        return
    household = current_household_id()
    entries = get_ledger().entries(household)
    removed = {json.loads(e["payload_json"]).get("removes") for e in entries if e["kind"] == "bill_removed"}
    bills = []
    for e in entries:
        if e["kind"] != "bill" or e["id"] in removed:
            continue
        p = json.loads(e["payload_json"])
        path = bill_path(household, e["content_hash"], p.get("ext", "bin"))
        bills.append({"name": p.get("name", e["ref"]), "ext": p.get("ext", "bin"),
                      "data": read_bill_file(path) or b"", "month": p.get("month", ""),
                      "details": p.get("details", {}), "saved_at": e["created_at"][:16],
                      "ledger_id": e["id"], "path": path})
    st.session_state.bills = bills


def bill_status(bill: dict):
    """(ok, message) comparing the stored file and figures with the sealed record."""
    entry = next((e for e in get_ledger().entries(current_household_id(), "bill")
                  if e["id"] == bill.get("ledger_id")), None)
    if entry is None:
        return False, "No sealed record exists for this bill."
    on_disk = read_bill_file(bill.get("path", ""))
    if on_disk is None:
        return False, "The saved file is missing."
    payload = {"name": bill["name"], "ext": bill["ext"], "month": bill["month"], "details": bill["details"]}
    return get_ledger().check_content(entry, on_disk, payload)


@st.dialog("Remove this bill?")
def confirm_bill_removal(idx: int):
    bills = st.session_state.get("bills", [])
    if idx >= len(bills):
        st.rerun()
    b = bills[idx]
    st.write(f"**{b['name']}** will be taken off your list. The sealed record of it stays, "
             f"and the removal is recorded too, so the history cannot be quietly rewritten.")
    keep, remove = st.columns(2)
    if keep.button("Keep it", key="bill_keep", width="stretch"):
        st.rerun()
    if remove.button("Remove", key="bill_remove", type="primary", width="stretch"):
        # Records are never erased: the removal itself is recorded.
        get_ledger().record(current_household_id(), "bill_removed", b["name"], b"",
                            {"removes": b.get("ledger_id")})
        bills.pop(idx)
        notify("info", f"{b['name']} was removed. The removal is on record.")
        st.rerun()


def bill_signature(data: bytes, name: str) -> str:
    return f"{name}:{hashlib.md5(data).hexdigest()}"


def render_saved_bills():
    bills = st.session_state.get("bills", [])
    if not bills:
        st.markdown(f"""
        <div class="empty-state">
            <div class="empty-state-title">{T('no_bills_yet')}</div>
        </div>
        """, unsafe_allow_html=True)
        return
    section(T("saved_bills_header", n=len(bills)))
    for idx in range(len(bills) - 1, -1, -1):
        b = bills[idx]
        with st.container():
            c_img, c_info, c_del = st.columns([1, 2.2, 0.6])
            with c_img:
                shown = False
                if b["ext"] in IMAGE_EXTS:
                    try:
                        st.image(b["data"], width=140)
                        shown = True
                    except Exception:
                        shown = False
                if not shown:
                    icon = "PDF" if b["ext"] == "pdf" else "Image"
                    st.markdown(f'<div style="font-size:0.9rem;color:var(--mist);text-align:center;padding:1.6rem 0;border:1px solid var(--line);border-radius:10px;">{icon}</div>',
                                unsafe_allow_html=True)
            with c_info:
                st.markdown(
                    f'<div style="font-weight:700;font-size:0.88rem;color:var(--plaster);'
                    f'overflow-wrap:anywhere;">{b["name"]}</div>'
                    f'<div style="font-size:0.74rem;color:var(--mist);margin-top:2px;">'
                    f'{b["month"]} · {T("billing_period")}: '
                    f'{b["details"].get("period", "-")}<br>'
                    f'{T("units_consumed")}: {b["details"].get("units", "-")} · '
                    f'{T("amount_paid")}: Rs. {b["details"].get("amount", "-")}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                ok, message = bill_status(b)
                st.markdown(f'<div class="ep-seal {"ok" if ok else "bad"}">{message}</div>',
                            unsafe_allow_html=True)
            with c_del:
                st.write("")
                if st.button(T("btn_delete_bill"), key=f"bill_del_{idx}",
                             type="secondary"):
                    confirm_bill_removal(idx)


def render_bills_tab():
    init_language()
    load_saved_bills()
    section(T("tab_bills"))
    st.markdown(f'<div style="color:var(--mist);font-size:0.85rem;margin-bottom:0.8rem;">'
                f'{T("bill_intro")}</div>', unsafe_allow_html=True)

    up_col, cam_col = st.columns(2)
    uploaded = None
    with up_col:
        uploaded = st.file_uploader(
            T("bill_uploader_label"),
            type=BILL_FILE_TYPES,
            help=T("bill_uploader_help"),
            key="bill_uploader",
        )
    with cam_col:
        camera_img = st.camera_input(
            T("bill_camera_label"),
            help=T("bill_camera_help"),
            key="bill_camera",
        )

    month_choice = st.selectbox(T("bill_month_label"), month_options(),
                                index=0, key="bill_month")

    if uploaded is not None:
        data = uploaded.getvalue()
        sig = bill_signature(data, uploaded.name)
        seen = st.session_state.setdefault("bill_seen", [])
        if sig not in seen:
            seen.append(sig)
            ext = uploaded.name.lower().split(".")[-1] or "bin"
            save_uploaded_bill(data, uploaded.name, ext, month_choice)
    elif camera_img is not None:
        data = camera_img.getvalue()
        sig = bill_signature(data, "camera")
        seen = st.session_state.setdefault("bill_seen", [])
        if sig not in seen:
            seen.append(sig)
            ts = local_now().strftime("%Y%m%d_%H%M%S")
            save_uploaded_bill(data, f"bill_camera_{ts}.jpg", "jpeg", month_choice)

    render_saved_bills()
    render_record_checks()


def render_record_checks():
    """Tamper checks: the record chain, any file against the records, and sealed reports."""
    household = current_household_id()
    ledger = get_ledger()
    section("Are these records untouched?")
    problems = ledger.verify_chain(household)
    count = len(ledger.entries(household))
    if problems:
        st.error("The saved records have been altered.")
        for p in problems[:6]:
            st.markdown(f'<div class="ep-seal bad">{p}</div>', unsafe_allow_html=True)
    elif count:
        st.markdown(f'<div class="ep-seal ok">All {count} records are intact. Each one is fingerprinted, '
                    f'linked to the one before it and sealed.</div>', unsafe_allow_html=True)
    else:
        st.caption("Nothing has been saved yet. Bills are fingerprinted and sealed when you save them.")

    c_check, c_report = st.columns(2, gap="large")
    with c_check:
        with st.container(border=True):
            st.markdown("**Check a bill file**")
            st.caption("Upload a copy of a bill. It is compared with the fingerprints of the bills saved here.")
            probe = st.file_uploader("Bill file to check", type=BILL_FILE_TYPES, key="bill_probe",
                                     label_visibility="collapsed")
            if probe is not None:
                match = ledger.find_by_content(household, probe.getvalue())
                if match:
                    st.markdown(f'<div class="ep-seal ok">Identical to "{match["ref"]}", saved on '
                                f'{match["created_at"]}.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="ep-seal bad">No saved bill matches this file. It has been '
                                'changed, or it was never saved here.</div>', unsafe_allow_html=True)
    with c_report:
        with st.container(border=True):
            st.markdown("**Sealed report**")
            st.caption("A summary of the last 30 days with a seal at the end. If anyone edits a number, "
                       "the seal stops matching.")
            hd = st.session_state.get("home_details") or {}
            body = report_body(float(hd.get("tariff_rate", 8.0)), owned_appliances(hd),
                               st.session_state.get("auth", {}).get("name") or "Household")
            st.download_button("Download sealed report", sign_report(body),
                               file_name=f"energypulse_report_{local_now():%Y%m%d}.txt",
                               mime="text/plain", key="report_download")
            check = st.file_uploader("Report to check", type=["txt"], key="report_probe")
            if check is not None:
                ok, message = verify_report(check.getvalue().decode("utf-8", errors="replace"))
                st.markdown(f'<div class="ep-seal {"ok" if ok else "bad"}">{message}</div>',
                            unsafe_allow_html=True)


def main_dashboard():
    init_language()
    try:
        _main_dashboard_inner()
    except Exception as e:
        st.markdown(f"""
        <div class="error-card">
            <div class="error-card-title">{T('error_title')}</div>
            <div class="error-card-msg">
                {T('error_msg')}
            </div>
        </div>
        """, unsafe_allow_html=True)
        if e:
            # The exception text is kept for debugging but is never the primary
            # message: it is untranslated, can contain markup, and means nothing
            # to the person looking at the screen.
            with st.expander(T("error_details_toggle")):
                st.code(f"{type(e).__name__}: {str(e)[:500]}", language="text")
        c1, c2 = st.columns(2)
        with c1:
            if st.button(T("btn_go_home"), type="primary", key="err_go_home"):
                st.session_state.home_details = None
                st.session_state.onboard_step = 1
                st.rerun()
        with c2:
            if st.button(T("btn_retry"), key="err_retry"):
                st.rerun()


def _main_dashboard_inner():
    home_details = st.session_state.home_details
    set_tz(home_details.get("timezone") or user_tz())
    ss = st.session_state.get("sidebar_settings", {})
    tariff_rate = home_details.get("tariff_rate", 8.0)
    peak_morning = ss.get("peak_morning", (6, 10))
    peak_evening = ss.get("peak_evening", (18, 22))
    display_count = ss.get("display_count", 100)
    cal_month = ss.get("cal_month", local_now().month)
    cal_year = ss.get("cal_year", local_now().year)
    sf = compute_scaling_factor(home_details)

    auth_email = st.session_state.auth["email"]
    primary_user = get_db().get_user(auth_email)
    household_id = primary_user["household_id"] if primary_user else hashlib.md5(
        (auth_email or "").strip().lower().encode()).hexdigest()

    with st.spinner(T("spinner_preparing")):
        source = get_data_source()
        raw_data = active_dataframe()
        xgb_model = load_xgb()
        lstm_model = load_lstm()
        scaler = load_scaler()
        meta = load_meta()
        forecast_df = load_forecast()

    if raw_data is None or raw_data.empty:
        st.error(T("src_upload_fail", reason=T("src_need_power")))
        return

    src_info = source_summary(source)
    can_predict = src_info["caps"].get("prediction", {}).get("enabled", False)
    can_appliances = src_info["caps"].get("appliances", {}).get("enabled", False)

    full_data = raw_data
    if not forecast_df.empty and can_predict:
        forecast_df = shift_forecast_to_current_dates(forecast_df, raw_data, full_data)

    full_data = full_data.copy()
    full_data["Global_active_power"] = full_data["Global_active_power"] * sf
    if not forecast_df.empty:
        forecast_df["predicted_kw"] = forecast_df["predicted_kw"] * sf
        forecast_df["predicted_kwh"] = forecast_df["predicted_kwh"] * sf

    sim = get_simulator()
    hybrid_mae = meta.get("hybrid_mae", 0.15)

    # The replay advances on a background thread, but Streamlit only redraws on
    # interaction, so without this the displayed reading stays frozen while the
    # simulator moves on. Refresh only while the replay is actually moving, and
    # stop once it reaches the end so the session does not rerun forever.
    if st.session_state.get("page") == "overview":
        refresh_replay(sim)

    # Only the shifted sample dataset may be matched against the wall clock.
    # An upload keeps its real timestamps, so its "current" reading is simply
    # the newest one it contains.
    if source.is_sample:
        live_row = sim.get_live_row_for_current_time(full_data)
    else:
        replayed = sim.get_full_history()
        live_row = replayed.iloc[-1].to_dict() if not replayed.empty else None
    if live_row is None:
        live_row = sim.latest_row

    rows_played, total_rows = sim.progress
    progress_pct = rows_played / total_rows if total_rows > 0 else 0

    if live_row:
        gap = live_row.get(TARGET, 0)
        row_time = live_row.get("datetime")
        when_text = ""
        if row_time is not None and not pd.isna(row_time):
            when_text = T("src_latest",
                          when=format_localized_date(row_time),
                          age=format_data_age(
                              (local_now() - pd.Timestamp(row_time))
                              .total_seconds() / 3600.0))
        live_bar_html = (f"""
        <div class="live-bar">
            <div class="live-dot"></div>
            <span class="live-label">{T('live_label')}</span>
            <span style="color:rgba(0,0,0,0.3);">&middot;</span>
            <span style="color:var(--mist);">{T('live_updated')}</span>
            <span style="color:rgba(0,0,0,0.3);">&middot;</span>
            <span style="color:var(--mist);">{when_text}</span>
            <div style="margin-left:auto;display:flex;align-items:center;gap:8px;">
                <div class="data-progress" style="width:80px;" title="{rows_played:,} / {total_rows:,}">
                    <div class="data-progress-fill" style="width:{progress_pct*100:.0f}%;"></div>
                </div>
            </div>
        </div>
        """)
        # Say plainly what the numbers above are based on.
        source_text = f"{src_info['title']} — {src_info['body']} {T('src_replaying')}"
        missing_notes = []
        if not can_predict:
            missing_notes.append(T(
                "src_below_forecast",
                cols=", ".join(src_info["caps"]["prediction"]["missing"])))
    else:
        st.info(T("getting_ready"))
        return

    replay_window_raw = sim.get_window(WINDOW_SIZE + 10)

    # Predict on the raw readings, then apply the household scaling factor
    # once to the outputs. Pre-scaling the inputs would double-scale the LSTM,
    # which consumes Global_active_power as one of its input features.
    if can_predict:
        pred_kw, xgb_p, lstm_p, pred_info = predict_next_period(
            live_row, xgb_model, lstm_model, scaler, replay_window_raw,
            return_info=True, weights=meta.get("hybrid_weights"),
        )
    else:
        pred_kw, xgb_p, lstm_p = gap, None, None
        pred_info = {"lstm_used": False, "lstm_reason": "unsupported_dataset",
                     "w_xgb": 1.0, "w_lstm": 0.0}
    pred_kw *= sf
    if xgb_p is not None:
        xgb_p *= sf
    if lstm_p is not None:
        lstm_p *= sf
    uncertainty = hybrid_mae * sf
    lower = max(0, pred_kw - uncertainty)
    upper = pred_kw + uncertainty

    history = sim.get_full_history()
    if not history.empty:
        history = history.copy()
        history["Global_active_power"] = history["Global_active_power"] * sf
        latest_date = history["datetime"].dt.date.iloc[-1]
        today_info = daily_cost(history, str(latest_date), tariff_rate,
                                tuple(peak_morning), tuple(peak_evening))
        week_start_date = latest_date - pd.Timedelta(days=6)
        week_info = weekly_cost(history, str(week_start_date), tariff_rate,
                                tuple(peak_morning), tuple(peak_evening))
        nm_info = next_month_cost(forecast_df, tariff_rate,
                                  tuple(peak_morning), tuple(peak_evening))
    else:
        today_info = {"total_cost": 0, "total_kwh": 0}
        week_info = {"total_cost": 0, "pct_change": 0}
        nm_info = {"total_cost": 0, "month": "N/A"}

    page = st.session_state.get("page", "home")
    owned = owned_appliances(home_details)
    user_display = st.session_state.auth.get("name") or T("guest_name")
    # A new animation name per page replays the entrance on navigation only,
    # not on the periodic refresh.
    st.markdown(f"<style>@keyframes ep-in-{page} {{ from {{ opacity: 0; transform: translateY(14px); }} "
                f"to {{ opacity: 1; transform: none; }} }} "
                f"[data-testid='stMainBlockContainer'] > div {{ animation: ep-in-{page} .5s cubic-bezier(.2,.7,.2,1) both; }}"
                f"</style>", unsafe_allow_html=True)

    def head(chip="", chip_kind=""):
        title, lead = PAGE_TITLES[page]
        if st.session_state.get("lang", "en") != "en":
            title, lead = T(dict(p for p in PAGES if p)[page]), ""
        page_header(title, lead, chip, chip_kind)

    meter_chip = "Simulated meter"

    def prepare_meter(kind):
        """Show loading shapes the first time the meter history is worked out, then swap in the page."""
        scenario, _ = meter_settings()
        key = (today_str(), scenario, owned)
        warm = st.session_state.setdefault("_warm", set())
        if key in warm:
            return
        holder = st.empty()
        holder.markdown(skeleton(kind, "Reading the meter and working out which appliances are running..."),
                        unsafe_allow_html=True)
        detected_history(*key)
        warm.add(key)
        holder.empty()

    if page == "home":
        head()
        prepare_meter("console")
        render_brief_strip(todays_brief(tariff_rate, owned, household_id, user_display, auth_email))
        left_mark = (household_id, today_str())
        if st.session_state.get("_left_on_mark") != left_mark:      # once per sign-in per day
            st.session_state["_left_on_mark"] = left_mark
            try:
                scenario_, tod_ = meter_settings()
                today_rows = detected_history(today_str(), scenario_, owned).tail(1440)
                flags = flag_long_runs(extract_sessions(today_rows, Tariff(rate=float(tariff_rate), tod_enabled=tod_)))
                for text in send_left_on(get_db(), household_id,
                                         household_people(get_db(), household_id, user_display, auth_email),
                                         flags, st.session_state.get("lang", "en"))[:2]:
                    notify("warning", text + ". See the Safety page.")
            except Exception:
                pass
        render_home_tab(tariff_rate, home_details, db=get_db(), household_id=household_id,
                        user_name=st.session_state.auth.get("name") or T("guest_name"),
                        user_email=auth_email, theme=current_theme(),
                        language=st.session_state.get("lang", "en"))

    if page == "overview":
        head(T("live_label") + " &middot; " + when_text, "")
        for note in missing_notes:
            st.warning(note)
        prepare_meter("page")
        render_brief_section(todays_brief(tariff_rate, owned, household_id, user_display, auth_email))
        section("Whole-home meter, next hour and next month")
        pct_change = week_info.get("pct_change", 0)
        if pct_change > 2:
            week_trend = (T("trend_vs_week", pct=f"+{pct_change:.1f}"), "up")
        elif pct_change < -2:
            week_trend = (T("trend_vs_week", pct=f"{pct_change:.1f}"), "down")
        else:
            week_trend = (T("steady"), "neutral")

        c1, c2, c3, c4 = st.columns(4)
        metric_card(c1, label=T("card_current_usage"), value=f"{gap:.2f}", unit="kW",
                    sub=T("card_current_sub"))
        metric_card(c2, label=T("card_next_hour"), value=f"{pred_kw:.2f}", unit="kW",
                    sub=T("card_next_hour_sub", lo=f"{lower:.2f}", hi=f"{upper:.2f}"))
        metric_card(c3, label=T("card_today_cost"),
                    value=f"Rs. {fmt_rs(today_info['total_cost'])}",
                    sub=T("card_kwh_today", kwh=fmt_kwh(today_info['total_kwh'])))
        metric_card(c4, label=T("card_this_week"),
                    value=f"Rs. {fmt_rs(week_info['total_cost'])}",
                    sub="", trend_text=week_trend[0], trend_dir=week_trend[1])

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        chart_col, side_col = st.columns([2.1, 1], gap="medium")
        with chart_col:
            with st.container(border=True):
                section(T("sec_how_much"))
                # The two days leading up to the reading shown in the cards above.
                recent = history.tail(48) if not history.empty else history
                if row_time is not None and not pd.isna(row_time) and not full_data.empty:
                    upto = full_data[full_data["datetime"] <= pd.Timestamp(row_time)]
                    if len(upto) >= 2:
                        recent = upto.tail(48)
                fig_next = go.Figure()
                if not recent.empty:
                    fig_next.add_trace(go.Scatter(
                        x=recent["datetime"], y=recent[TARGET], mode="lines", name=T("legend_power"),
                        line=dict(color=tc("plaster"), width=1.6), fill="tozeroy",
                        fillcolor="rgba(230,233,240,0.05)",
                        hovertemplate="%{x|%a %I %p}<br>%{y:.2f} kW<extra></extra>"))
                    last_t = pd.Timestamp(recent["datetime"].iloc[-1])
                    next_t = last_t + pd.Timedelta(hours=1)
                    fig_next.add_trace(go.Scatter(
                        x=[last_t, next_t], y=[float(recent[TARGET].iloc[-1]), pred_kw], mode="lines",
                        line=dict(color=ACCENT, width=2, dash="dot"), showlegend=False, hoverinfo="skip"))
                    fig_next.add_trace(go.Scatter(
                        x=[next_t], y=[pred_kw], mode="markers", name=T("card_next_hour"),
                        marker=dict(color=ACCENT, size=11, line=dict(color=tc("ink"), width=2)),
                        error_y=dict(type="data", symmetric=False, array=[upper - pred_kw],
                                     arrayminus=[pred_kw - lower], color=ACCENT, thickness=1.5, width=6),
                        hovertemplate="%{x|%a %I %p}<br>%{y:.2f} kW<extra>" + T("card_next_hour") + "</extra>"))
                fig_next.update_layout(**PLOTLY_LAYOUT(
                    height=330, yaxis_title="kW", showlegend=True, margin=dict(l=40, r=20, t=10, b=30),
                    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1)))
                st.plotly_chart(fig_next, width="stretch", key="overview_next_hour")
        with side_col:
            confidence = max(0, min(100, (1 - uncertainty / max(pred_kw, 0.01)) * 100))
            # The stored forecast file is dated by the original recording; the
            # estimate is for the month after the current one.
            next_month_start = local_now().normalize().replace(day=1) + pd.DateOffset(months=1)
            model_name = T("pred_model_hybrid" if pred_info.get("lstm_used") else "pred_model_xgb_only")
            st.markdown(
                f"<div class='ep-stat'><small>{T('card_next_month')}</small>"
                f"<strong>Rs. {fmt_rs(nm_info['total_cost'])}</strong>"
                f"<span>{T('card_projected_for', month=format_localized_month(next_month_start))}</span></div>"
                f"<div class='ep-stat'><small>{T('card_smart_forecast')}</small>"
                f"<strong>{confidence:.0f}%</strong>"
                f"<span>{model_name}. {T('cap_forecast_mae', mae=f'{hybrid_mae:.3f}')}</span></div>",
                unsafe_allow_html=True)
        st.caption(source_text)
        st.markdown(f"""<div class="disc"><p>{T('about_demo')}</p></div>""", unsafe_allow_html=True)

    if page == "appliances":
        head(meter_chip, "sim")
        prepare_meter("page")
        render_appliance_tab(tariff_rate, owned, section, PLOTLY_LAYOUT)

    if page == "analysis":
        head(meter_chip, "sim")
        prepare_meter("page")
        render_analysis_tab(tariff_rate, owned, section, PLOTLY_LAYOUT)

    if page == "save":
        head(meter_chip, "sim")
        prepare_meter("page")
        render_save_tab(tariff_rate, owned, section, PLOTLY_LAYOUT)

    def toolkit_ctx():
        scenario, tod = meter_settings()
        db_ = get_db()
        return {"pred": detected_history(today_str(), scenario, owned), "owned": owned,
                "switched": household_devices(home_details)[1], "tariff": Tariff(rate=float(tariff_rate), tod_enabled=tod),
                "rate": float(tariff_rate), "db": db_, "household_id": household_id, "user_name": user_display,
                "people": household_people(db_, household_id, user_display, auth_email),
                "section": section, "metric_card": metric_card, "layout": PLOTLY_LAYOUT, "ledger": get_ledger(),
                "brief": todays_brief(tariff_rate, owned, household_id, user_display, auth_email)}

    if page in ("goals", "safety", "upgrades"):
        head(meter_chip, "sim")
        prepare_meter("page")
        {"goals": toolkit_ui.render_goals, "safety": toolkit_ui.render_safety,
         "upgrades": toolkit_ui.render_upgrades}[page](toolkit_ctx())

    if page == "help":
        head()
        prepare_meter("page")
        toolkit_ui.render_help(toolkit_ctx())

    if page == "bills":
        head()
        prepare_meter("page")
        load_saved_bills()
        toolkit_ui.render_bill_tools(toolkit_ctx(), st.session_state.get("bills"))
        render_bills_tab()

    if page == "trends":
        head()
        section(T("sec_trends"))
        # Build the plotted window straight from the full remapped dataset so
        # the chart is populated immediately (the replay simulator replays this
        # exact dataset, so its timestamps align — no dependency on how many
        # rows the simulator has reached).
        plot_df = pd.DataFrame()
        if not full_data.empty:
            plot_df = full_data.tail(display_count)[["datetime", "Global_active_power"]].copy()
            plot_df["Global_active_power"] = plot_df["Global_active_power"].astype(float) * sf
            plot_df["datetime"] = pd.to_datetime(plot_df["datetime"])
            # Bin to a clean hourly grid: one point per hour, no duplicate x labels.
            plot_df = (plot_df.assign(hour_bin=plot_df["datetime"].dt.floor("h"))
                              .groupby("hour_bin", as_index=False)["Global_active_power"]
                              .mean()
                              .rename(columns={"hour_bin": "datetime"})
                              .sort_values("datetime")
                              .reset_index(drop=True))
            # Guard against any nulls sneaking in from bad rows.
            plot_df = plot_df.dropna(subset=["datetime", "Global_active_power"])

        if len(plot_df) >= 2:
            sparse = len(plot_df) < 40
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=plot_df["datetime"], y=plot_df["Global_active_power"],
                name=T("legend_power"), mode="lines+markers" if sparse else "lines",
                line=dict(color=ACCENT, width=2, shape="spline", smoothing=0.6),
                marker=dict(size=6, color=ACCENT) if sparse else None,
                fill="tozeroy", fillcolor="rgba(245,168,60,0.08)",
                hovertemplate="%{x|%a, %b %d %I:%M %p}<br>%{y:.2f} kW<extra>" + T("legend_power") + "</extra>",
            ))
            if len(plot_df) >= 5:
                rolling = plot_df["Global_active_power"].rolling(5, min_periods=1).mean()
                fig_trend.add_trace(go.Scatter(
                    x=plot_df["datetime"], y=rolling,
                    name=T("legend_avg"), mode="lines",
                    line=dict(color="#F08A5D", width=2, dash="dash"),
                    hovertemplate="%{x|%a, %b %d %I:%M %p}<br>%{y:.2f} kW<extra>" + T("legend_avg") + "</extra>",
                ))
            time_span = plot_df["datetime"].max() - plot_df["datetime"].min()
            if time_span <= pd.Timedelta(hours=24):
                tick_fmt = "%I %p"          # single day: plain hour labels
            elif time_span <= pd.Timedelta(days=10):
                tick_fmt = "%b %d %I %p"    # crosses midnight: keep date in label (unique)
            else:
                tick_fmt = "%b %d"
            fig_trend.update_layout(**PLOTLY_LAYOUT(
                height=340, xaxis_title=T("x_time"), yaxis_title="kW",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(tickformat=tick_fmt, nticks=10,
                           gridcolor="rgba(0,0,0,0.05)")))
            st.plotly_chart(fig_trend, width="stretch", key="trend_chart")
        else:
            st.markdown(f"""
            <div class="empty-state">
                <div class="empty-state-title">{T('trends_empty_title')}</div>
                <div>{T('trends_empty_msg')}</div>
            </div>
            """, unsafe_allow_html=True)
        section(T("sec_calendar"))
        cal_year_int = int(cal_year)
        cal_month_int = int(cal_month)
        render_calendar(full_data, forecast_df, cal_year_int, cal_month_int, tariff_rate)
        st.markdown(f"""
        <div style="display:flex;flex-wrap:wrap;gap:12px 16px;margin-top:0.8rem;font-size:0.75rem;color:var(--mist);">
            <span><span style="color:var(--ok);">&#9632;</span> {T('cal_legend_low')}</span>
            <span><span style="color:var(--current);">&#9632;</span> {T('cal_legend_avg')}</span>
            <span><span style="color:var(--alert);">&#9632;</span> {T('cal_legend_high')}</span>
            <span><span style="color:var(--mist);">&#9632;</span> {T('cal_legend_future')}</span>
        </div>
        """, unsafe_allow_html=True)

    if page == "family":
        head()
        render_family_tab(get_db(), household_id)
        prepare_meter("page")
        toolkit_ui.render_family_extras(toolkit_ctx())

    if page == "alerts":
        head()
        render_notifications_tab(
            get_db(), household_id, auth_email,
            language=st.session_state.get("lang", "en"),
            tariff_rate=tariff_rate,
            history=history,
            forecast_df=forecast_df,
        )

    if page == "assistant":
        head()
        render_chat_tab(
            get_db(), household_id, auth_email,
            language=st.session_state.get("lang", "en"),
            full_data=full_data,
            scaling_factor=sf,
            tariff_rate=tariff_rate,
        )

    if page == "settings":
        head()
        render_settings_page()


def main():
    apply_theme()
    inject_skin()
    if not render_login_screen():
        motion_runtime("sign-in")
        return
    if not render_onboarding():
        motion_runtime(f"setup-{st.session_state.get('onboard_step', 1)}")
        return
    render_sidebar()
    main_dashboard()
    motion_runtime(st.session_state.get("page", "home"))


if __name__ == "__main__":
    main()
