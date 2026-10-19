"""
EnergyPulse — Home Electricity Dashboard
==================================================
Smart home energy monitoring, predictions, cost tracking,
appliance breakdown, optimization tips, and usage trends.

Launch (from the project folder):
    streamlit run frontend/app.py
"""

import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# The screens live in frontend/, the logic in backend/; make both importable
# whichever folder the app is started from.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _dir in (os.path.join(_ROOT, "frontend"), os.path.join(_ROOT, "backend")):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

import copy
import calendar as cal
import io
import html
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
import streamlit.components.v1 as components

from replay import ReplaySimulator
from appliances import get_appliance_breakdown
from cost import daily_cost, weekly_cost, next_month_cost
from optimize import detect_anomalies, generate_anomaly_tips, calendar_tips, whatif_simulator
from model import load_data
from forecast_features import TARGET
from forecast import Forecaster, month_slice
from ep_log import get_logger

log = get_logger("app")
from data import (
    CLEAN_CSV, remap_to_current_dates, shift_forecast_to_current_dates,
    SAMPLE_META_PATH,
)
from data_source import (
    MAX_UPLOAD_BYTES, POWER_UNITS, DataSource, normalize_uploaded,
    source_from_sample, source_from_upload, source_from_api, fetch_meter_api,
)
from i18n import T, TLIST, LANGS, HOME_TYPES, home_type_label, localized_appliance_category, localized_appliance_label, localized_appliance_type, format_localized_month, format_localized_date
from db import get_db
from features_ui import render_family_tab, render_notifications_tab, render_chat_tab, ensure_login
from appliance_ui import (
    render_appliance_tab, render_analysis_tab, render_save_tab, render_meter_settings, report_body,
)
from meter_source import owned_appliances, SOURCE_NOTE, detected_history, today_str
from appliance_ui import meter_settings
from auth import (Accounts, PASSWORD_RULES, new_guest_email, is_guest_email, session_expired,
                  COOKIE_NAME, REMEMBER_SECONDS, RESET_SECONDS, TERMS_VERSION,
                  validate_signup_form, validate_login_form)
from motion import notify, runtime as motion_runtime, skeleton
from integrity import Ledger, store_bill_file, read_bill_file, bill_path, sign_report, verify_report
from home_ui import render_home_tab, _members as household_people
from daily_brief import build_brief, send_daily_brief
from meter_source import last_days, now_minute
from tariff import Tariff
from ensemble import clean_setting, resolve_weights
from notifications import get_notification_service
from devices import household_devices
from sessions import extract_sessions, flag_long_runs
import toolkit_ui
import catalog
from toolkit import send_left_on
from shell_ui import tc, apply_theme, theme_toggle, current_theme, inject_skin, page_header, sidebar_nav, login_hero, step_list, wordmark, wordmark_mini
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
from paths import DATA_DIR, MODEL_DIR


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
    elif source.kind == "api":
        body = T("src_api_body",
                 host=html.escape(source.label),
                 rows=f"{rows:,}",
                 start=format_localized_date(df["datetime"].min()),
                 end=format_localized_date(df["datetime"].max()),
                 when=(source.report or {}).get("fetched_at", ""))
    else:
        body = T("src_upload_body",
                 name=html.escape(source.label),
                 rows=f"{rows:,}",
                 start=format_localized_date(df["datetime"].min()),
                 end=format_localized_date(df["datetime"].max()))
    return {
        "kind": source.kind,
        "title": (T("src_sample_title") if source.is_sample
                  else T("src_api_title") if source.kind == "api" else T("src_upload_title")),
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


def handle_api_fetch(url, token, power_unit):
    """Fetch readings from a meter API and make them the active data source."""
    frame, reason, host = fetch_meter_api(url, token=token or None)
    if frame is None:
        st.session_state.data_upload_error = T("src_api_fail", reason=reason)
        return False
    canonical, report = normalize_uploaded(frame, power_unit=power_unit)
    if not report.get("ok") or canonical.empty:
        st.session_state.data_upload_error = T("src_api_fail", reason=T("src_need_power"))
        return False
    report["fetched_at"] = local_now().strftime("%d %b %Y, %H:%M")
    st.session_state.data_upload = {"df": canonical,
                                    "source": source_from_api(canonical, host, report=report)}
    st.session_state.data_upload_sig = ("api", url)
    st.session_state.data_upload_error = None
    # The token is kept for this browser session only, so "Refresh" can reuse it.
    st.session_state.data_api = {"url": url, "token": token or ""}
    if "simulator" in st.session_state:
        st.session_state.simulator.reset()
    return True


def render_meter_api_panel(power_unit):
    """Third data source: readings fetched from a smart-meter or utility API."""
    active = st.session_state.get("data_api") or {}
    with st.expander(T("src_api_connect"), expanded=bool(active)):
        url = st.text_input(T("src_api_url"), value=active.get("url", ""),
                            placeholder="https://meter.example.com/api/readings", key="data_api_url")
        token = st.text_input(T("src_api_token"), value=active.get("token", ""),
                              type="password", key="data_api_token")
        st.caption(T("src_api_help"))
        label = T("src_api_refresh") if active and active.get("url") == url else T("src_api_fetch")
        if st.button(label, width="stretch", key="data_api_fetch", disabled=not url.strip()):
            with st.spinner(T("src_api_fetch") + "..."):
                ok = handle_api_fetch(url.strip(), token.strip(), power_unit)
            if ok:
                st.rerun()


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

    render_meter_api_panel(power_unit)

    if st.session_state.get("data_upload_error"):
        st.error(st.session_state.data_upload_error)
    elif st.session_state.get("data_upload") is not None:
        if st.button(T("src_use_sample"), width="stretch", key="sb_back_to_sample"):
            st.session_state.data_upload = None
            st.session_state.data_upload_sig = None
            st.session_state.pop("data_api", None)
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


@st.cache_resource(show_spinner=False)
def load_forecaster():
    """Next-hour and bill forecasters trained by model.py (see forecast.py).
    ENERGYPULSE_DISABLE_LSTM=1 skips TensorFlow on hosts with little memory."""
    no_lstm = os.environ.get("ENERGYPULSE_DISABLE_LSTM", "").strip().lower() in ("1", "true", "yes")
    return Forecaster.load(MODEL_DIR, use_lstm=not no_lstm)


@st.cache_data(show_spinner=False)
def load_forecast():
    """The bundled sample's 62-day forecast, on its real (2010) dates."""
    path = os.path.join(DATA_DIR, "next_month_forecast.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["datetime"])
    return pd.DataFrame()


@st.cache_data(show_spinner=False, max_entries=8)
def _upload_forecast(signature, data):
    return load_forecaster().next_days(data, seasonal=False)


def build_forecast_df(source, raw_data):
    """
    Hourly forecast for the coming 62 days, on the same dates as the readings shown.

    The sample uses its precomputed forecast (household-specific model, real
    dates), moved forward by exactly the shift applied to its readings.  Any
    other data gets a forecast from its own recent use (generic model).
    """
    if raw_data is None or raw_data.empty:
        return pd.DataFrame()
    if source.is_sample:
        fc = load_forecast()
        original_end = (source.original_range or (None, None))[1]
        if fc.empty or not original_end:
            return pd.DataFrame()
        fc = fc.copy()
        shift = pd.to_datetime(raw_data["datetime"]).max() - pd.Timestamp(original_end)
        fc["datetime"] = pd.to_datetime(fc["datetime"]) + shift
        fc["day"] = fc["datetime"].dt.date
        return fc
    data = raw_data[["datetime", TARGET]].copy()
    signature = (source.kind, source.label, len(data), str(data["datetime"].max()))
    try:
        return _upload_forecast(signature, data)
    except Exception as exc:
        log.warning("Bill forecast unavailable: %s", exc)
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
        sim = st.session_state.simulator
        # Save final replay history to DB
        try:
            auth_email = st.session_state.auth.get("email")
            if auth_email:
                primary_user = get_db().get_user(auth_email)
                household_id = primary_user["household_id"] if primary_user else hashlib.md5(
                    (auth_email or "").strip().lower().encode()).hexdigest()
                history = sim.get_full_history()
                if not history.empty:
                    get_db().save_simulation_history(household_id, history)
        except Exception:
            pass
        sim.stop()
    auth_state = st.session_state.get("auth") or {}
    # "Keep me signed in": revoke this browser's token and remove its cookie.
    if auth_state.get("token"):
        try:
            get_accounts().revoke_token(auth_state["token"])
        except Exception as exc:
            log.warning("Token revoke failed: %s", type(exc).__name__)
        st.session_state["_clear_cookie"] = True
    st.session_state["_cookie_checked"] = True       # never sign straight back in after logging out
    # A guest home is not kept: remove everything it stored.
    if auth_state.get("guest") or is_guest_email(auth_state.get("email") or ""):
        try:
            guest_email = auth_state.get("email")
            user = get_db().get_user(guest_email) if guest_email else None
            if user:
                get_db().delete_household(user["household_id"], guest_email)
        except Exception as exc:
            log.warning("Guest cleanup failed: %s", exc)
    for k in ["auth", "home_details", "simulator", "onboard_data", "onboard_step", "sidebar_settings",
              "page", "bills", "bill_seen", "home_state", "nilm_dismissed", "data_upload", "ensemble_setting",
              "data_upload_sig", "data_upload_error", "data_api", "sanctioned_kw"]:
        st.session_state.pop(k, None)


def _write_session_cookie(value: str, max_age: int) -> None:
    """Set (or, with max_age 0, remove) the sign-in cookie in the visitor's browser."""
    components.html(
        "<script>(function(){var w=window.parent;var secure=w.location.protocol==='https:'?'; Secure':'';"
        f"w.document.cookie='{COOKIE_NAME}='+{json.dumps(value)}+'; Max-Age={int(max_age)}; Path=/; SameSite=Strict'+secure;"
        "})();</script>", height=0)


def flush_cookie_changes() -> None:
    """Apply a cookie change requested before the last rerun."""
    token = st.session_state.pop("_set_cookie", None)
    if token:
        _write_session_cookie(token, REMEMBER_SECONDS)
    if st.session_state.pop("_clear_cookie", False):
        _write_session_cookie("", 0)


def restore_remembered_session(auth: dict) -> None:
    """Sign straight in when this browser holds a valid "keep me signed in" token."""
    if st.session_state.get("_cookie_checked"):
        return
    st.session_state["_cookie_checked"] = True
    try:
        token = st.context.cookies.get(COOKIE_NAME)
    except Exception:
        token = None
    if not isinstance(token, str) or not token:
        return
    found = get_accounts().check_token(token)
    if not found:
        _write_session_cookie("", 0)                 # expired or revoked: forget it
        return
    email, name, issued_at = found
    auth.update({"logged_in": True, "email": email, "name": name, "guest": False,
                 "signed_in_at": issued_at, "remember": True, "token": token})
    household = ensure_login(email, name=name)
    saved_home = get_db().get_home_details(household)
    if saved_home:
        st.session_state.home_details = saved_home
    st.rerun()


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


TERMS_TEXT = """
**Terms of Use.** EnergyPulse gives estimates from a simulated meter and from data you upload. Forecasts,
savings and alerts are estimates, not measurements or billing figures, and are offered without warranty.
Keep your password private; you are responsible for what is done with your account.

**Privacy Policy.** We store your name, email, a salted hash of your password (never the password itself),
your home and appliance setup, uploaded bills (encrypted on disk) and the alerts we send you. Your data is used
only to run EnergyPulse for your household and is not sold or shared. Email is sent only to people you add to
your household. You can delete a guest home by logging out; ask the administrator to delete an account.
"""


def render_reset_form(auth: dict) -> None:
    """Forgotten password: email a one-time code, then set a new password with it."""
    st.markdown("**Reset your password**")
    st.caption("We email a one-time code to the address you signed up with. It expires in "
               f"{RESET_SECONDS // 60} minutes.")
    with st.form("reset_request"):
        email = st.text_input("Email address", key="reset_email", max_chars=254)
        if st.form_submit_button("Email me a code", width="stretch"):
            if not email.strip():
                st.error("Enter your email address.")
            else:
                try:
                    with st.spinner("Sending..."):
                        service = get_notification_service()
                        if not service.email_configured():
                            st.error("Password reset needs email, which is not set up on this server. "
                                     "Ask the administrator to reset your password.")
                        else:
                            code = get_accounts().request_reset(email)
                            if code:
                                service.send_reset_code(Accounts.normalise(email), code, RESET_SECONDS // 60)
                            # Same answer whether or not the email has an account.
                            st.success("If that email has an account, a code is on its way. "
                                       "Check your inbox (and spam folder).")
                            auth["reset_email"] = Accounts.normalise(email)
                except Exception as exc:
                    log.error("Password reset request failed: %s", type(exc).__name__)
                    st.error("Password reset is not available just now. Please try again later.")
    with st.form("reset_confirm"):
        email2 = st.text_input("Email address", value=auth.get("reset_email", ""), key="reset_email2", max_chars=254)
        code = st.text_input("Code from the email", key="reset_code", max_chars=20)
        pw = st.text_input("New password", type="password", key="reset_pw", max_chars=256)
        pw2 = st.text_input("Confirm new password", type="password", key="reset_pw2", max_chars=256)
        if st.form_submit_button("Set new password", type="primary", width="stretch"):
            if not (email2.strip() and code.strip() and pw):
                st.error("Fill in the email, the code and the new password.")
            elif pw != pw2:
                st.error("The two passwords do not match. Type the same password in both boxes.")
            else:
                try:
                    ok, msg = get_accounts().reset_password(email2, code, pw)
                except Exception as exc:
                    log.error("Password reset failed: %s", type(exc).__name__)
                    ok, msg = False, "Password reset is not available just now. Please try again later."
                if ok:
                    auth["mode"] = "login"
                    auth.pop("reset_email", None)
                    notify("success", msg)
                    st.rerun()
                st.error(msg)
    if st.button("Back to sign in", key="reset_back", width="stretch"):
        auth["mode"] = "login"
        st.rerun()


def render_login_screen():
    if "auth" not in st.session_state:
        st.session_state.auth = {"logged_in": False, "email": None, "mode": "login"}
    auth = st.session_state.auth
    if auth.get("logged_in"):
        if session_expired(auth.get("signed_in_at"), remember=bool(auth.get("remember"))):
            sign_out()
            st.rerun()
        return True

    restore_remembered_session(auth)

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
        if mode == "reset":
            render_reset_form(auth)
            return False
        with st.form("login_form", clear_on_submit=False):
            if is_signup:
                name = st.text_input("Full name", placeholder=T("ph_name"), key="login_name", max_chars=80)
            email = st.text_input(T("email_label"), placeholder=T("ph_email"), key="login_email", max_chars=254)
            # Streamlit's password box has its own show/hide (eye) button.
            password = st.text_input(T("password_label"), type="password",
                                     placeholder=T("ph_password"), key="login_pw", max_chars=256)
            if is_signup:
                confirm = st.text_input("Confirm password", type="password",
                                        placeholder="Type the password again", key="login_pw2", max_chars=256)
                st.markdown(
                    "<div class='ep-pw-rules'><b>Your password must have</b><ul>"
                    + "".join(f"<li>{label}</li>" for _, label in PASSWORD_RULES)
                    + "</ul></div>", unsafe_allow_html=True)
                terms = st.checkbox("I accept the Terms of Use and Privacy Policy", key="login_terms")
            remember = st.checkbox(T("remember_me"), value=True, key="login_remember")
            btn_label = T("btn_create_account") if is_signup else T("btn_sign_in")
            submitted = st.form_submit_button(btn_label, width="stretch", type="primary")
            if submitted:
                problems = (validate_signup_form(name, email, password, confirm, terms) if is_signup
                            else validate_login_form(email, password))
                if problems:
                    for p in problems:
                        st.error(p)
                    return False
                try:
                    with st.spinner("Creating your account..." if is_signup else "Signing in..."):
                        accounts = get_accounts()
                        ok, result = (accounts.sign_up(email, password, name, terms_version=TERMS_VERSION)
                                      if is_signup else accounts.sign_in(email, password))
                except Exception as exc:
                    log.error("Sign-in service failed: %s", type(exc).__name__)
                    st.error("Signing in is not available just now. Please try again in a minute.")
                    return False
                if not ok:
                    st.error(result)
                    return False
                email = accounts.normalise(email)
                auth.update({"logged_in": True, "email": email, "name": result,
                             "guest": False, "signed_in_at": time.time(), "remember": False})
                if remember:
                    token = accounts.issue_token(email)
                    auth.update({"remember": True, "token": token})
                    st.session_state["_set_cookie"] = token
                household = ensure_login(email, name=result)
                saved_home = None if is_signup else get_db().get_home_details(household)
                if saved_home:
                    st.session_state.home_details = saved_home      # straight to their own home
                notify("success", f"Account created. Welcome, {result}." if is_signup else f"Signed in as {result}.")
                st.rerun()
        if is_signup:
            with st.expander("Terms of Use and Privacy Policy"):
                st.markdown(TERMS_TEXT)
        else:
            if st.button("Forgot password?", key="forgot_pw", type="tertiary"):
                auth["mode"] = "reset"
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
    """
    Handle onboarding for both new and returning users.
    Returns True when user has a valid home setup and can proceed to dashboard.
    """
    # If already have home details in session, we're good to go
    if st.session_state.get("home_details") is not None:
        return True
    
    init_language()
    auth = st.session_state.auth
    household_id = current_household_id()
    db = get_db()
    user_name = auth.get("name", "there")
    is_guest = auth.get("guest", False)
    
    # For guests, always show onboarding (they don't have persistent data)
    if is_guest:
        return _render_new_user_onboarding(user_name)
    
    # For authenticated users, check if they have completed onboarding
    saved_home = db.get_home_details(household_id)
    onboarding_done = db.is_onboarding_completed(household_id)
    
    if onboarding_done and saved_home:
        # Returning user with completed onboarding - show welcome screen
        return _render_returning_user_welcome(saved_home, user_name, household_id, db)
    elif saved_home:
        # User has saved home but hasn't completed onboarding (edge case)
        # Pre-populate and continue
        st.session_state.home_details = saved_home
        return True
    else:
        # New user - show full onboarding
        return _render_new_user_onboarding(user_name)


def _render_returning_user_welcome(saved_home: dict, user_name: str, household_id: str, db) -> bool:
    """Show welcome screen for returning users with Continue/Edit options."""
    init_language()
    
    # Check if user wants to edit
    if st.session_state.get("edit_mode"):
        return _render_edit_onboarding(saved_home, user_name, household_id, db)
    
    # Prepare appliance summary
    appliances = saved_home.get("appliances", [])
    appliance_names = []
    for a in appliances:
        if isinstance(a, dict):
            name = a.get("name", a.get("type_key", "Unknown"))
            type_key = a.get("type_key", a.get("type", ""))
            if type_key:
                t = catalog.get_type(type_key)
                if t:
                    name = t.name
            appliance_names.append(name)
        else:
            appliance_names.append(str(a))
    
    home_type = home_type_label(saved_home.get("home_type", "Apartment"))
    occupants = saved_home.get("occupants", 0)
    tariff_rate = saved_home.get("tariff_rate", 8.0)
    city = saved_home.get("city", "")
    home_size = saved_home.get("home_size", 0)
    
    side, main_col = st.columns([1, 1.75], gap="large")
    with side:
        st.markdown(
            wordmark()
            + f"<div class='ep-head' style='margin-top:2.6rem'><h1>{T('welcome_back', name=user_name)}</h1>"
              f"<p>{T('welcome_back_msg')}</p></div>", unsafe_allow_html=True)
        st.markdown("")
        render_language_selector(key="lang_select_welcome")
    
    with main_col:
        with st.container(key="onboard_panel"):
            # Summary card
            # Built outside the f-string: Python 3.11 allows no backslash inside {}.
            dot = " \u00b7 "
            city_part = f"{dot}{html.escape(str(city))}" if city else ""
            size_part = f"{dot}{T('sqft_suffix', n=home_size)}" if home_size else ""
            st.markdown(f"""
            <div class='ep-brief' style='margin-bottom:1.5rem'>
                <b>{T('your_current_setup')}</b><br>
                <b>{home_type}</b>{dot}{T('people_suffix', n=occupants)}{dot}{T('rate_suffix', rate=tariff_rate)}
                {city_part}
                {size_part}
            </div>
            """, unsafe_allow_html=True)
            
            if appliance_names:
                st.markdown(f"**{T('your_appliances')}** ({len(appliance_names)})")
                for i, name in enumerate(appliance_names[:10], 1):
                    st.markdown(f"  {i}. {name}")
                if len(appliance_names) > 10:
                    st.caption(f"... and {len(appliance_names) - 10} more")
            
            st.markdown("")
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                if st.button(T("btn_continue_saved"), width="stretch", type="primary", key="btn_continue"):
                    st.session_state.home_details = saved_home
                    notify("success", T("loaded_saved_setup"))
                    st.rerun()
            with c2:
                if st.button(T("btn_edit_setup"), width="stretch", key="btn_edit"):
                    st.session_state.edit_mode = True
                    st.session_state.onboard_data = _prepare_edit_data(saved_home)
                    st.session_state.onboard_step = 1
                    st.rerun()
    
    return False


def _prepare_edit_data(saved_home: dict) -> dict:
    """Prepare onboarding data from saved home details for editing."""
    appliances = saved_home.get("appliances", [])
    # Convert appliances to the format expected by onboarding
    normalized_appliances = []
    for a in appliances:
        if isinstance(a, dict):
            normalized_appliances.append(a)
        else:
            normalized_appliances.append({"name": str(a), "type": "Other", "usage": "Medium"})
    
    return {
        "home_type": saved_home.get("home_type", "Apartment"),
        "occupants": saved_home.get("occupants", 1),
        "appliances": normalized_appliances,
        "tariff_rate": saved_home.get("tariff_rate", 8.0),
        "city": saved_home.get("city", ""),
        "home_size": saved_home.get("home_size", 0) or 0,
        "timezone": saved_home.get("timezone") or user_tz(),
    }


def _render_edit_onboarding(saved_home: dict, user_name: str, household_id: str, db) -> bool:
    """Render onboarding flow pre-populated with existing data for editing."""
    init_language()
    
    if "onboard_step" not in st.session_state:
        st.session_state.onboard_step = 1
    if "onboard_data" not in st.session_state:
        st.session_state.onboard_data = _prepare_edit_data(saved_home)
    
    step = st.session_state.onboard_step
    od = st.session_state.onboard_data
    
    step_labels = [T("step_home_info"), T("step_appliances"), T("step_rate")]
    side, main_col = st.columns([1, 1.75], gap="large")
    with side:
        st.markdown(
            wordmark()
            + f"<div class='ep-head' style='margin-top:2.6rem'><h1>{T('edit_your_setup')}</h1>"
              f"<p>{T('edit_setup_msg')}</p></div>"
            + step_list(step_labels, step), unsafe_allow_html=True)
        st.markdown("")
        render_language_selector(key="lang_select_onboard_edit")
    
    with main_col:
        with st.container(key="onboard_panel"):
            if step == 1:
                _render_onboard_step1(od)
            elif step == 2:
                _render_onboard_step2(od)
            elif step == 3:
                _render_onboard_step3_edit(od, household_id, db, saved_home)
    
    return False


def _render_onboard_step3_edit(od, household_id: str, db, saved_home: dict):
    """Step 3 for edit mode - shows save/cancel instead of finish."""
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
    
    st.markdown("")
    c_back, c_cancel, c_save = st.columns([1, 1, 1])
    with c_back:
        if st.button(T("btn_back"), width="stretch", key="ob_back3_edit"):
            st.session_state.onboard_step = 2
            st.rerun()
    with c_cancel:
        if st.button(T("btn_cancel"), width="stretch", key="ob_cancel_edit"):
            if _has_unsaved_changes(od, saved_home):
                st.session_state.confirm_cancel = True
                st.rerun()
            else:
                st.session_state.edit_mode = False
                st.session_state.pop("onboard_data", None)
                st.session_state.pop("onboard_step", None)
                st.rerun()
    with c_save:
        if st.button(T("btn_save_changes"), width="stretch", type="primary", key="ob_save_edit"):
            _save_edited_setup(od, household_id, db)
            st.session_state.edit_mode = False
            st.session_state.pop("onboard_data", None)
            st.session_state.pop("onboard_step", None)
            notify("success", T("setup_saved"))
            st.rerun()
    
    # Confirm cancel dialog
    if st.session_state.get("confirm_cancel"):
        st.warning(T("confirm_discard_changes"))
        cc1, cc2 = st.columns(2)
        with cc1:
            if st.button(T("yes_discard"), width="stretch", type="primary", key="confirm_discard_yes"):
                st.session_state.edit_mode = False
                st.session_state.confirm_cancel = False
                st.session_state.pop("onboard_data", None)
                st.session_state.pop("onboard_step", None)
                st.rerun()
        with cc2:
            if st.button(T("no_keep_editing"), width="stretch", key="confirm_discard_no"):
                st.session_state.confirm_cancel = False
                st.rerun()


def _has_unsaved_changes(od: dict, saved_home: dict) -> bool:
    """Check if there are unsaved changes compared to saved home."""
    # Compare key fields
    if od.get("home_type") != saved_home.get("home_type"):
        return True
    if od.get("occupants") != saved_home.get("occupants"):
        return True
    if od.get("tariff_rate") != saved_home.get("tariff_rate"):
        return True
    if od.get("city") != saved_home.get("city"):
        return True
    if od.get("home_size") != (saved_home.get("home_size") or 0):
        return True
    # Compare appliances (simplified)
    saved_appliances = saved_home.get("appliances", [])
    if len(od.get("appliances", [])) != len(saved_appliances):
        return True
    return False


def _save_edited_setup(od: dict, household_id: str, db):
    """Save edited home setup to database."""
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
        db.save_home_details(household_id, st.session_state.home_details)
    except Exception:
        pass
    st.session_state.pop("home_state", None)


def _render_new_user_onboarding(user_name: str) -> bool:
    """Original onboarding flow for new users."""
    if "onboard_step" not in st.session_state:
        st.session_state.onboard_step = 1
    if "onboard_data" not in st.session_state:
        st.session_state.onboard_data = {
            "home_type": "Apartment", "occupants": 4, "appliances": [],
            "tariff_rate": 8.0, "city": "", "home_size": 0,
        }
    step = st.session_state.onboard_step
    od = st.session_state.onboard_data
    
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
                _render_onboard_step3_new(od)
    return False


def _render_onboard_step3_new(od):
    """Step 3 for new users - original finish button."""
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
                get_db().mark_onboarding_completed(current_household_id())
                saved = True
            except Exception:
                log.exception("Saving the home setup failed")
                saved = False
            st.session_state.pop("home_state", None)
            for k in ["onboard_data", "onboard_step"]:
                if k in st.session_state:
                    del st.session_state[k]
            if saved:
                notify("success", "Home saved. Every appliance you listed is now in the 3D home.")
            else:
                notify("warning", "Your home could not be saved just now. It works for this visit; "
                                  "save it again from Edit appliances so it is remembered.")
            st.rerun()


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


from appliance_manager import spec_table, spec_editor, render_page as render_my_appliances


def _render_onboard_step2(od):
    section(T("ob_your_appliances"))
    st.markdown('<div style="color:var(--mist);font-size:0.84rem;margin-bottom:0.6rem;">'
                'Pick the appliances in your home. Each one comes with its specification, and you can choose the '
                'model class that matches yours. Only what you pick is shown in your 3D home.</div>',
                unsafe_allow_html=True)
    items = od.setdefault("appliances", [])
    tabs = st.tabs(catalog.CATEGORIES + ["Something else"])
    for tab, category in zip(tabs, catalog.CATEGORIES):
        with tab:
            types = catalog.types_in(category)
            names = [t.name for t in types]
            have = [t.name for t in types if any(i.get("type_key") == t.key for i in items if isinstance(i, dict))]
            key = f"ob_pick_{category}"
            if key not in st.session_state:
                st.session_state[key] = have
            picked = st.pills(category, names, selection_mode="multi", key=key, label_visibility="collapsed")
            picked = set(picked or [])
            changed = False
            for t in types:
                exists = any(isinstance(i, dict) and i.get("type_key") == t.key for i in items)
                if t.name in picked and not exists:
                    items.append(catalog.item_for(t.key))
                    changed = True
                elif t.name not in picked and exists:
                    od["appliances"] = items = [i for i in items if not (isinstance(i, dict) and i.get("type_key") == t.key)]
                    changed = True
            if changed:
                st.rerun()
    with tabs[-1]:
        st.caption("Not in the list? Add it by name. It is placed in your home and shown from its switch.")
        c1, c2, c3 = st.columns([2, 2, 1], vertical_alignment="bottom")
        custom = c1.text_input(T("name_label"), placeholder="For example: sewing machine", key="ob_app_custom")
        type_options = ["Cooling", "Heating", "Kitchen", "Laundry", "Electronics", "Lighting", "Other"]
        kind = c2.selectbox(T("category_label"), type_options, index=type_options.index("Other"), key="ob_app_type_custom")
        if c3.button(T("btn_add"), key="ob_add_app", type="primary", disabled=not custom.strip()):
            items.append({"name": custom.strip()[:40], "type": kind, "usage": "Medium", "icon": "", "spec_mode": "own", "own": {}})
            st.rerun()

    if items:
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        section(f"Your list ({len(items)})")
    for i, item in enumerate(list(items)):
        if not isinstance(item, dict):
            item = {"name": str(item), "type": "Other", "usage": "Medium"}
            items[i] = item
        t = catalog.get_type(item.get("type_key", ""))
        with st.container(border=True):
            cols = st.columns([1.6, 1.6, 0.9], vertical_alignment="center")
            cols[0].markdown(f'<div class="ep-row-name">{html.escape(str(item["name"]))}</div>'
                             f'<div class="ep-row-kind">{html.escape(str(item.get("type", "")))}</div>',
                             unsafe_allow_html=True)
            item["usage"] = cols[1].select_slider(T("typical_usage"), options=["Low", "Medium", "High"],
                                                  value=item.get("usage", "Medium"), key=f"app_usage_{i}",
                                                  label_visibility="collapsed")
            act = cols[2]
            if t and act.button("Add another", key=f"ob_more_{i}"):
                items.insert(i + 1, catalog.item_for(t.key, item.get("model_id"), item.get("usage", "Medium")))
                st.rerun()
            if act.button("Remove", key=f"app_del_{i}"):
                items.pop(i)
                if t and not any(isinstance(x, dict) and x.get("type_key") == t.key for x in items):
                    key = f"ob_pick_{t.category}"
                    st.session_state[key] = [n for n in st.session_state.get(key, []) if n != t.name]
                st.rerun()
            if spec_editor(item, current_household_id(), f"ob_{i}_{item.get('type_key', 'x')}"):
                st.rerun()

    st.markdown("")
    if not items:
        st.caption("Pick at least one appliance to continue. Your 3D home shows only what you pick here.")
    c_back, c_next = st.columns([1, 1])
    with c_back:
        if st.button(T("btn_back"), width="stretch", key="ob_back2"):
            st.session_state.onboard_step = 1
            st.rerun()
    with c_next:
        if st.button(T("btn_next_rate"), width="stretch", type="primary", key="ob_next2",
                     disabled=not items):
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
                saved = True
            except Exception:
                log.exception("Saving the home setup failed")
                saved = False   # the session copy still works; it is just not remembered
            st.session_state.pop("home_state", None)        # rebuild the 3D home from the new list
            for k in ["onboard_data", "onboard_step"]:
                if k in st.session_state:
                    del st.session_state[k]
            if saved:
                notify("success", "Home saved. Every appliance you listed is now in the 3D home.")
            else:
                notify("warning", "Your home could not be saved just now. It works for this visit; "
                                  "save it again from Edit appliances so it is remembered.")
            st.rerun()


@st.cache_data(show_spinner=False)
def _brief_cached(date, scenario, owned, rate, tod):
    """Today's per-appliance forecast and savings plan (see daily_brief.py)."""
    from daily_brief import make_brief
    return make_brief(date, scenario, owned, rate, tod)


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


@st.cache_data(show_spinner=False, max_entries=8)
def _now_cached(date, scenario, owned, rate, tod, minute):
    from daily_brief import right_now
    return right_now(detected_history(date, scenario, owned), owned, Tariff(rate=rate, tod_enabled=tod), minute)


@st.cache_data(show_spinner=False)
def _tomorrow_cached(date, scenario, owned, rate, tod):
    from daily_brief import tomorrow_outlook
    from meter_source import scenario_place, temperatures
    from weather import forecast_high
    brief = _brief_cached(date, scenario, owned, rate, tod) or {}
    temps = temperatures(date, scenario)
    nxt = pd.Timestamp(date) + pd.Timedelta(days=1)
    return tomorrow_outlook(detected_history(date, scenario, owned), owned, Tariff(rate=rate, tod_enabled=tod), brief,
                            temps=temps, temp_tomorrow=forecast_high(nxt, scenario_place(scenario)[0]))


def render_tomorrow_section(tariff_rate, owned):
    """What tomorrow is expected to need, and the plan for it."""
    if not owned:
        return
    scenario, tod = meter_settings()
    t = _tomorrow_cached(today_str(), scenario, tuple(owned), float(tariff_rate), tod)
    section(f"Tomorrow, {t['date_label']}")
    left, right = st.columns([1, 1.3], gap="medium")
    with left:
        c1, c2 = st.columns(2)
        metric_card(c1, label="Expected tomorrow", value=f"{t['total_kwh']:.1f}", unit="units",
                    sub=f"about Rs. {t['total_cost']:.0f}")
        metric_card(c2, label="Forecast high", value=f"{t['temp']:.0f}" if t["temp"] is not None else "-", unit="C",
                    sub="simulated weather")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        _rows_html = "".join(f"<tr><td class='n'>{a['name']}</td><td class='v'>{a['kwh']:.2f} units<small>Rs. {a['cost']:.1f}</small></td></tr>"
                             for a in t["appliances"])
        st.markdown(f"<table class='ep-rows'>{_rows_html}</table>", unsafe_allow_html=True)
    with right:
        with st.container(border=True):
            section("Plan for tomorrow")
            if t["plan"]:
                st.markdown("".join(f"<div class='ep-plan-item'><i>{n}</i><b style='grid-column:2/4;font-weight:500'>{line}</b></div>"
                                    for n, line in enumerate(t["plan"], 1)), unsafe_allow_html=True)
            else:
                st.caption("Nothing to plan: no change would save more than the small-saving limit.")
            st.caption("Built from your usual times on the meter, tomorrow's weekday and forecast temperature, "
                       "and the changes worth making.")
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=16)
def _meter_forecast_cached(date, scenario, owned, hour):
    """
    The next-hour model on the simulated meter, run once per clock hour: forecasts
    for the hour in progress and the next one, from the finished hours before them.
    """
    from forecast import meter_upcoming
    from sim_dataset import minute_to_hourly
    pred = detected_history(date, scenario, owned)
    cutoff = pd.Timestamp(date) + pd.Timedelta(hours=int(hour))
    hourly = minute_to_hourly(pred[pd.to_datetime(pred["datetime"]) < cutoff]).dropna()
    from meter_source import meter_context
    upcoming = meter_upcoming(load_forecaster(), hourly, steps=2, context=meter_context(scenario, owned))
    log.info("Meter forecast for %s %02d:00 from %d finished hours: %s", date, int(hour), len(hourly),
             [round(float(v), 3) for v in upcoming.get("predicted_kw", [])])
    return upcoming


def _meter_now_ts(date) -> pd.Timestamp:
    return pd.Timestamp(date) + pd.Timedelta(minutes=now_minute())


def render_freshness(date, upcoming, monitor):
    """When each stage of the live pipeline last ran, with a warning for any that is out of date."""
    from readings import freshness
    now = _meter_now_ts(date)
    brief_date = (st.session_state.get("_brief_mark") or (None, None))[1]
    stamps = {
        "measurement": now.floor("min"),
        "prediction": now.floor("h") if upcoming is not None and not upcoming.empty else None,
        "optimization": pd.Timestamp(brief_date) if brief_date else None,
        "alert_check": (monitor or {}).get("checked_at"),
    }
    f = freshness(stamps, now)
    labels = {"measurement": "Reading", "prediction": "Forecast", "optimization": "Savings plan",
              "alert_check": "Alert check"}
    parts = []
    for key, label in labels.items():
        at = f[key]["at"]
        parts.append(f"{label}: " + (f"{at:%d %b %H:%M}" if at is not None else "not yet")
                     + (" (out of date)" if f[key]["stale"] and at is not None else ""))
    st.caption("Last updated · " + " · ".join(parts))
    stale = [labels[k] for k, v in f.items() if v["stale"] and v["at"] is not None]
    if stale:
        st.warning(", ".join(stale) + " is out of date. The figures shown are from the time given above.")
    if (monitor or {}).get("error"):
        st.warning("The alert check could not run this hour; it will try again. The dashboard is otherwise unaffected.")


def run_meter_monitoring(household_id, owned, people, scenario, date):
    """Log the meter forecasts, record finished hours and check for overconsumption (once per hour)."""
    hour = now_minute() // 60
    mark = (household_id, date, scenario, tuple(owned), hour)
    if st.session_state.get("_monitor_mark") == mark:
        return st.session_state.get("_monitor_result")
    try:
        import monitoring
        from meter_source import meter_context
        res = monitoring.run_check(get_db(), load_forecaster(), household_id, detected_history(date, scenario, owned),
                                   _meter_now_ts(date), owned, people=people, service=get_notification_service(),
                                   language=st.session_state.get("lang", "en"),
                                   context=meter_context(scenario, owned))
        log.info("Monitoring %s: logged %d forecasts, filled %d hours, status %s%s", household_id, res["logged"],
                 res["filled"], res["result"]["status"], ", ALERT" if res["alert"] else "")
    except Exception:
        log.exception("Monitoring failed for %s", household_id)
        res = {"error": True}
    st.session_state["_monitor_mark"], st.session_state["_monitor_result"] = mark, res
    if res.get("alert"):
        notify("warning", res["alert"]["subject"] + ". Details are on the Dashboard.")
    return res


def render_overuse_banner(household_id):
    """The latest overconsumption alert of the last 24 hours, with its figures."""
    try:
        import monitoring
        entry = monitoring.latest_alert(get_db(), household_id)
        state = get_db().get_store(household_id, monitoring.STATE_KEY) or {}
    except Exception:
        log.exception("Could not read alerts for %s", household_id)
        return
    if not entry:
        return
    sent = pd.Timestamp(entry.get("sent_at")) if entry.get("sent_at") else None
    if sent is not None and pd.Timestamp.utcnow().tz_localize(None) - sent > pd.Timedelta(hours=24):
        return
    ongoing = bool(state.get("open"))
    delivered = {"sent": "Emailed to the household.", "failed": "Email could not be sent; shown here.",
                 "in-app": "Shown here (email is not set up)."}.get(entry.get("status"), "")
    box = st.warning if ongoing else st.info
    box(f"**{entry.get('subject', 'High energy use')}{'' if ongoing else ' (back to normal now)'}**  \n"
        f"{entry.get('body_text', '')}  \n<small>{delivered}</small>", icon="⚡" if ongoing else "ℹ️")


def render_now_section(tariff_rate, owned, household_id=None, people=None):
    """
    The simulated meter, refreshed at three speeds:
      every second  the live tile (clock and the reading for this minute; cached, no model)
      every minute  today's chart and cost so far
      every hour    the next-hour forecast and the overconsumption check (and a full refresh)
    """
    scenario, tod = meter_settings()
    date = today_str()
    hour = now_minute() // 60
    monitor = None
    if household_id:
        monitor = run_meter_monitoring(household_id, tuple(owned), people or [], scenario, date)
        render_overuse_banner(household_id)
    try:
        upcoming = _meter_forecast_cached(date, scenario, tuple(owned), hour)
    except Exception:
        log.exception("Meter forecast failed")
        upcoming = pd.DataFrame()
    fc_meta = load_forecaster().meta.get("simulated_meter", {})

    @st.fragment(run_every=1)
    def live_tile():
        if now_minute() // 60 != hour or today_str() != date:
            st.rerun(scope="app")                         # a new hour: new forecast and alert check
        minute = now_minute()
        now = _now_cached(today_str(), scenario, tuple(owned), float(tariff_rate), tod, minute)
        section(f"Right now, {local_now():%a %d %b, %H:%M:%S}")
        c1, c2, c3, c4 = st.columns(4)
        metric_card(c1, label="Load now", value=f"{now['kw']:.2f}", unit="kW",
                    sub=(("running: " + ", ".join(now["running"])) if now["running"] else "no large appliance running")
                    + f" · reading for {minute // 60:02d}:{minute % 60:02d}")
        if not upcoming.empty:
            nxt = upcoming.iloc[-1]
            t0 = pd.Timestamp(nxt["target_hour"])
            band = (f"90% range {nxt['lower_kw']:.2f}-{nxt['upper_kw']:.2f}" if pd.notna(nxt.get("lower_kw")) else "")
            metric_card(c2, label=f"Forecast {t0:%H:%M}-{t0 + pd.Timedelta(hours=1):%H:%M}",
                        value=f"{nxt['predicted_kw']:.2f}", unit="kWh",
                        sub=f"{band} · made at {hour:02d}:00 · usual {now['next_kw']:.2f}")
        else:
            metric_card(c2, label="Next hour, usually", value=f"{now['next_kw']:.2f}", unit="kW",
                        sub=f"between {now['next_lo']:.2f} and {now['next_hi']:.2f} on the last {now['next_days']} days")
        metric_card(c3, label="Today so far", value=f"Rs. {now['cost']:.0f}",
                    sub=f"{now['kwh']:.1f} units; {now['period'].lower()} rate Rs. {now['rate']:.2f}")
        change = now["week_change"]
        metric_card(c4, label="Last 7 days", value=f"Rs. {now['week_cost']:.0f}",
                    sub=f"{now['week_kwh']:.0f} units",
                    trend_text="" if change is None else f"{change:+.0f}% on the week before",
                    trend_dir="neutral" if change is None or abs(change) < 2 else ("up" if change > 0 else "down"))

    live_tile()
    if upcoming.empty:
        st.info("The next-hour forecast is not available right now"
                + (f" ({load_forecaster().sim_error})." if getattr(load_forecaster(), "sim_error", None) else ".")
                + " The live reading still updates; no forecast figure is shown rather than a made-up one.")
    if not upcoming.empty and fc_meta:
        comp = fc_meta.get("comparison", {})
        model_name = {"lightgbm": "LightGBM", "xgboost": "XGBoost"}.get(fc_meta.get("model", "xgboost"), "machine-learning")
        chosen = comp.get("chosen") or comp.get("xgboost") or {}
        st.caption(
            f"Forecast by a {model_name} model trained on {fc_meta.get('homes', {}).get('train', 0)} simulated homes and tested on "
            f"{fc_meta.get('homes', {}).get('test', 0)} others it never saw: average error "
            f"{chosen.get('mae_kw', 0):.2f} kWh per hour, "
            f"{fc_meta.get('improvement_over_best_baseline_pct', 0):.0f}% lower than the best simple baseline. "
            "The live tile refreshes every second; the forecast and the alert check run once an hour.")
    render_freshness(date, upcoming, monitor)
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    @st.fragment(run_every=60)
    def today_chart():
        minute = now_minute()
        now = _now_cached(today_str(), scenario, tuple(owned), float(tariff_rate), tod, minute)
        with st.container(border=True):
            section("Today on the meter")
            base_day = pd.Timestamp(today_str())
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[base_day + pd.Timedelta(minutes=i) for i in range(0, len(now["typical"]), 5)],
                y=now["typical"][::5], mode="lines", name="A usual day (average of the last month)",
                line=dict(color=tc("faint"), width=1.2, dash="dot"),
                hovertemplate="%{x|%H:%M}<br>%{y:.2f} kW usually<extra></extra>"))
            fig.add_trace(go.Scatter(
                x=[base_day + pd.Timedelta(minutes=i) for i in range(len(now["trace"]))], y=now["trace"],
                mode="lines", name="Today so far", line=dict(color=ACCENT, width=1.8),
                fill="tozeroy", fillcolor="rgba(245,168,60,0.10)",
                hovertemplate="%{x|%H:%M}<br>%{y:.2f} kW<extra></extra>"))
            if not upcoming.empty:
                fig.add_trace(go.Scatter(
                    x=[pd.Timestamp(t) + pd.Timedelta(minutes=30) for t in upcoming["target_hour"]],
                    y=upcoming["predicted_kw"], mode="markers", name="Forecast (hourly average)",
                    marker=dict(color=ACCENT, size=9, symbol="diamond"),
                    error_y=dict(type="data", symmetric=False,
                                 array=(upcoming["upper_kw"] - upcoming["predicted_kw"]).fillna(0),
                                 arrayminus=(upcoming["predicted_kw"] - upcoming["lower_kw"]).fillna(0),
                                 color=ACCENT, thickness=1.2, width=5),
                    hovertemplate="%{x|%H:00}<br>%{y:.2f} kW forecast<extra></extra>"))
            fig.update_layout(**PLOTLY_LAYOUT(
                height=300, yaxis_title="kW", showlegend=True, margin=dict(l=40, r=20, t=10, b=30),
                xaxis=dict(range=[base_day, base_day + pd.Timedelta(days=1)], tickformat="%H:%M"),
                legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1)))
            st.plotly_chart(fig, width="stretch", key="overview_today")
            st.caption(f"Simulated main-meter reading up to {minute // 60:02d}:{minute % 60:02d}. The rest of today "
                       "is not known yet, so it is not drawn; the diamonds are the model's forecast. "
                       "This chart updates every minute.")

    today_chart()
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)


def _meter_answer(question, ctx):
    import meter_chat
    return meter_chat.answer(question, toolkit_ui.base(ctx), ctx.get("brief"), ctx["owned"], ctx["switched"])


def render_brief_strip(brief):
    """One line above the 3D home."""
    if not brief:
        return
    top = brief["actions"][0] if brief["actions"] else None
    st.markdown(
        f"<div class='ep-brief'><span><b>Today's forecast</b> &nbsp;<strong>{brief['total_kwh']:.1f}</strong> units, "
        f"about <strong>Rs. {brief['total_cost']:.0f}</strong></span>"
        + (f"<span>high of <strong>{brief['temp_today']:.0f}&deg;C</strong> expected</span>" if brief.get("temp_today") is not None else "")
        + f"<span>off by {brief['error_pct']:.1f}% on recent days</span>"
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
    if brief.get("temp_today") is not None:
        st.caption(f"Weather: a high of {brief['temp_today']:.0f} C is expected today"
                   + (f" in {brief['city']}" if brief.get("city") else "")
                   + (f"; yesterday reached {brief['temp_yesterday']:.0f} C" if brief.get("temp_yesterday") is not None else "")
                   + ". The weather is simulated for your city, and the air-conditioner forecast follows the temperature.")
    st.caption(f"Forecast rule: the average of the last 7 days and of the same weekday in the last 4 weeks, from "
               f"{brief['history_days']} days of main-meter history. This notification is sent once a day and is "
               f"kept on the Notifications page.")
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)


PAGES = [
    # Core Dashboard
    ("dashboard", "tab_dashboard"),
    # Appliance Intelligence
    ("appliances", "tab_appliances"),
    # Optimization & Savings
    ("optimize", "tab_optimize"),
    # Bills & Safety
    ("bills_safety", "tab_bills_safety"),
    # Household
    ("household", "tab_household"),
    None,
    # Settings & Help
    ("settings", "tab_settings"),
    ("help", "tab_help"),
]

# Page titles are written for English; other languages use the page's own name.
PAGE_TITLES = {
    "dashboard": ("Dashboard", "Real-time meter, next-hour forecast, and usage trends at a glance."),
    "appliances": ("Appliances", "Your devices, what the meter detects, and where every watt goes."),
    "optimize": ("Optimize", "Savings tips, monthly goals, and upgrade calculator from your own data."),
    "bills_safety": ("Bills & Safety", "Past bills, fault alerts, and overload warnings in one place."),
    "household": ("Household", "Family alerts, languages, and your energy assistant."),
    "settings": ("Settings", "Your home, tariff, and data source."),
    "help": ("Help", "Log power cuts, draft complaints, and get support."),
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
    st.session_state.setdefault("page", "dashboard")
    if "sidebar_settings" not in st.session_state:
        st.session_state.sidebar_settings = _default_settings(hd)
    with st.sidebar:
        st.markdown(wordmark_mini(), unsafe_allow_html=True)
        sidebar_nav(PAGES, st.session_state.page)
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


_APPLIANCE_WIDGETS = ("ap_qty_", "ap_hrs_", "ap_use_")


def _reset_appliance_widgets() -> None:
    """Forget the editor's input values.  The inputs are keyed by list position, so after a
    delete, cancel or save they must be rebuilt from the list, not from what was typed before."""
    for key in [k for k in st.session_state.keys() if isinstance(k, str) and k.startswith(_APPLIANCE_WIDGETS)]:
        st.session_state.pop(key, None)
    # The browser also remembers a widget's value by its key; new keys make it start afresh.
    st.session_state["appl_widget_ver"] = st.session_state.get("appl_widget_ver", 0) + 1


def _appliances_for_compare(items) -> str:
    """The list as it would be saved, so an untouched list never counts as changed."""
    from home_config import validate_home_details
    try:
        clean = validate_home_details({"appliances": list(items or [])})[0]["appliances"]
    except Exception:
        clean = items
    return json.dumps(clean, sort_keys=True, default=str)


def _render_appliances_editor(hd: dict, lang: str, closable: bool = False) -> None:
    """
    Edit the household's appliances: add, remove, quantity, hours a day, usage.
    Changes go into a draft; "Save changes" validates and writes them to the
    database (the authoritative copy) and "Cancel" drops the draft.
    """
    import catalog
    from home_config import InvalidSetup, typical_hours, validate_home_details
    db = get_db()
    household_id = current_household_id()
    saved_items = hd.get("appliances", []) or []
    draft = st.session_state.get("appl_draft")
    if draft is None or st.session_state.get("appl_draft_for") != household_id:
        draft = [dict(i) if isinstance(i, dict) else {"name": str(i), "type": "Other", "usage": "Medium"}
                 for i in saved_items]
        st.session_state["appl_draft"], st.session_state["appl_draft_for"] = draft, household_id

    ver = st.session_state.get("appl_widget_ver", 0)
    if not draft:
        st.caption("No appliances yet. Add one below.")
    for i, item in enumerate(list(draft)):
        spec = catalog.effective_specs(item)
        summary = f"{spec['label']} · {spec['rated_w']:.0f} W" if spec else item.get("type", "Unknown")
        st.markdown(f"**{html.escape(item['name'])}**  \n<small>{html.escape(str(summary))}</small>",
                    unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([1.1, 1.3, 1.6, 0.7])
        item["quantity"] = int(c1.number_input("How many", 1, 10, int(item.get("quantity", 1) or 1), 1,
                                               key=f"ap_qty_{household_id}_{ver}_{i}"))
        typical = typical_hours(item)
        hours_default = float(item["hours_per_day"]) if item.get("hours_per_day") not in (None, "") \
            else (typical or 0.0)
        hours = c2.number_input("Hours a day", 0.0, 24.0, hours_default, 0.5, key=f"ap_hrs_{household_id}_{ver}_{i}",
                                help=f"Typical: {typical:g} h" if typical else None)
        if item.get("hours_per_day") in (None, "") and abs(hours - (typical or 0.0)) < 1e-9:
            item.pop("hours_per_day", None)               # left at the typical value (or not given)
        else:
            item["hours_per_day"] = float(hours)
        item["usage"] = c3.select_slider("Usage", ["Low", "Medium", "High"], value=item.get("usage", "Medium"),
                                         key=f"ap_use_{household_id}_{ver}_{i}",
                                         help="Used when hours a day is not known.")
        if c4.button("🗑️", key=f"ap_del_{household_id}_{ver}_{i}", help="Remove appliance", width="stretch",
                     disabled=len(draft) <= 1):
            draft.pop(i)
            _reset_appliance_widgets()
            st.rerun()

    st.divider()
    cat_cols = st.columns([2, 3, 1])
    category = cat_cols[0].selectbox("Category", catalog.CATEGORIES + ["Custom"], key="sb_cat",
                                     label_visibility="collapsed")
    if category == "Custom":
        name = cat_cols[1].text_input("Name", placeholder="e.g., sewing machine", key="sb_custom",
                                      label_visibility="collapsed")
        if cat_cols[2].button("Add", type="primary", key="sb_add_custom", width="stretch", disabled=not name.strip()):
            draft.append({"name": name.strip()[:40], "type": "Other", "usage": "Medium", "spec_mode": "own", "own": {}})
            st.rerun()
    else:
        types = catalog.types_in(category)
        pick = cat_cols[1].selectbox("Appliance", [t.name for t in types], key="sb_type", label_visibility="collapsed")
        if cat_cols[2].button("Add", type="primary", key="sb_add", width="stretch"):
            t = next(t for t in types if t.name == pick)
            draft.append(catalog.item_for(t.key))
            st.rerun()

    changed = _appliances_for_compare(draft) != _appliances_for_compare(saved_items)
    if changed:
        st.caption("You have unsaved changes to your appliances.")
    save_col, cancel_col = st.columns(2)
    if save_col.button("Save changes", type="primary", key="ap_save", width="stretch", disabled=not changed):
        try:
            # Appliance values are checked strictly: any problem means nothing is saved and the
            # draft stays here to be fixed.  (Other home fields keep their usual correction.)
            problems = validate_home_details({"appliances": draft})[1]
            if problems:
                raise InvalidSetup(" ".join(problems))
            clean, problems = validate_home_details({**hd, "appliances": draft})
            db.save_home_details(household_id, clean)
        except InvalidSetup as exc:
            st.error(f"These appliances could not be saved: {exc}")
            return
        except Exception as exc:
            log.exception("Saving appliances failed for household %s", household_id)
            st.error("The appliances could not be saved just now. Your changes are kept here; try again.")
            return
        hd["appliances"] = clean["appliances"]
        st.session_state.home_details = hd
        for key in ("appl_draft", "home_state", "_warm", "appl_edit_open"):
            st.session_state.pop(key, None)
        _reset_appliance_widgets()
        st.cache_data.clear()                 # predictions and tips are worked out again from the new list
        log.info("Saved %d appliances for household %s", len(clean["appliances"]), household_id)
        notify("success", "Appliances saved. Forecasts and tips now use the new list."
               + (" Corrected: " + "; ".join(problems) if problems else ""))
        st.rerun()
    if cancel_col.button("Cancel", key="ap_cancel", width="stretch", disabled=not (changed or closable)):
        st.session_state.pop("appl_draft", None)
        st.session_state.pop("appl_edit_open", None)
        _reset_appliance_widgets()
        st.rerun()


def render_dashboard_appliances(hd: dict) -> None:
    """The saved appliance list on the Dashboard, with an Edit appliances button beside its heading."""
    items = [a for a in (hd.get("appliances") or []) if isinstance(a, dict)]
    with st.container(border=True, key="dash_appliances"):
        head_col, btn_col = st.columns([4, 1.3], vertical_alignment="center")
        with head_col:
            section(f"Your appliances ({len(items)})")
        editing = bool(st.session_state.get("appl_edit_open"))
        if not editing and btn_col.button("Edit appliances", key="dash_edit_appliances", width="stretch",
                                          icon=":material/edit:"):
            st.session_state["appl_edit_open"] = True
            st.session_state.pop("appl_draft", None)
            _reset_appliance_widgets()
            st.rerun()
        if editing:
            _render_appliances_editor(hd, st.session_state.get("lang", "en"), closable=True)
        elif items:
            st.caption(" · ".join(
                f"{a.get('name', 'Appliance')}" + (f" ×{int(a.get('quantity', 1))}" if int(a.get('quantity', 1) or 1) > 1 else "")
                for a in items))
        else:
            st.caption("No appliances saved yet. Use Edit appliances to add the ones in your home.")


def ensemble_setting() -> dict:
    """The household's XGBoost/LSTM blend, remembered per household."""
    if "ensemble_setting" not in st.session_state:
        stored = None
        try:
            stored = get_db().get_store(current_household_id(), "ensemble_weights")
        except Exception:
            pass
        st.session_state.ensemble_setting = clean_setting(stored)
    return st.session_state.ensemble_setting


def _trained_blend(fc) -> tuple:
    b = fc.meta.get("blend", {})
    return float(b.get("w_xgb", 1.0)), float(b.get("w_lstm", 0.0))


def forecast_lstm_weight(fc):
    """LSTM share for the next-hour forecast: None keeps the blend chosen in training."""
    setting = ensemble_setting()
    if setting["mode"] != "custom" or not fc.lstm_available:
        return None
    return resolve_weights(setting, _trained_blend(fc))[1]


def render_ensemble_settings():
    fc = load_forecaster()
    val = fc.meta.get("hourly", {}).get("validation_mae_kw", {})
    if not fc.lstm_available:
        st.caption(T("ens_xgb_only"))
        return
    setting = dict(ensemble_setting())
    modes = ["auto", "custom"]
    mode = st.radio(T("ens_mode"), modes, index=modes.index(setting["mode"]),
                    format_func=lambda m: T("ens_auto" if m == "auto" else "ens_custom"),
                    key="ens_mode", horizontal=True)
    if mode == "custom":
        share = st.slider(T("ens_share"), 0, 100, int(round(setting["xgb_share"] * 100)), 5, key="ens_share")
        new = {"mode": "custom", "xgb_share": share / 100}
        st.caption(T("ens_custom_caption", x=share, l=100 - share))
    else:
        new = {"mode": "auto", "xgb_share": setting["xgb_share"]}
        w_x, w_l = resolve_weights(new, _trained_blend(fc))
        st.caption(T("ens_auto_caption", x=round(w_x * 100), l=round(w_l * 100),
                     xm=f"{float(val.get('xgboost') or 0):.3f}",
                     lm=f"{float(val.get('lstm') or 0):.3f}"))
    if new != setting:
        st.session_state.ensemble_setting = clean_setting(new)
        try:
            get_db().set_store(current_household_id(), "ensemble_weights", st.session_state.ensemble_setting)
        except Exception:
            pass


def render_alert_settings():
    """When to warn that the home is using more than forecast (saved per household)."""
    db, household_id = get_db(), current_household_id()
    try:
        import monitoring
        cur = monitoring.AlertSettings.clean(db.get_store(household_id, monitoring.SETTINGS_KEY))
    except Exception:
        log.exception("Could not read alert settings")
        st.info("Alert settings are not available right now.")
        return
    enabled = st.toggle("Warn me when use is above the forecast", value=cur.enabled, key="al_on")
    c1, c2, c3 = st.columns(3)
    pct = c1.number_input("More than (%)", 5.0, 200.0, float(cur.threshold_pct), 5.0, key="al_pct",
                          disabled=not enabled)
    kwh = c2.number_input("And at least (units/hour)", 0.05, 5.0, float(cur.min_excess_kwh), 0.05, key="al_kwh",
                          disabled=not enabled)
    hours = c3.number_input("For hours in a row", 1, 6, int(cur.consecutive_hours), 1, key="al_hrs",
                            disabled=not enabled)
    new = monitoring.AlertSettings.clean({"enabled": enabled, "threshold_pct": pct, "min_excess_kwh": kwh,
                                          "consecutive_hours": hours, "max_age_hours": cur.max_age_hours})
    if new != cur:
        try:
            db.set_store(household_id, monitoring.SETTINGS_KEY, new.__dict__)
            st.session_state.pop("_monitor_mark", None)          # check again with the new limits
        except Exception:
            log.exception("Could not save alert settings")
            st.error("The alert settings could not be saved just now. Try again.")
    st.caption(f"An alert is raised once per event, when each of {new.consecutive_hours} finished hour(s) in a row "
               f"used more than {new.threshold_pct:.0f}% and at least {new.min_excess_kwh:.2f} units above the "
               "forecast; a new alert needs use to return to normal first. Alerts are shown on the Dashboard and "
               "kept on the Notifications page, and emailed when email is set up.")


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
            section("Your Appliances")
            _render_appliances_editor(hd, lang)
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
            section(T("ens_title"))
            render_ensemble_settings()
        with st.container(border=True):
            section("Usage alerts")
            render_alert_settings()
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
        # The details go to the log (with the traceback) and never to the
        # screen: they mean nothing to the person and can leak internals.
        log.exception("Dashboard failed to render: %s", type(e).__name__)
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
        forecaster = load_forecaster()
        forecast_df = build_forecast_df(source, raw_data)

    if raw_data is None or raw_data.empty:
        st.error(T("src_upload_fail", reason=T("src_need_power")))
        return

    src_info = source_summary(source)
    can_predict = src_info["caps"].get("prediction", {}).get("enabled", False)
    can_appliances = src_info["caps"].get("appliances", {}).get("enabled", False)

    full_data = raw_data.copy()
    full_data["Global_active_power"] = full_data["Global_active_power"] * sf
    if not forecast_df.empty:
        forecast_df = forecast_df.copy()
        forecast_df["predicted_kw"] = forecast_df["predicted_kw"] * sf
        forecast_df["predicted_kwh"] = forecast_df["predicted_kwh"] * sf

    sim = get_simulator()
    # "Next month" is the calendar month after the newest reading (for the sample,
    # that is next month on today's calendar).
    newest = pd.to_datetime(raw_data["datetime"]).max()
    next_month_start = (newest.normalize().replace(day=1) + pd.DateOffset(months=1))
    fc_hourly = forecaster.meta.get("hourly", {})
    if "sanctioned_kw" not in st.session_state:          # the scheduler keeps moves under it
        try:
            stored = get_db().get_store(household_id, "toolkit", {}) or {}
            st.session_state["sanctioned_kw"] = float(stored.get("sanctioned_kw", 3.0))
        except Exception:
            st.session_state["sanctioned_kw"] = 3.0
    hybrid_mae = fc_hourly.get("mae_kw", {}).get("hybrid") or 0.3

    # Load previous simulation history from DB and combine with current replay
    if "sim_history_loaded" not in st.session_state:
        db = get_db()
        hist_df = db.get_simulation_history(household_id, limit=5000)
        if not hist_df.empty:
            # Combine historical data with current replay source data
            sim.set_data(pd.concat([hist_df, source.df]).drop_duplicates(subset=["datetime"]).sort_values("datetime"), restart=False)
        st.session_state.sim_history_loaded = True

    # The replay advances on a background thread, but Streamlit only redraws on
    # interaction, so without this the displayed reading stays frozen while the
    # simulator moves on. Refresh only while the replay is actually moving, and
    # stop once it reaches the end so the session does not rerun forever.
    if st.session_state.get("page") == "trends":
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

    # Next-hour forecast: the household's raw readings up to the "current"
    # hour go in; the household scaling factor is applied once to the outputs.
    nh = None
    if can_predict and forecaster.ready and live_row:
        row_t = live_row.get("datetime")
        upto = raw_data
        if row_t is not None and not pd.isna(row_t):
            upto = raw_data[pd.to_datetime(raw_data["datetime"]) <= pd.Timestamp(row_t)]
        try:
            nh = forecaster.next_hour(upto, w_lstm=forecast_lstm_weight(forecaster))
        except Exception as exc:
            log.exception("Next-hour forecast failed")
            nh = None
    if nh:
        pred_kw = nh["pred_kw"] * sf
        xgb_p = nh["xgb_kw"] * sf
        lstm_p = nh["lstm_kw"] * sf if nh["lstm_kw"] is not None else None
        pred_info = {"lstm_used": nh["lstm_used"], "w_xgb": nh["w_xgb"], "w_lstm": nh["w_lstm"],
                     "improvement_pct": nh.get("improvement_pct"),
                     "persistence_mae": nh.get("persistence_mae_kw")}
        if nh["lower_kw"] is not None:
            lower, upper = nh["lower_kw"] * sf, nh["upper_kw"] * sf
            pred_info.update(pi_lower=lower, pi_upper=upper, pi_coverage=nh.get("coverage"))
        else:
            lower, upper = max(0.0, pred_kw - hybrid_mae * sf), pred_kw + hybrid_mae * sf
    else:
        pred_kw, xgb_p, lstm_p = gap, None, None
        pred_info = {"lstm_used": False, "w_xgb": 1.0, "w_lstm": 0.0}
        lower, upper = max(0.0, pred_kw - hybrid_mae * sf), pred_kw + hybrid_mae * sf
    uncertainty = (upper - lower) / 2.0

    # Costs for today and the week use every reading up to the current hour
    # (full_data is already scaled), not only what the replay has shown so far.
    row_t = live_row.get("datetime") if live_row else None
    if row_t is not None and not pd.isna(row_t):
        history = full_data[pd.to_datetime(full_data["datetime"]) <= pd.Timestamp(row_t)].copy()
    else:
        replayed = sim.get_full_history()
        history = replayed.copy() if not replayed.empty else replayed
        if not history.empty:
            history["Global_active_power"] = history["Global_active_power"] * sf
    if not history.empty:
        
        # Save replay history to DB periodically (every 50 rows), unscaled, so it
        # is not scaled a second time when it is replayed in a later session.
        if rows_played > 0 and rows_played % 50 == 0:
            db = get_db()
            db.save_simulation_history(household_id, sim.get_full_history())
        
        latest_date = history["datetime"].dt.date.iloc[-1]
        today_info = daily_cost(history, str(latest_date), tariff_rate,
                                tuple(peak_morning), tuple(peak_evening))
        week_start_date = latest_date - pd.Timedelta(days=6)
        week_info = weekly_cost(history, str(week_start_date), tariff_rate,
                                tuple(peak_morning), tuple(peak_evening))
        nm_info = next_month_cost(month_slice(forecast_df, next_month_start.year, next_month_start.month),
                                  tariff_rate, tuple(peak_morning), tuple(peak_evening))
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

    # Helper functions for page rendering
    def recorded_block():
        """Next hour and next month from the recorded reference dataset, with the hybrid model."""
        for note in missing_notes:
            st.warning(note)
        section("Reference recording: next hour and next month (hybrid model)")
        st.caption("This part uses a real household recording replayed at today's date, to show the "
                   "LSTM and XGBoost forecast model. It is a different home from your simulated meter, "
                   "so its figures are not your bill.")
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
            base_mae = fc_hourly.get("mae_kw", {}).get("persistence")
            gain = fc_hourly.get("improvement_over_persistence_pct")
            accuracy_text = (T("cap_forecast_vs_base", pct=f"{gain:.0f}", base=f"±{base_mae * sf:.2f}")
                             if base_mae and gain is not None else "")
            model_name = T("pred_model_hybrid" if pred_info.get("lstm_used") else "pred_model_xgb_only")
            st.markdown(
                f"<div class='ep-stat'><small>{T('card_next_month')}</small>"
                f"<strong>Rs. {fmt_rs(nm_info['total_cost'])}</strong>"
                f"<span>{T('card_projected_for', month=format_localized_month(next_month_start))}</span></div>"
                f"<div class='ep-stat'><small>{T('card_smart_forecast')}</small>"
                f"<strong>&plusmn;{hybrid_mae * sf:.2f} kW</strong>"
                f"<span>{model_name}. {accuracy_text}</span></div>",
                unsafe_allow_html=True)
        st.caption(source_text)
        st.markdown(f"""<div class="disc"><p>{T('about_demo')}</p></div>""", unsafe_allow_html=True)

    def toolkit_ctx():
        scenario, tod = meter_settings()
        db_ = get_db()
        return {"pred": detected_history(today_str(), scenario, owned), "owned": owned,
                "switched": household_devices(home_details)[1], "tariff": Tariff(rate=float(tariff_rate), tod_enabled=tod),
                "rate": float(tariff_rate), "db": db_, "household_id": household_id, "user_name": user_display,
                "people": household_people(db_, household_id, user_display, auth_email),
                "section": section, "metric_card": metric_card, "layout": PLOTLY_LAYOUT, "ledger": get_ledger(),
                "brief": todays_brief(tariff_rate, owned, household_id, user_display, auth_email),
                "specs": __import__("devices").appliance_specs(home_details)}

    if page == "dashboard":
        head()
        prepare_meter("console")
        render_brief_strip(todays_brief(tariff_rate, owned, household_id, user_display, auth_email))
        left_mark = (household_id, today_str())
        if st.session_state.get("_left_on_mark") != left_mark:      # once per sign-in per day
            st.session_state["_left_on_mark"] = left_mark
            try:
                scenario_, tod_ = meter_settings()
                today_rows = detected_history(today_str(), scenario_, owned).tail(1440).iloc[: now_minute() + 1]
                flags = flag_long_runs(extract_sessions(today_rows, Tariff(rate=float(tariff_rate), tod_enabled=tod_)))
                for text in send_left_on(get_db(), household_id,
                                         household_people(get_db(), household_id, user_display, auth_email),
                                         flags, st.session_state.get("lang", "en"))[:2]:
                    notify("warning", text + ". See the Safety page.")
            except Exception:
                pass
        render_dashboard_appliances(home_details)
        render_home_tab(tariff_rate, home_details, db=get_db(), household_id=household_id,
                        user_name=st.session_state.auth.get("name") or T("guest_name"),
                        user_email=auth_email, theme=current_theme(),
                        language=st.session_state.get("lang", "en"))

        # Add forecast overview section (from old overview page)
        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
        head(meter_chip, "sim")
        prepare_meter("page")
        try:
            people_now = household_people(get_db(), household_id, user_display, auth_email)
        except Exception:
            log.exception("Could not read household members")
            people_now = [{"name": user_display, "email": auth_email}]
        render_now_section(tariff_rate, owned, household_id=household_id, people=people_now)
        render_brief_section(todays_brief(tariff_rate, owned, household_id, user_display, auth_email))
        render_tomorrow_section(tariff_rate, owned)

        # Add trends (from old trends page)
        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
        head(T("live_label") + " &middot; " + when_text, "")
        recorded_block()
        section(T("sec_trends"))
        plot_df = pd.DataFrame()
        if not full_data.empty:
            plot_df = full_data.tail(display_count)[["datetime", "Global_active_power"]].copy()
            plot_df["Global_active_power"] = plot_df["Global_active_power"].astype(float) * sf
            plot_df["datetime"] = pd.to_datetime(plot_df["datetime"])
            plot_df = (plot_df.assign(hour_bin=plot_df["datetime"].dt.floor("h"))
                              .groupby("hour_bin", as_index=False)["Global_active_power"]
                              .mean()
                              .rename(columns={"hour_bin": "datetime"})
                              .sort_values("datetime")
                              .reset_index(drop=True))
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
                tick_fmt = "%I %p"
            elif time_span <= pd.Timedelta(days=10):
                tick_fmt = "%b %d %I %p"
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

    if page == "appliances":
        head(meter_chip, "sim")
        prepare_meter("page")
        # Tabs for: My Devices | What Meter Shows | Where Power Goes
        tab1, tab2, tab3 = st.tabs(["My Devices", "What Meter Shows", "Where Power Goes"])
        with tab1:
            render_my_appliances(get_db(), household_id, section, notify)
        with tab2:
            render_appliance_tab(tariff_rate, owned, section, PLOTLY_LAYOUT)
        with tab3:
            render_analysis_tab(tariff_rate, owned, section, PLOTLY_LAYOUT)

    if page == "optimize":
        head(meter_chip, "sim")
        prepare_meter("page")
        # Tabs for: Savings Tips | Monthly Goals | Upgrade Calculator
        tab1, tab2, tab3 = st.tabs(["Savings Tips", "Monthly Goals", "Upgrade Calculator"])
        ctx = toolkit_ctx()
        with tab1:
            render_save_tab(tariff_rate, owned, section, PLOTLY_LAYOUT)
        with tab2:
            toolkit_ui.render_goals(ctx)
        with tab3:
            toolkit_ui.render_upgrades(ctx)

    if page == "bills_safety":
        head()
        prepare_meter("page")
        # Tabs for: Bills | Safety | Alerts
        tab1, tab2, tab3 = st.tabs(["Bills", "Safety", "Alerts"])
        with tab1:
            load_saved_bills()
            toolkit_ui.render_bill_tools(toolkit_ctx(), st.session_state.get("bills"))
            render_bills_tab()
        with tab2:
            toolkit_ui.render_safety(toolkit_ctx())
        with tab3:
            render_notifications_tab(
                get_db(), household_id, auth_email,
                language=st.session_state.get("lang", "en"),
                tariff_rate=tariff_rate,
                history=history,
                forecast_df=month_slice(forecast_df, next_month_start.year, next_month_start.month),
            )

    if page == "household":
        head()
        prepare_meter("page")
        # Tabs for: Family | Assistant
        tab1, tab2 = st.tabs(["Family", "Energy Assistant"])
        with tab1:
            render_family_tab(get_db(), household_id)
            toolkit_ui.render_family_extras(toolkit_ctx())
        with tab2:
            render_chat_tab(
                get_db(), household_id, auth_email,
                language=st.session_state.get("lang", "en"),
                full_data=full_data,
                scaling_factor=sf,
                tariff_rate=tariff_rate,
                meter_answer=lambda q: _meter_answer(q, toolkit_ctx()),
            )

    if page == "help":
        head()
        prepare_meter("page")
        toolkit_ui.render_help(toolkit_ctx())

    if page == "settings":
        head()
        render_settings_page()


def main():
    # Initialize background scheduler once per process
    # ENERGYPULSE_NO_SCHEDULER=1 (automated tests) runs the app without background jobs.
    if "scheduler_initialized" not in st.session_state and \
            os.environ.get("ENERGYPULSE_NO_SCHEDULER", "").strip().lower() not in ("1", "true", "yes"):
        try:
            from scheduler_jobs import init_scheduler
            init_scheduler()
            st.session_state.scheduler_initialized = True
        except Exception as e:
            log.exception("Scheduler init failed")
    
    apply_theme()
    inject_skin()
    flush_cookie_changes()
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
