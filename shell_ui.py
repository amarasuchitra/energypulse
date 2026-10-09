"""
App shell
=========
Everything that makes the site look and move the way it does, in one place:

  inject_skin()     the stylesheet (type, colour, layout, motion)
  sidebar_nav()     the left navigation
  page_header()     the title block at the top of each page
  login_hero()      the brand panel on the sign-in screen
  step_list()       the numbered steps on the setup screen
  strip_emoji()     removes emoji from any text before it is shown

Type has three voices, each with one job:
  Sora              titles and statements
  Hanken Grotesk    everything you read or press
  IBM Plex Mono     meter readings and money
"""

import re
from urllib.parse import quote

import streamlit as st

from motion import MOTION_CSS

_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF☀-➿⬀-⯿←-⇿⏩-⏿️‍⃣]+")


def strip_emoji(text):
    """Text only: no emoji, no decorative arrows or ticks."""
    if not isinstance(text, str):
        return text
    return re.sub(r"[ \t]{2,}", " ", _EMOJI.sub("", text)).strip()


# ----------------------------------------------------------------- icons
# Stroke icons, drawn as CSS masks so they take the colour of the text.
_ICON_PATHS = {
    "home": "<path d='M4 11l8-7 8 7v9H4z'/><path d='M10 20v-6h4v6'/>",
    "overview": "<path d='M4 18a8 8 0 1 1 16 0'/><path d='M12 18l4-6'/>",
    "appliances": "<path d='M9 3v5M15 3v5M6 8h12v4a6 6 0 0 1-12 0zM12 18v3'/>",
    "analysis": "<path d='M12 3a9 9 0 1 0 9 9h-9z'/><path d='M15 3.5A9 9 0 0 1 20.5 9H15z'/>",
    "save": "<path d='M5 19c0-8 5-13 14-14 0 9-5 14-13 14'/><path d='M5 19c3-5 6-7 9-9'/>",
    "bills": "<path d='M6 3h12v18l-3-2-3 2-3-2-3 2z'/><path d='M9 8h6M9 12h6'/>",
    "trends": "<path d='M4 19V5M4 19h16'/><path d='M7 15l4-5 3 3 5-7'/>",
    "family": "<circle cx='9' cy='8' r='3'/><path d='M3 20a6 6 0 0 1 12 0'/><circle cx='17' cy='9' r='2.3'/><path d='M16 14.5a5 5 0 0 1 5 5.5'/>",
    "alerts": "<path d='M6 17V11a6 6 0 0 1 12 0v6l2 2H4z'/><path d='M10 21h4'/>",
    "assistant": "<path d='M4 5h16v11H9l-5 4z'/><path d='M8 9h8M8 12h5'/>",
    "devices": "<rect x='3' y='7' width='8' height='12' rx='1.5'/><rect x='13' y='3' width='8' height='16' rx='1.5'/><path d='M7 15.5h.01M17 15.5h.01M17 7h.01'/>",
    "safety": "<path d='M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z'/><path d='M9 12l2 2 4-4'/>",
    "upgrades": "<circle cx='12' cy='12' r='4'/><path d='M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M19 5l-2 2M7 17l-2 2'/>",
    "goals": "<circle cx='12' cy='12' r='9'/><circle cx='12' cy='12' r='5'/><circle cx='12' cy='12' r='1'/>",
    "help": "<circle cx='12' cy='12' r='9'/><path d='M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.7.4-1 .9-1 1.7M12 17h.01'/>",
    "settings": "<path d='M4 7h10M18 7h2M4 17h2M10 17h10'/><circle cx='16' cy='7' r='2'/><circle cx='8' cy='17' r='2'/>",
}


def _mask(name: str) -> str:
    svg = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' "
           "stroke-width='1.7' stroke-linecap='round' stroke-linejoin='round'>"
           f"{_ICON_PATHS[name]}</svg>")
    return f'url("data:image/svg+xml,{quote(svg)}")'


PULSE_SVG = ("<svg class='ep-pulse' viewBox='0 0 120 28' aria-hidden='true'>"
             "<path d='M1 14h26l5-9 7 20 7-24 6 18 4-5h63' pathLength='1'/></svg>")


def wordmark(size: str = "") -> str:
    return (f"<div class='ep-wordmark {size}'><svg viewBox='0 0 34 20' aria-hidden='true'>"
            "<path d='M1 10h8l2.5-6 4 13 4-15 3 11 2-3h8.5' pathLength='1'/></svg>"
            "<span>EnergyPulse</span></div>")


# ----------------------------------------------------------------- stylesheet
THEMES = {
    "dark": {"base": "dark", "primaryColor": "#F5A83C", "backgroundColor": "#141824",
             "secondaryBackgroundColor": "#1C2232", "textColor": "#E6E9F0"},
    "light": {"base": "light", "primaryColor": "#B96D05", "backgroundColor": "#F3F4F8",
              "secondaryBackgroundColor": "#FFFFFF", "textColor": "#171B28"},
}
_HEX = {
    "dark": {"ink": "#141824", "slate": "#1C2232", "line": "#313A55", "plaster": "#E6E9F0",
             "mist": "#939CB3", "faint": "#6F7994", "grid": "rgba(255,255,255,0.07)"},
    "light": {"ink": "#F3F4F8", "slate": "#FFFFFF", "line": "#D8DCE7", "plaster": "#171B28",
              "mist": "#596279", "faint": "#8A93A8", "grid": "rgba(23,27,40,0.08)"},
}
_LIGHT_CSS = """
:root {
  --ink: #f3f4f8; --slate: #ffffff; --slate-2: #eceef5; --line: #d8dce7;
  --plaster: #171b28; --soft: #39415a; --mist: #596279; --faint: #8a93a8;
  --current: #b96d05; --alert: #cf4630; --ok: #1c9a60;
  --fill: #f5a83c; --side: #fbfbfd; --field: #ffffff; --hover: rgba(23,27,40,.05);
}
[data-testid="stAlertContentWarning"] { color: #8a5200 !important; }
"""


def current_theme() -> str:
    try:
        return "light" if st.session_state.get("theme") == "light" else "dark"
    except Exception:
        return "dark"


def tc(name: str) -> str:
    """A colour from the current theme, as hex, for charts (which cannot read CSS variables)."""
    return _HEX[current_theme()][name]


def apply_theme() -> str:
    """
    Point Streamlit's own widgets at the chosen theme.  The setting belongs
    to the running server, so on a shared server everyone sees the same mode.
    """
    theme = current_theme()
    changed = False
    for key, value in THEMES[theme].items():
        if st.get_option(f"theme.{key}") != value:
            st._config.set_option(f"theme.{key}", value)
            changed = True
    if changed:
        st.rerun()
    return theme


def theme_toggle():
    theme = current_theme()
    label = "Light mode" if theme == "dark" else "Dark mode"
    if st.button(label, key="theme_toggle", width="stretch"):
        st.session_state.theme = "light" if theme == "dark" else "dark"
        st.rerun()


def inject_skin():
    nav_icons = "\n".join(
        f".st-key-nav_{k} button::before {{ -webkit-mask-image: {_mask(k)}; mask-image: {_mask(k)}; }}"
        for k in _ICON_PATHS)
    light = _LIGHT_CSS if current_theme() == "light" else ""
    pulse = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 120 28' fill='none' stroke='black' stroke-width='1.6' "
             "stroke-linecap='round' stroke-linejoin='round'><path d='M1 14h26l5-9 7 20 7-24 6 18 4-5h63'/></svg>")
    tokens = f":root {{ --pulse-mask: url(\"data:image/svg+xml,{quote(pulse)}\"); }}"
    st.markdown(f"<style>{_CSS}\n{nav_icons}\n{tokens}\n{MOTION_CSS}\n{light}</style>", unsafe_allow_html=True)


_CSS = r"""
@font-face { font-family: "Hanken Grotesk"; font-weight: 100 900; font-display: swap;
  src: url("app/static/fonts/hanken-grotesk-latin-wght-normal.woff2") format("woff2"); }
@font-face { font-family: "IBM Plex Mono"; font-weight: 500; font-display: swap;
  src: url("app/static/fonts/ibm-plex-mono-latin-500-normal.woff2") format("woff2"); }
@font-face { font-family: "Sora"; font-weight: 100 800; font-display: swap;
  src: url("app/static/fonts/sora-latin-wght-normal.woff2") format("woff2"); }

:root {
  --ink: #141824; --slate: #1c2232; --slate-2: #252c40; --line: #313a55;
  --plaster: #e6e9f0; --soft: #c5cbd9; --mist: #939cb3; --faint: #6f7994;
  --current: #f5a83c; --alert: #ef6a5b; --ok: #5fcb8f;
  --fill: #f5a83c; --side: #10141e; --field: #10141e; --hover: rgba(255,255,255,.045);
  --display: "Sora", "Hanken Grotesk", "Segoe UI", sans-serif;
  --ui: "Hanken Grotesk", "Segoe UI", system-ui, sans-serif;
  --digits: "IBM Plex Mono", ui-monospace, Consolas, monospace;
  --ease: cubic-bezier(.2, .7, .2, 1);
}

.stApp { background: var(--ink); }
.stApp, .stApp p, .stApp label, .stApp input, .stApp button, .stApp li,
section[data-testid="stSidebar"] { font-family: var(--ui) !important; }
#MainMenu, footer, [data-testid="stToolbarActions"], [data-testid="stMainMenu"], .stAppDeployButton,
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
header[data-testid="stHeader"] { visibility: visible !important; background: transparent !important; height: 2.6rem; pointer-events: none; }
header[data-testid="stHeader"] button { pointer-events: auto; visibility: visible !important; }
.block-container { padding: 2.2rem 2.6rem 4rem !important; max-width: 1480px; }
::selection { background: rgba(245,168,60,.35); }

/* ---------- motion: one entrance per page, nothing loops for decoration ---------- */
@keyframes ep-rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes ep-fade { from { opacity: 0; } to { opacity: 1; } }
@keyframes ep-draw { from { stroke-dashoffset: 1; } to { stroke-dashoffset: 0; } }
@keyframes ep-sweep { from { transform: scaleX(0); } to { transform: scaleX(1); } }
[class*="st-key-page_"] { animation: ep-rise .5s var(--ease) both; }
.ep-head { animation: ep-fade .5s ease both; }
.ep-head h1 { animation: ep-rise .6s var(--ease) both; }
.ep-head p { animation: ep-rise .6s var(--ease) .08s both; }
.ep-head .ep-rule { transform-origin: left; animation: ep-sweep .8s var(--ease) .15s both; }
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}

/* ---------- wordmark ---------- */
.ep-wordmark { display: flex; align-items: center; gap: 10px; color: var(--plaster); }
.ep-wordmark span { font-family: var(--display); font-weight: 650; font-size: 1.22rem; line-height: 1; letter-spacing: -0.035em; }
.ep-wordmark svg { width: 34px; height: 20px; flex: none; }
.ep-wordmark path, .ep-pulse path { fill: none; stroke: var(--current); stroke-width: 1.8;
  stroke-linecap: round; stroke-linejoin: round; stroke-dasharray: 1; animation: ep-draw 1.4s var(--ease) both; }
.ep-wordmark.big span { font-size: 1.5rem; }

/* ---------- page header ---------- */
.ep-head { margin: 0 0 1.4rem; }
.ep-head-row { display: flex; align-items: flex-end; justify-content: space-between; gap: 24px; flex-wrap: wrap; }
.ep-head h1 { font-family: var(--display); font-weight: 600; font-size: clamp(1.7rem, 2.5vw, 2.3rem);
  line-height: 1.1; letter-spacing: -0.04em; color: var(--plaster); margin: 0; padding: 0; }
.ep-head p { color: var(--mist); font-size: 0.98rem; margin: .55rem 0 0; max-width: 62ch; }
.ep-rule { height: 1px; background: var(--line); margin-top: 1.1rem; }
.ep-chip { display: inline-flex; align-items: center; gap: 8px; font-size: .8rem; color: var(--mist);
  border: 1px solid var(--line); border-radius: 999px; padding: 6px 12px; white-space: nowrap; }
.ep-chip i { width: 7px; height: 7px; border-radius: 50%; background: var(--ok); }
.ep-chip.sim i { background: var(--current); }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] { background: var(--side) !important; border-right: 1px solid var(--line) !important; }
section[data-testid="stSidebar"] > div { padding-top: 1.4rem; }
@media (min-width: 901px) {
  section[data-testid="stSidebar"][aria-expanded="true"] { min-width: 248px !important; max-width: 248px !important; }
  section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] { display: none; }
}
@media (max-width: 900px) {
  section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] { padding: 0 .6rem; height: 2.4rem; }
  section[data-testid="stSidebar"] > div { padding-top: .2rem; }
}
[class*="st-key-nav_"] { margin-bottom: -0.72rem; }
section[data-testid="stSidebar"] [class*="st-key-nav_"] button { justify-content: flex-start !important; gap: 12px; width: 100%;
  background: transparent; border: 0 !important; border-radius: 10px !important;
  padding: 9px 12px !important; color: var(--mist) !important; font-weight: 550 !important;
  transition: background .18s ease, color .18s ease, transform .18s var(--ease) !important; }
section[data-testid="stSidebar"] [class*="st-key-nav_"] button > div { flex: 0 1 auto !important; width: auto !important; margin: 0 !important; }
[class*="st-key-nav_"] button p { font-size: .92rem !important; font-weight: 550 !important; color: inherit !important; text-align: left; }
[class*="st-key-nav_"] button::before { content: ""; width: 18px; height: 18px; flex: none; background: currentColor;
  -webkit-mask: center / contain no-repeat; mask: center / contain no-repeat; }
section[data-testid="stSidebar"] [class*="st-key-nav_"] button:hover { background: var(--hover); color: var(--plaster) !important;
  transform: translateX(2px); }
[class*="st-key-nav_"] button:focus-visible { outline: 2px solid var(--current) !important; outline-offset: 2px; }
.ep-nav-gap { height: 1px; background: var(--line); margin: 1.1rem 4px .5rem; }
.ep-side-home { margin: 1.2rem 4px 0; padding: 12px; border: 1px solid var(--line); border-radius: 12px; }
.ep-side-home b { display: block; color: var(--plaster); font-weight: 600; font-size: .88rem; }
.ep-side-home span { color: var(--faint); font-size: .78rem; }

section[data-testid="stSidebar"] .stMarkdown p, section[data-testid="stSidebar"] .stMarkdown span { color: var(--mist) !important; }
.st-key-theme_toggle { margin: .9rem 4px 0; }
.st-key-theme_toggle button p { font-size: .84rem !important; }

/* ---------- surfaces ---------- */
[data-testid="stVerticalBlockBorderWrapper"] { background: var(--slate); border-color: var(--line) !important;
  border-radius: 16px !important; }
[data-testid="stExpander"] details { background: var(--slate); border: 1px solid var(--line) !important; border-radius: 14px !important; }
[data-testid="stAlert"] { background: var(--slate) !important; border: 1px solid var(--line) !important;
  border-radius: 12px !important; color: var(--soft) !important; }
[data-testid="stAlert"] p { color: var(--soft) !important; }
[data-testid="stAlertContentWarning"] { color: #ffd9a8 !important; }
[data-testid="stCaptionContainer"] p { color: var(--faint) !important; font-size: .82rem !important; }
hr { border-color: var(--line) !important; }

.metric-card { background: var(--slate) !important; border: 1px solid var(--line) !important; box-shadow: none !important;
  border-radius: 16px !important; padding: 1.1rem 1.2rem !important; min-height: 128px; transform: none !important;
  transition: border-color .2s ease; }
.metric-card:hover { border-color: #47537a !important; }
.metric-card::before { display: none !important; }
.mc-label { text-transform: none !important; letter-spacing: 0 !important; font-size: .84rem !important;
  font-weight: 600 !important; color: var(--mist) !important; }
.mc-value { font-family: var(--digits) !important; font-weight: 500 !important; font-size: 1.9rem !important;
  letter-spacing: -0.03em; color: var(--plaster) !important; margin-top: .35rem; }
.mc-unit { font-family: var(--ui); font-size: .82rem !important; color: var(--faint) !important; letter-spacing: 0; }
.mc-sub { color: var(--faint) !important; font-size: .78rem !important; }
.mc-trend { background: none !important; padding: 0 !important; font-size: .78rem !important; font-weight: 600 !important; }
.mc-trend.neutral { color: var(--mist) !important; }
.mc-trend.up { color: var(--alert) !important; }
.mc-trend.down { color: var(--ok) !important; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--line); background: transparent !important; }
.stTabs [data-baseweb="tab"] { padding: 9px 16px !important; border-radius: 10px 10px 0 0 !important; background: transparent !important;
  color: var(--mist) !important; font-weight: 500; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { background: var(--slate-2) !important; color: var(--plaster) !important;
  box-shadow: inset 0 -2px 0 var(--current); }
[data-baseweb="tab-panel"] { padding-top: 1rem !important; }
.stButton button:disabled, [data-testid="stBaseButton-primary"]:disabled {
  opacity: .38 !important; cursor: not-allowed !important; filter: saturate(.4); }
[class*="st-key-am_drop_zone"] [data-testid="stFileUploaderDropzone"],
[class*="st-key-home_drop_zone"] [data-testid="stFileUploaderDropzone"] {
  min-height: 132px; border: 2px dashed var(--line) !important; border-radius: 16px !important;
  background: var(--slate) !important; transition: border-color var(--d2, .2s) ease, background var(--d2, .2s) ease; }
[class*="st-key-am_drop_zone"] [data-testid="stFileUploaderDropzone"]:hover,
[class*="st-key-home_drop_zone"] [data-testid="stFileUploaderDropzone"]:hover,
[data-testid="stFileUploaderDropzone"][data-dragging="true"] {
  border-color: var(--current) !important; background: var(--slate-2) !important; }
.ep-pw-rules { margin: -.2rem 0 .7rem; padding: .7rem .9rem; border: 1px solid var(--line); border-radius: 12px;
  background: var(--ink); font-size: .78rem; color: var(--mist); }
.ep-pw-rules b { color: var(--plaster); font-weight: 600; }
.ep-pw-rules ul { margin: .35rem 0 0; padding-left: 1.1rem; }
.ep-pw-rules li { margin: .1rem 0; }
.ep-brief { display: flex; flex-wrap: wrap; gap: .4rem 1.4rem; align-items: baseline; margin: 0 0 1rem;
  padding: .85rem 1.1rem; background: var(--slate); border: 1px solid var(--line);
  border-left: 3px solid var(--current); border-radius: 14px; font-size: .88rem; color: var(--mist); }
.ep-brief b { color: var(--plaster); font-weight: 600; }
.ep-brief strong { font-family: 'IBM Plex Mono', monospace; font-weight: 500; color: var(--plaster); }
.ep-plan { padding: .2rem 0 .1rem; }
.ep-plan-item { display: grid; grid-template-columns: 1.7rem minmax(0, 1fr) auto; gap: .2rem .8rem; padding: .8rem 0;
  border-top: 1px solid var(--line); }
.ep-plan-item:first-child { border-top: 0; }
.ep-plan-item i { font-style: normal; font-family: 'IBM Plex Mono', monospace; color: var(--current); }
.ep-plan-item b { color: var(--plaster); font-weight: 600; }
.ep-plan-item p { margin: .2rem 0 0; font-size: .82rem; color: var(--mist); grid-column: 2 / 4; }
.ep-plan-item code { display: block; margin-top: .35rem; grid-column: 2 / 4; font-size: .76rem; white-space: normal;
  background: var(--ink); color: var(--mist); border: 1px solid var(--line); border-radius: 8px; padding: .4rem .6rem; }
.ep-plan-item em { font-style: normal; font-family: 'IBM Plex Mono', monospace; color: var(--ok); white-space: nowrap; }
.ep-rows { width: 100%; border-collapse: collapse; font-size: .84rem; border: 0 !important; }
.ep-rows td { padding: .5rem .2rem; border: 0 !important; border-top: 1px solid var(--line) !important; color: var(--mist); vertical-align: middle; }
.ep-rows tr:first-child td { border-top: 0 !important; }
.ep-rows td.n { color: var(--plaster); font-weight: 600; }
.ep-rows td.v { text-align: right; font-family: 'IBM Plex Mono', monospace; color: var(--plaster); white-space: nowrap; }
.ep-rows small { display: block; font-size: .74rem; color: var(--mist); font-weight: 400; }
.ep-rows .bar { height: 6px; border-radius: 3px; background: var(--line); min-width: 70px; overflow: hidden; }
.ep-rows .bar span { display: block; height: 100%; background: var(--current); border-radius: 3px; }
.ep-stat { background: var(--slate); border: 1px solid var(--line); border-radius: 16px; padding: 1.2rem 1.3rem; }
.ep-stat + .ep-stat { margin-top: 12px; }
.ep-stat small { display: block; color: var(--mist); font-size: .84rem; font-weight: 600; }
.ep-stat strong { display: block; font-family: var(--digits); font-weight: 500; font-size: 2.3rem;
  letter-spacing: -0.03em; color: var(--plaster); margin: .3rem 0 .2rem; }
.ep-stat span { color: var(--faint); font-size: .8rem; }

.section-header { margin: 1.6rem 0 .6rem !important; }
[data-testid="stVerticalBlockBorderWrapper"] .section-header, .st-key-onboard_panel .section-header { margin-top: .1rem !important; }
.section-header h2 { font-family: var(--display) !important; font-weight: 600 !important; font-size: 1.12rem !important;
  letter-spacing: -0.03em; color: var(--plaster) !important; }
.section-line { display: none !important; }
.tip-box, .live-bar, .disc, .error-card { background: var(--slate) !important; border: 1px solid var(--line) !important;
  box-shadow: none !important; border-radius: 12px !important; }
.disc { border-left: 1px solid var(--line) !important; margin-top: 2.4rem; }
.disc p, .disc strong { color: var(--faint) !important; font-size: .78rem !important; }

/* ---------- controls ---------- */
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
  border-radius: 999px !important; box-shadow: none !important; font-weight: 600 !important;
  border: 1px solid var(--line) !important; background: transparent; color: var(--plaster);
  transition: background .18s ease, border-color .18s ease, transform .12s ease !important; }
.stButton > button:hover, .stFormSubmitButton > button:hover { border-color: #56628c !important; transform: none !important; }
.stButton > button:active, .stFormSubmitButton > button:active { transform: scale(.985) !important; }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind^="primary"],
button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-primaryFormSubmit"] {
  background: var(--fill) !important; border-color: var(--fill) !important; }
.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primaryFormSubmit"]:hover { background: #ffb955 !important; }
.stButton > button[kind="primary"] p, button[data-testid="stBaseButton-primaryFormSubmit"] p,
button[data-testid="stBaseButton-primary"] p { color: #17130a !important; }
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div, [data-baseweb="textarea"],
[data-testid="stNumberInputContainer"], [data-testid="stTextInputRootElement"] {
  background: var(--field) !important; border-color: var(--line) !important; border-radius: 10px !important; }
[data-testid="stTextInputRootElement"], [data-testid="stNumberInputContainer"] { border: 1px solid var(--line) !important; }
[data-testid="stTextInputRootElement"]:focus-within, [data-testid="stNumberInputContainer"]:focus-within { border-color: var(--current) !important; }
[data-baseweb="input"] input, [data-baseweb="base-input"] input { background: transparent !important; color: var(--plaster) !important; }
/* the refresh timer is invisible and takes no room */
[data-testid="stElementContainer"]:has(iframe[title="streamlit_autorefresh.st_autorefresh"]) {
  position: absolute; height: 0; overflow: hidden; }
/* style-only blocks take no room */
[data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] style) { display: none !important; }
[data-testid="stForm"] { border: 0 !important; padding: 0 !important; }
[data-testid="stWidgetLabel"] p { color: var(--mist) !important; font-size: .84rem !important; font-weight: 550; }
[data-testid="stMetricValue"] { font-family: var(--digits) !important; font-weight: 500 !important; letter-spacing: -0.02em; }
[data-testid="stMetricLabel"] p { color: var(--mist) !important; }

.cal-day.low { background: rgba(95,203,143,.14) !important; color: var(--ok) !important; }
.cal-day.medium { background: rgba(245,168,60,.16) !important; color: var(--current) !important; }
.cal-day.high { background: rgba(239,106,91,.16) !important; color: var(--alert) !important; }
.cal-day.future { background: rgba(255,255,255,.04) !important; color: var(--mist) !important; }
.cal-day:hover { transform: none !important; }
.cal-header { text-transform: none !important; letter-spacing: 0 !important; color: var(--faint) !important; }
.stApp h3 { font-family: var(--display) !important; font-weight: 600 !important; font-size: 1.12rem !important; letter-spacing: -0.03em; }

.ep-seal { font-size: .84rem; padding: 7px 11px 7px 30px; border-radius: 9px; margin-top: 8px; position: relative;
  border: 1px solid var(--line); color: var(--soft); }
.ep-seal::before { content: ""; position: absolute; left: 11px; top: 50%; width: 9px; height: 9px; margin-top: -4.5px; border-radius: 50%; }
.ep-seal.ok { border-color: rgba(95,203,143,.45); } .ep-seal.ok::before { background: var(--ok); }
.ep-seal.bad { border-color: rgba(239,106,91,.6); color: var(--alert); } .ep-seal.bad::before { background: var(--alert); }

/* ---------- sign-in ---------- */
.ep-hero { padding: 3.5rem 0 0 1rem; }
.ep-hero h1 { font-family: var(--display); font-weight: 650; font-size: clamp(2.4rem, 4.3vw, 4rem);
  line-height: 1.04; letter-spacing: -0.055em; color: var(--plaster); margin: 3.2rem 0 0; padding: 0; }
.ep-hero h1 span { display: block; animation: ep-rise .8s var(--ease) both; }
.ep-hero h1 span:nth-child(2) { animation-delay: .12s; font-weight: 250; }
.ep-hero h1 span:nth-child(3) { animation-delay: .24s; }
.ep-hero p { color: var(--mist); font-size: 1.05rem; max-width: 40ch; margin: 1.6rem 0 0; animation: ep-rise .8s var(--ease) .36s both; }
.ep-pulse { display: block; width: min(460px, 90%); height: auto; margin-top: 2.6rem; }
.ep-pulse path { stroke-width: .9; animation-duration: 2.4s; animation-delay: .5s; }
.ep-form-title { font-family: var(--display); font-weight: 600; font-size: 1.55rem; letter-spacing: -0.04em; line-height: 1.1; color: var(--plaster); margin: 0 0 .3rem; }
.ep-form-sub { color: var(--mist); font-size: .92rem; margin: 0 0 1.2rem; }
.st-key-login_panel { background: var(--slate); border: 1px solid var(--line); border-radius: 20px;
  padding: 2rem 2rem 1.6rem; margin-top: 3rem; animation: ep-rise .7s var(--ease) .2s both; }
.login-divider { color: var(--faint) !important; }
.login-divider::before, .login-divider::after { background: var(--line) !important; }

/* ---------- setup ---------- */
.ep-steps { list-style: none; margin: 2rem 0 0; padding: 0; }
.ep-steps li { display: flex; align-items: center; gap: 14px; padding: 12px 0; color: var(--faint); font-size: .98rem;
  border-top: 1px solid var(--line); }
.ep-steps li:last-child { border-bottom: 1px solid var(--line); }
.ep-steps b { font-family: var(--digits); font-weight: 500; font-size: .85rem; width: 28px; height: 28px; border-radius: 50%;
  border: 1px solid var(--line); display: grid; place-items: center; flex: none; }
.ep-steps li.is-now { color: var(--plaster); }
.ep-steps li.is-now b { border-color: var(--current); color: var(--current); }
.ep-steps li.is-done b { background: var(--ok); border-color: var(--ok); color: #0e1a14; }
.st-key-onboard_panel { background: var(--slate); border: 1px solid var(--line); border-radius: 20px;
  padding: 1.6rem 1.8rem; animation: ep-rise .5s var(--ease) both; }
.ep-row-name { font-size: .95rem; font-weight: 600; color: var(--plaster); }
.ep-row-kind { font-size: .8rem; color: var(--faint); }

@media (max-width: 900px) {
  .block-container { padding: 1.2rem 1rem 3rem !important; }
  .ep-hero { padding: 1rem 0 0; }
  .ep-hero h1 { margin-top: 1.6rem; }
}
"""


# ----------------------------------------------------------------- pieces
def page_header(title: str, lead: str = "", chip: str = "", chip_kind: str = ""):
    chip_html = f"<span class='ep-chip {chip_kind}'><i></i>{chip}</span>" if chip else ""
    lead_html = f"<p>{lead}</p>" if lead else ""
    st.markdown(
        f"<div class='ep-head'><div class='ep-head-row'><div><h1>{title}</h1>{lead_html}</div>"
        f"{chip_html}</div><div class='ep-rule'></div></div>", unsafe_allow_html=True)


def sidebar_nav(pages, current: str) -> str:
    """
    pages: list of (key, label) or None for a divider.  Returns the page to show.
    The pressed button is remembered in st.session_state['page'].
    """
    st.markdown(
        f"<style>section[data-testid='stSidebar'] .st-key-nav_{current} button {{ background: var(--slate-2) !important; "
        f"color: var(--plaster) !important; box-shadow: inset 2px 0 0 var(--fill); }}</style>", unsafe_allow_html=True)
    for item in pages:
        if item is None:
            st.markdown("<div class='ep-nav-gap'></div>", unsafe_allow_html=True)
            continue
        key, label = item
        if st.button(label, key=f"nav_{key}", width="stretch") and key != current:
            st.session_state.page = key
            st.rerun()
    return current


def login_hero(subtitle: str) -> str:
    return (f"<div class='ep-hero'>{wordmark('big')}"
            "<h1><span>One meter.</span><span>Every appliance,</span><span>accounted for.</span></h1>"
            f"<p>{subtitle}.</p>{PULSE_SVG}</div>")


def step_list(labels, step: int) -> str:
    items = []
    for i, label in enumerate(labels, start=1):
        cls = "is-now" if i == step else ("is-done" if i < step else "")
        items.append(f"<li class='{cls}'><b>{i}</b>{label}</li>")
    return f"<ol class='ep-steps'>{''.join(items)}</ol>"
