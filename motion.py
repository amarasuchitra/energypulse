"""
Motion system
=============
One place for how the site moves, so every page behaves the same way.

  MOTION_CSS      timing tokens, micro-interactions, reveals, skeletons, toasts, modals
  runtime(page)   a small script that runs in the page: reveals content as it
                  scrolls into view, counts numbers up, shows toasts
  notify(kind, text)   queue a toast: "success", "warning", "error" or "info"
  skeleton(kind)  placeholder shapes shown while a page's data is prepared

Rules it follows
----------------
- Three durations (120 / 200 / 380 ms, plus 600 ms for entrances) and one
  easing curve.  Nothing loops for decoration.
- Movement is small: 12-14 px rises, 3 px lifts, no bouncing.
- Everything is disabled under prefers-reduced-motion, and shortened on
  low-powered devices.
- It only adds classes and styles.  It never changes what a control does.
"""

import json
import uuid

import streamlit as st
import streamlit.components.v1 as components

MOTION_CSS = r"""
:root { --d1: 120ms; --d2: 200ms; --d3: 380ms; --d4: 600ms;
  --ease: cubic-bezier(.2, .7, .2, 1); --ease-io: cubic-bezier(.45, 0, .2, 1);
  --lift: 0 14px 30px rgba(6, 10, 24, .28); }

/* ---------- content arriving ---------- */
.ep-reveal { opacity: 0; transform: translateY(14px); }
.ep-reveal.ep-in { opacity: 1; transform: none;
  transition: opacity var(--d3) var(--ease) var(--stagger, 0ms), transform var(--d4) var(--ease) var(--stagger, 0ms); }
@keyframes ep-chart { from { clip-path: inset(0 100% 0 0); opacity: .35; } to { clip-path: inset(0 0 0 0); opacity: 1; } }
.stPlotlyChart .js-plotly-plot { animation: ep-chart 900ms var(--ease) both; }
/* a re-run dims content only if it takes long enough to notice */
.stElementContainer[data-stale="true"], [data-testid="stElementContainer"][data-stale="true"] {
  opacity: .62 !important; transition: opacity var(--d3) ease .4s !important; }

/* the script carrier takes no room */
[data-testid="stElementContainer"]:has(iframe[title="st.iframe"]), .stElementContainer:has(iframe[height="0"]) {
  position: absolute !important; height: 0 !important; overflow: hidden; }

/* ---------- cards ---------- */
.metric-card, .ep-stat, .tip-box { transition: transform var(--d2) var(--ease), box-shadow var(--d2) ease, border-color var(--d2) ease !important; }
.metric-card:hover, .ep-stat:hover { transform: translateY(-3px) !important; box-shadow: var(--lift) !important; }
[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stExpander"] details { transition: border-color var(--d2) ease, box-shadow var(--d2) ease; }
[data-testid="stVerticalBlockBorderWrapper"]:hover, [data-testid="stExpander"] details:hover { border-color: var(--faint) !important; }
[data-testid="stExpander"] summary svg { transition: transform var(--d2) var(--ease); }
[data-testid="stExpanderDetails"] { animation: ep-rise var(--d3) var(--ease) both; }

/* ---------- buttons ---------- */
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
  transition: background var(--d2) ease, border-color var(--d2) ease, box-shadow var(--d2) ease, transform var(--d1) var(--ease) !important; }
.stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover {
  transform: translateY(-1px) !important; box-shadow: 0 6px 16px rgba(6, 10, 24, .18) !important; }
.stButton > button:active, .stFormSubmitButton > button:active, .stDownloadButton > button:active {
  transform: translateY(0) scale(.975) !important; box-shadow: none !important; transition-duration: 60ms !important; }
button[data-testid="stBaseButton-primary"]:hover, button[data-testid="stBaseButton-primaryFormSubmit"]:hover {
  box-shadow: 0 8px 22px rgba(245, 168, 60, .32) !important; }
.stButton > button:focus-visible, .stFormSubmitButton > button:focus-visible { outline: 2px solid var(--fill) !important; outline-offset: 2px; }

/* ---------- navigation ---------- */
section[data-testid="stSidebar"] [class*="st-key-nav_"] button { transition: background var(--d2) ease, color var(--d2) ease, box-shadow var(--d2) ease, transform var(--d2) var(--ease) !important; }
section[data-testid="stSidebar"] [class*="st-key-nav_"] button:hover { transform: translateX(3px) !important; box-shadow: none !important; }
section[data-testid="stSidebar"] [class*="st-key-nav_"] button:active { transform: translateX(1px) scale(.985) !important; }
section[data-testid="stSidebar"] [class*="st-key-nav_"] button::before { transition: transform var(--d2) var(--ease), background var(--d2) ease; }
section[data-testid="stSidebar"] [class*="st-key-nav_"] button:hover::before { transform: scale(1.12); }
@keyframes ep-nav-on { from { box-shadow: inset 0 0 0 var(--fill); } to { box-shadow: inset 2px 0 0 var(--fill); } }
section[data-testid="stSidebar"] { transition: transform var(--d3) var(--ease), min-width var(--d3) var(--ease) !important; }

/* ---------- inputs, selects, toggles ---------- */
[data-testid="stTextInputRootElement"], [data-testid="stNumberInputContainer"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {
  transition: border-color var(--d2) ease, box-shadow var(--d2) ease, background var(--d2) ease !important; }
[data-testid="stTextInputRootElement"]:focus-within, [data-testid="stNumberInputContainer"]:focus-within,
[data-baseweb="select"] > div:focus-within, [data-baseweb="textarea"]:focus-within {
  border-color: var(--fill) !important; box-shadow: 0 0 0 3px rgba(245, 168, 60, .18) !important; }
@keyframes ep-pop { from { opacity: 0; transform: translateY(-6px) scale(.98); } to { opacity: 1; transform: none; } }
[data-baseweb="popover"] > div { animation: ep-pop var(--d2) var(--ease) both; transform-origin: top center; }
[data-baseweb="menu"] li { transition: background var(--d1) ease; }
[data-testid="stCheckbox"] label > div:first-child, [data-baseweb="checkbox"] > div { transition: background var(--d2) ease, border-color var(--d2) ease; }
[data-baseweb="checkbox"] > div > div { transition: transform var(--d2) var(--ease), background var(--d2) ease !important; }
[data-baseweb="slider"] [role="slider"] { transition: transform var(--d1) var(--ease), box-shadow var(--d2) ease; }
[data-baseweb="slider"] [role="slider"]:hover { transform: scale(1.15); box-shadow: 0 0 0 6px rgba(245, 168, 60, .16); }

/* ---------- messages ---------- */
[data-testid="stAlert"] { animation: ep-rise var(--d3) var(--ease) both; }
@keyframes ep-shake { 0%, 100% { transform: none; } 25% { transform: translateX(-5px); } 60% { transform: translateX(4px); } 85% { transform: translateX(-2px); } }
[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) { animation: ep-rise var(--d2) var(--ease) both, ep-shake var(--d3) var(--ease-io) var(--d2); border-color: rgba(239, 106, 91, .55) !important; }
[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) { border-color: rgba(95, 203, 143, .45) !important; }
.ep-seal { animation: ep-rise var(--d3) var(--ease) both; }

/* ---------- modal ---------- */
@keyframes ep-modal { from { opacity: 0; transform: translateY(10px) scale(.96); } to { opacity: 1; transform: none; } }
@keyframes ep-fade-in { from { opacity: 0; } to { opacity: 1; } }
[data-testid="stDialog"] { animation: ep-fade-in var(--d2) ease both; backdrop-filter: blur(7px); -webkit-backdrop-filter: blur(7px);
  background: rgba(8, 11, 22, .45) !important; }
[data-testid="stDialog"] > div { animation: ep-modal var(--d3) var(--ease) both; border: 1px solid var(--line);
  border-radius: 20px !important; background: var(--slate) !important; box-shadow: 0 30px 80px rgba(6, 10, 24, .5); }
[data-testid="stDialog"] [role="dialog"] h2, [data-testid="stDialog"] [role="dialog"] header { font-family: var(--display) !important; letter-spacing: -0.03em; }

/* ---------- loading shapes ---------- */
@keyframes ep-shimmer { from { background-position: 180% 0; } to { background-position: -80% 0; } }
.ep-skel { display: grid; gap: 14px; animation: ep-fade-in var(--d3) ease both; }
.ep-skel i { display: block; border-radius: 16px; background-color: var(--slate);
  background-image: linear-gradient(100deg, transparent 30%, var(--slate-2) 50%, transparent 70%);
  background-size: 220% 100%; animation: ep-shimmer 1.5s linear infinite; border: 1px solid var(--line); }
.ep-skel .row { display: grid; gap: 14px; }
.ep-skel p { margin: 0; color: var(--faint); font-size: .84rem; }

/* ---------- empty and error states ---------- */
@keyframes ep-breathe { 0%, 100% { opacity: .35; } 50% { opacity: .8; } }
.empty-state { text-align: center; padding: 2.6rem 1rem !important; border: 1px dashed var(--line) !important; border-radius: 16px !important;
  background: transparent !important; color: var(--mist) !important; animation: ep-rise var(--d3) var(--ease) both; }
.empty-state::before { content: ""; display: block; width: 120px; height: 28px; margin: 0 auto 14px; background: var(--faint);
  -webkit-mask: var(--pulse-mask) center / contain no-repeat; mask: var(--pulse-mask) center / contain no-repeat;
  animation: ep-breathe 3.2s ease-in-out infinite; }
.error-card { animation: ep-rise var(--d3) var(--ease) both; border-left: 3px solid var(--alert) !important; }

/* ---------- toasts ---------- */
.ep-toasts { position: fixed; right: 22px; bottom: 22px; z-index: 1000000; display: flex; flex-direction: column; gap: 10px;
  width: min(380px, calc(100vw - 32px)); pointer-events: none; font-family: "Hanken Grotesk", "Segoe UI", system-ui, sans-serif; }
.ep-toast { pointer-events: auto; display: grid; grid-template-columns: 22px 1fr; gap: 11px; align-items: start;
  background: var(--slate); color: var(--plaster); border: 1px solid var(--line); border-left: 3px solid var(--tone);
  border-radius: 14px; padding: 12px 14px; font-size: .9rem; line-height: 1.4; cursor: pointer;
  box-shadow: 0 18px 40px rgba(6, 10, 24, .34); position: relative; overflow: hidden;
  opacity: 0; transform: translateX(28px); transition: opacity var(--d3) var(--ease), transform var(--d3) var(--ease); }
.ep-toast.in { opacity: 1; transform: none; }
.ep-toast.out { opacity: 0; transform: translateX(28px) scale(.98); transition-duration: var(--d2); }
.ep-toast svg { width: 20px; height: 20px; color: var(--tone); margin-top: 1px; }
.ep-toast::after { content: ""; position: absolute; left: 0; bottom: 0; height: 2px; width: 100%; background: var(--tone); opacity: .5;
  transform-origin: left; animation: ep-toast-life var(--life, 4600ms) linear forwards; }
@keyframes ep-toast-life { from { transform: scaleX(1); } to { transform: scaleX(0); } }
.ep-toast.success { --tone: var(--ok); } .ep-toast.warning { --tone: var(--fill); }
.ep-toast.error { --tone: var(--alert); } .ep-toast.info { --tone: #6fa8dc; }

/* ---------- phones and tablets ---------- */
@media (max-width: 900px) {
  .ep-toasts { right: 16px; left: 16px; bottom: 16px; width: auto; }
  .metric-card:hover, .ep-stat:hover { transform: none !important; box-shadow: none !important; }
  .block-container { padding-top: 3.6rem !important; }      /* room for the menu button */
  .metric-card { margin-bottom: 12px; min-height: 0 !important; }
  header[data-testid="stHeader"] button { background: var(--slate) !important; border: 1px solid var(--line) !important; border-radius: 10px !important; }
}
/* ---------- less motion where it is not wanted or not affordable ---------- */
html.ep-lite .ep-reveal { opacity: 1 !important; transform: none !important; }
html.ep-lite .stPlotlyChart .js-plotly-plot, html.ep-lite .ep-skel i { animation: none !important; }
html.ep-lite [data-testid="stDialog"] { backdrop-filter: none; -webkit-backdrop-filter: none; }
@media (prefers-reduced-motion: reduce) {
  .ep-reveal { opacity: 1 !important; transform: none !important; }
  .ep-toast { transition: opacity var(--d2) ease !important; transform: none !important; }
}
"""

_ICONS = {
    "success": "<path d='M5 12.5l4.5 4.5L19 7.5'/>",
    "warning": "<path d='M12 4l9 16H3z'/><path d='M12 10v4M12 17.3v.2'/>",
    "error": "<circle cx='12' cy='12' r='9'/><path d='M9 9l6 6M15 9l-6 6'/>",
    "info": "<circle cx='12' cy='12' r='9'/><path d='M12 11v5M12 7.6v.2'/>",
}

# Installed once into the page itself (not the small frame that delivers it),
# so its timers and observers survive when that frame is replaced.
_INSTALL_JS = r"""
(() => {
  if (window.__ep) return;
  const D = document, ep = window.__ep = { seen: new Set(), page: null, navAt: 0, nums: new WeakMap(), icons: {} };
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const lite = (navigator.hardwareConcurrency || 8) <= 2 || (navigator.deviceMemory || 8) <= 2;
  ep.still = reduce || lite;
  if (lite) D.documentElement.classList.add("ep-lite");

  // ---- toasts
  const tray = D.createElement("div"); tray.className = "ep-toasts"; tray.setAttribute("aria-live", "polite");
  D.body.appendChild(tray);
  const close = (el) => { if (el.dataset.out) return; el.dataset.out = 1; el.classList.add("out"); setTimeout(() => el.remove(), 240); };
  ep.toast = (kind, text, life = 4600) => {
    const el = D.createElement("div");
    el.className = `ep-toast ${kind}`; el.setAttribute("role", kind === "error" ? "alert" : "status");
    el.style.setProperty("--life", `${life}ms`);
    el.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ep.icons[kind] || ep.icons.info || ""}</svg><span></span>`;
    el.querySelector("span").textContent = text;
    el.addEventListener("click", () => close(el));
    tray.appendChild(el);
    while (tray.children.length > 4) tray.firstChild.remove();
    requestAnimationFrame(() => requestAnimationFrame(() => el.classList.add("in")));
    setTimeout(() => close(el), life);
  };

  // ---- content rises into view, cards in a row one after another
  const REVEAL = ".metric-card, .ep-stat, [data-testid='stVerticalBlockBorderWrapper'], .stPlotlyChart, .section-header, " +
                 ".tip-box, [data-testid='stDataFrame'], [data-testid='stExpander'], .empty-state, iframe[title='home_ui.energy_home']";
  const settle = (el) => { el.classList.remove("ep-reveal", "ep-in"); el.style.removeProperty("--stagger"); };
  const io = new IntersectionObserver((entries) => {
    let i = 0;
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      const el = e.target; io.unobserve(el);
      el.style.setProperty("--stagger", `${Math.min(i++, 6) * 70}ms`);
      el.classList.add("ep-in");
      setTimeout(() => settle(el), 1300);
    }
  }, { rootMargin: "0px 0px -6% 0px", threshold: 0.05 });
  ep.reveal = () => {
    if (ep.still) return;
    const entering = performance.now() - ep.navAt < 1600;
    D.querySelectorAll(REVEAL).forEach((el) => {
      if (el.dataset.epSeen) return;
      el.dataset.epSeen = 1;
      const below = el.getBoundingClientRect().top > innerHeight * 0.92;
      // Only content that is arriving (a new page) or still below the fold is animated;
      // something already on screen is never hidden and shown again.
      if (!entering && !below) return;
      el.classList.add("ep-reveal"); io.observe(el);
      // Never leave anything hidden: content on screen is shown within a moment regardless.
      setTimeout(() => { if (el.getBoundingClientRect().top < innerHeight) { io.unobserve(el); el.classList.add("ep-in"); setTimeout(() => settle(el), 1300); } }, 1800);
    });
  };

  // ---- numbers count up, and ease to a new value when it changes
  const NUM = ".mc-value, .ep-stat strong, [data-testid='stMetricValue'] > div";
  const firstText = (el) => { for (const n of el.childNodes) if (n.nodeType === 3 && /\d/.test(n.nodeValue)) return n; return null; };
  ep.count = () => {
    D.querySelectorAll(NUM).forEach((el) => {
      const node = firstText(el); if (!node) return;
      const st = ep.nums.get(el), text = node.nodeValue;
      if (st && text === st.written) return;                  // our own write
      const m = text.match(/-?\d[\d,]*\.?\d*/); if (!m) return;
      const target = parseFloat(m[0].replace(/,/g, "")); if (!isFinite(target)) return;
      const decimals = (m[0].split(".")[1] || "").length, grouped = m[0].includes(",");
      const pre = text.slice(0, m.index), post = text.slice(m.index + m[0].length);
      const fmt = (v) => pre + (grouped ? v.toLocaleString("en-IN", { minimumFractionDigits: decimals, maximumFractionDigits: decimals }) : v.toFixed(decimals)) + post;
      const from = st ? st.value : 0;
      const state = { value: target, written: fmt(target) };
      ep.nums.set(el, state);
      if (ep.still || from === target) { node.nodeValue = state.written; return; }
      const t0 = performance.now(), dur = st ? 520 : 820;
      const step = (now) => {
        if (ep.nums.get(el) !== state || !node.isConnected) return;
        const k = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - k, 3);
        state.written = fmt(from + (target - from) * e); node.nodeValue = state.written;
        if (k < 1) requestAnimationFrame(step); else { state.written = fmt(target); node.nodeValue = state.written; }
      };
      requestAnimationFrame(step);
    });
  };

  let timer = null;
  const scan = () => { timer = null; ep.reveal(); ep.count(); };
  new MutationObserver(() => { if (!timer) timer = setTimeout(scan, 40); })
    .observe(D.body, { childList: true, subtree: true, characterData: true });

  ep.update = (data) => {
    ep.icons = data.icons;
    if (ep.page !== data.page) { ep.page = data.page; ep.navAt = performance.now(); }
    scan();
    for (const t of data.toasts) { if (!ep.seen.has(t.id)) { ep.seen.add(t.id); ep.toast(t.kind, t.text); } }
  };
})();
"""

_CARRIER_JS = r"""
(() => {
  const W = window.parent, D = W.document;
  if (!W.__ep) { const s = D.createElement("script"); s.textContent = __CODE__; D.head.appendChild(s); }
  W.__ep.update(__DATA__);
})();
"""


def notify(kind: str, text: str) -> None:
    """Queue a toast.  kind: success | warning | error | info."""
    if kind not in _ICONS:
        kind = "info"
    queue = st.session_state.setdefault("_toasts", [])
    queue.append({"id": uuid.uuid4().hex, "kind": kind, "text": str(text)[:240]})
    del queue[:-6]


def runtime(page: str) -> None:
    """Render once per run, anywhere on the page.  It takes no room."""
    data = {"page": page, "toasts": st.session_state.get("_toasts", []), "icons": _ICONS}
    safe = lambda text: json.dumps(text).replace("</", "<\\/")
    js = _CARRIER_JS.replace("__CODE__", safe(_INSTALL_JS)).replace("__DATA__", safe(data))
    components.html(f"<script>{js}</script>", height=0)


def skeleton(kind: str = "page", note: str = "") -> str:
    """Placeholder shapes for a page whose data is still being prepared."""
    if kind == "console":
        body = ("<div class='row' style='grid-template-columns: minmax(0,1fr) 340px'>"
                "<i style='height:620px'></i>"
                "<div class='row'><i style='height:200px'></i><i style='height:250px'></i><i style='height:142px'></i></div></div>")
    elif kind == "cards":
        body = ("<div class='row' style='grid-template-columns: repeat(4, 1fr)'>" + "<i style='height:128px'></i>" * 4 + "</div>"
                "<div class='row' style='grid-template-columns: 2.1fr 1fr'><i style='height:330px'></i><i style='height:330px'></i></div>")
    else:
        body = ("<div class='row' style='grid-template-columns: 3fr 2fr'><i style='height:300px'></i><i style='height:300px'></i></div>"
                "<i style='height:64px'></i><i style='height:260px'></i>")
    caption = f"<p>{note}</p>" if note else ""
    return f"<div class='ep-skel' aria-busy='true' aria-label='Loading'>{body}{caption}</div>"
