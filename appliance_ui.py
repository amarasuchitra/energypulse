"""
Appliances, Analysis and Save Energy tabs
=========================================
All three read the same detected meter history (meter_source.py), so the
numbers on one tab always match the others and the Home console.

  Appliances   what was detected: biggest users, long runs, every use, accuracy
  Analysis     where the energy and the money go, by appliance and by tariff period
  Save Energy  realistic suggestions and a per-appliance "what if"
"""

import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from shell_ui import tc

from appliance_profiles import (
    ALWAYS_ON, APPLIANCE_KEYS, CATEGORY_LABELS, DEFAULT_PROFILES, HouseholdPrefs, can_switch_off,
)
from disaggregate import METRICS_PATH
from meter_source import (
    HISTORY_DAYS, SCENARIOS, SOURCE_NOTE, detected_history, get_model, last_days, today_str,
)
from scheduler import recommend
from sessions import extract_sessions, totals
from tariff import Tariff

COLORS = {
    "ac": "#6FA8DC", "geyser": "#F08A5D", "fridge": "#7BC8A4",
    "washing_machine": "#B39DDB", "water_pump": "#4FC3C7",
    "microwave": "#F28FB1", "other": "#55607F",
}
PERIOD_COLORS = {"Normal": "#6F7994", "Solar hours": "#5FCB8F", "Evening peak": "#EF6A5B"}

_model = get_model          # kept for older imports


def _name(key: str) -> str:
    return "Other (lights, fans, TV...)" if key == "other" else DEFAULT_PROFILES[key].name


def meter_settings():
    """Simulated home and tariff mode, chosen on the Settings page; shared by every page."""
    scenario = st.session_state.get("meter_scenario", list(SCENARIOS)[0])
    if scenario not in SCENARIOS:
        scenario = list(SCENARIOS)[0]
    return scenario, bool(st.session_state.get("meter_tod", True))


def render_meter_settings():
    """
    Controls for the simulated home.  The choices are copied into plain
    session keys so they survive when the Settings page is not on screen.
    """
    scenario, tod = meter_settings()
    names = list(SCENARIOS)
    chosen = st.selectbox("Simulated home", names, index=names.index(scenario), key="w_meter_scenario",
                          help="Pick a home with a fault to see alerts and health checks.")
    tod_on = st.toggle("Time-of-day tariff", value=tod, key="w_meter_tod",
                       help="On: 20% cheaper 9 am-5 pm, 20% costlier 6-10 pm. "
                            "Off: same rate all day. Check your own bill.")
    st.session_state["meter_scenario"] = chosen
    st.session_state["meter_tod"] = tod_on


def _history(owned, days):
    scenario, tod = meter_settings()
    pred = detected_history(today_str(), scenario, tuple(owned))
    return last_days(pred, days), tod


# Streamlit re-runs every tab on each refresh, so anything slower than a
# lookup is cached on the inputs that define it.
@st.cache_data(show_spinner=False)
def _sessions_cached(date, scenario, owned, days, rate, tod):
    pred = last_days(detected_history(date, scenario, owned), days)
    return extract_sessions(pred, Tariff(rate=rate, tod_enabled=tod))


@st.cache_data(show_spinner=False)
def _totals_cached(date, scenario, owned, days, rate, tod):
    pred = last_days(detected_history(date, scenario, owned), days)
    tot = totals(pred, Tariff(rate=rate, tod_enabled=tod))
    return tot[tot["appliance"].isin(list(owned) + ["other"])].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _recommend_cached(date, scenario, owned, days, rate, tod, min_saving, dismissed):
    pred = last_days(detected_history(date, scenario, owned), days)
    sessions = _sessions_cached(date, scenario, owned, days, rate, tod)
    prefs = HouseholdPrefs(min_monthly_saving_rs=float(min_saving), dismissed=set(dismissed))
    return recommend(pred, sessions, Tariff(rate=rate, tod_enabled=tod), prefs)


def _keys(owned, days, tariff_rate):
    scenario, tod = meter_settings()
    return (today_str(), scenario, tuple(owned), int(days), float(tariff_rate), tod)


def _span_days(pred) -> float:
    return max(1.0, (pred["datetime"].iloc[-1] - pred["datetime"].iloc[0]).total_seconds() / 86400)


def _stack_chart(pred: pd.DataFrame, layout_fn, owned, height: int = 320) -> go.Figure:
    fig = go.Figure()
    for key in ["other"] + [k for k in APPLIANCE_KEYS if k in owned]:
        fig.add_trace(go.Scatter(
            x=pred["datetime"], y=pred[f"{key}_kw"], name=_name(key),
            mode="lines", line=dict(width=0), stackgroup="one", fillcolor=COLORS[key],
            hovertemplate="%{y:.2f} kW<extra>" + _name(key) + "</extra>"))
    fig.add_trace(go.Scatter(
        x=pred["datetime"], y=pred["mains_kw"], name="Main meter",
        mode="lines", line=dict(color=tc("plaster"), width=1),
        hovertemplate="%{y:.2f} kW<extra>Main meter</extra>"))
    fig.update_layout(**layout_fn(
        height=height, hovermode="x unified", yaxis=dict(title="kW"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)))
    return fig




# ------------------------------------------------------------------ Appliances
def _render_accuracy():
    if not os.path.exists(METRICS_PATH):
        return
    with open(METRICS_PATH) as fh:
        report = json.load(fh)
    with st.expander("How accurate is the detection?"):
        st.caption(report["data"])
        rows = []
        for key in APPLIANCE_KEYS:
            m, b = report["model"][key], report["edge_baseline"][key]
            rows.append({
                "Appliance": _name(key),
                "On/off F1 (model)": m["f1"],
                "On/off F1 (rule baseline)": b["f1"],
                "Power error (W)": m["mae_w"],
                "Energy error (%)": m["energy_error_pct"],
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"Trained on {report['train']}; tested on {report['test']}.")


def _render_usage(pred, tot, section, layout_fn):
    section("Which appliances use the most")
    days = _span_days(pred)
    c_left, c_right = st.columns([3, 2])
    with c_left:
        order = tot.sort_values("kwh")
        fig = go.Figure(go.Bar(
            x=order["kwh"], y=[_name(k) for k in order["appliance"]], orientation="h",
            marker_color=[COLORS[k] for k in order["appliance"]],
            text=[f"{p:.0f}%  ·  Rs. {c:.0f}" for p, c in zip(order["share_pct"], order["cost_rs"])],
            textposition="outside", cliponaxis=False,
            hovertemplate="%{x:.1f} kWh<extra>%{y}</extra>"))
        fig.update_layout(**layout_fn(height=320, xaxis=dict(title=f"kWh in {days:.0f} days"),
                                      margin=dict(l=10, r=110, t=10, b=40)))
        st.plotly_chart(fig, width="stretch", key="appl_usage_chart")
    with c_right:
        st.dataframe(pd.DataFrame({
            "Appliance": [_name(k) for k in tot["appliance"]],
            "Type": [CATEGORY_LABELS[DEFAULT_PROFILES[k].category] if k != "other" else "-"
                     for k in tot["appliance"]],
            "Hours/day": [round(h / days, 1) if k != "other" else None
                          for k, h in zip(tot["appliance"], tot["hours"])],
            "Rs./month": (tot["cost_rs"] / days * 30).round(0),
        }), hide_index=True, width="stretch")
        st.caption("Hours/day is time the appliance was drawing power. For the AC and "
                   "fridge that is compressor time, not the time it was switched on.")


def _render_alerts(result, section):
    section("Running longer than normal")
    if result["alerts"]:
        for a in result["alerts"][:5]:
            st.warning(a["message"])
    else:
        st.success("No appliance ran abnormally long in this period.")


def _render_suggestions(result, owned, section, key_prefix):
    section("What you can realistically change")
    if not result["suggestions"]:
        st.info("Nothing worth changing right now. Small savings are not shown.")
    for s in result["suggestions"]:
        with st.container(border=True):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(f"**{s['title']}**")
                st.write(s["detail"])
                st.caption(f"How this is calculated: {s['calculation']}")
            with c2:
                st.metric("Saves about", f"Rs. {s['monthly_saving_rs']:.0f}")
                st.caption("per month")
                if st.button("Not for us", key=f"{key_prefix}_dismiss_{s['id']}"):
                    st.session_state.setdefault("nilm_dismissed", set()).add(s["id"])
                    st.rerun()
    for note in result["notes"]:
        st.caption(f"• {note}")
    if st.session_state.get("nilm_dismissed"):
        if st.button("Show dismissed suggestions again", key=f"{key_prefix}_undismiss"):
            st.session_state["nilm_dismissed"] = set()
            st.rerun()
    with st.expander("What the system will never suggest"):
        for key in owned:
            prof = DEFAULT_PROFILES[key]
            lock = "" if can_switch_off(prof) else " (locked on)"
            st.markdown(f"**{prof.name}{lock}** — {CATEGORY_LABELS[prof.category]}. {prof.note}")


def _render_day(pred, owned, section, layout_fn):
    section("Main meter and what was detected")
    dates = sorted(pred["datetime"].dt.date.unique())
    day = st.select_slider("Day", options=dates, value=dates[-1],
                           format_func=lambda d: d.strftime("%a %d %b"), key="nilm_day")
    one = pred[pred["datetime"].dt.date == day]
    st.plotly_chart(_stack_chart(one, layout_fn, owned), width="stretch", key="appl_day_chart")
    st.caption("The white line is the only thing measured. The coloured areas are the "
               "model's estimate of what made up that total.")


def _render_sessions(sessions, section):
    section("Every detected use")
    if sessions.empty:
        st.info("No appliance use detected.")
        return
    view = sessions.sort_values("start", ascending=False).head(200)
    st.dataframe(pd.DataFrame({
        "Appliance": [_name(k) for k in view["appliance"]],
        "Started": view["start"].dt.strftime("%d %b %H:%M"),
        "Ran for (min)": view["span_min"],
        "kWh": view["kwh"],
        "Cost (Rs.)": view["cost_rs"],
    }), hide_index=True, width="stretch", height=280)


def _advice(keys, min_saving):
    dismissed = tuple(sorted(st.session_state.get("nilm_dismissed", set())))
    return _sessions_cached(*keys), _recommend_cached(*keys, float(min_saving), dismissed)


def render_appliance_tab(tariff_rate: float, owned, section, layout_fn):
    st.caption(SOURCE_NOTE)
    days = st.radio("Period", [7, 14, HISTORY_DAYS], index=1, horizontal=True,
                    format_func=lambda d: f"Last {d} days", key="nilm_days")
    pred, tod = _history(owned, days)
    tariff = Tariff(rate=float(tariff_rate), tod_enabled=tod)
    keys = _keys(owned, days, tariff_rate)
    sessions, result = _advice(keys, st.session_state.get("save_min_saving", 30))

    _render_usage(pred, _totals_cached(*keys), section, layout_fn)
    _render_alerts(result, section)
    _render_day(pred, owned, section, layout_fn)
    _render_sessions(sessions, section)
    _render_accuracy()
    st.caption("To try switching appliances on yourself, use Test mode on the Home tab.")


# ------------------------------------------------------------------ Analysis
def render_analysis_tab(tariff_rate: float, owned, section, layout_fn):
    st.caption(SOURCE_NOTE)
    days = st.radio("Period", [1, 7, HISTORY_DAYS], index=1, horizontal=True, key="analysis_days",
                    format_func=lambda d: "Today" if d == 1 else f"Last {d} days")
    pred, tod = _history(owned, days)
    tariff = Tariff(rate=float(tariff_rate), tod_enabled=tod)
    tot = _totals_cached(*_keys(owned, days, tariff_rate))

    c_left, c_right = st.columns(2)
    with c_left:
        section("Where your power goes")
        fig = go.Figure(go.Pie(
            labels=[_name(k) for k in tot["appliance"]], values=tot["kwh"], hole=0.58, sort=False,
            marker=dict(colors=[COLORS[k] for k in tot["appliance"]], line=dict(color=tc("ink"), width=2)),
            textinfo="percent", textposition="inside", insidetextorientation="horizontal",
            hovertemplate="%{label}: %{value:.1f} kWh<extra></extra>"))
        fig.update_layout(**layout_fn(
            height=340, showlegend=True, uniformtext=dict(minsize=11, mode="hide"),
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.0),
            margin=dict(l=10, r=10, t=20, b=20),
            annotations=[dict(text=f"{tot['kwh'].sum():.1f}<br>kWh", x=0.5, y=0.5,
                              font=dict(size=16, color=tc("plaster")), showarrow=False)]))
        st.plotly_chart(fig, width="stretch", key="analysis_pie")
    with c_right:
        section("What it costs, by time of day")
        dts = pred["datetime"]
        minute = (dts.dt.hour * 60 + dts.dt.minute).to_numpy()
        rate = tariff.minute_rates()[minute]
        label = [tariff.label_at(m / 60) for m in range(1440)]
        frame = pd.DataFrame({"period": [label[m] for m in minute],
                              "kwh": pred["mains_kw"].to_numpy() / 60,
                              "cost": pred["mains_kw"].to_numpy() / 60 * rate})
        by = frame.groupby("period", as_index=False)[["kwh", "cost"]].sum()
        fig = go.Figure(go.Bar(
            x=by["period"], y=by["cost"], marker_color=[PERIOD_COLORS.get(p, "#6F7994") for p in by["period"]],
            text=[f"Rs. {c:,.0f}" for c in by["cost"]], textposition="outside", cliponaxis=False,
            hovertemplate="%{x}: Rs. %{y:.0f}<extra></extra>"))
        fig.update_layout(**layout_fn(height=340, yaxis=dict(title="Rs."), margin=dict(l=40, r=20, t=30, b=40)))
        st.plotly_chart(fig, width="stretch", key="analysis_period_chart")
        if not tod:
            st.caption("Your tariff is flat, so every hour costs the same per unit.")

    section("Appliance by appliance")
    span = _span_days(pred)
    st.dataframe(pd.DataFrame({
        "Appliance": [_name(k) for k in tot["appliance"]],
        "Units (kWh)": tot["kwh"].round(1),
        "Share (%)": tot["share_pct"],
        "Cost (Rs.)": tot["cost_rs"].round(0),
        "Hours drawing power": [h if k != "other" else None for k, h in zip(tot["appliance"], tot["hours"])],
        "Projected Rs./month": (tot["cost_rs"] / span * 30).round(0),
    }), hide_index=True, width="stretch")


# ------------------------------------------------------------------ Save Energy
def render_save_tab(tariff_rate: float, owned, section, layout_fn):
    st.caption(SOURCE_NOTE)
    min_saving = st.number_input("Hide tips saving less than (Rs./month)", 0, 500, 30, 10,
                                 key="save_min_saving")
    pred, tod = _history(owned, HISTORY_DAYS)
    tariff = Tariff(rate=float(tariff_rate), tod_enabled=tod)
    sessions, result = _advice(_keys(owned, HISTORY_DAYS, tariff_rate), min_saving)

    _render_suggestions(result, owned, section, "save")
    _render_alerts(result, section)

    section("What if you used one appliance less?")
    choices = [k for k in owned if DEFAULT_PROFILES[k].category != ALWAYS_ON
               and not sessions[sessions["appliance"] == k].empty]
    if not choices:
        st.info("No appliance with adjustable use was detected in this period.")
        return
    c1, c2 = st.columns([2, 3])
    key = c1.selectbox("Appliance", choices, format_func=_name, key="whatif_appliance")
    group = sessions[sessions["appliance"] == key]
    days = _span_days(pred)
    minutes_per_day = group["on_min"].sum() / days
    cut = c2.slider("Minutes less per day", 0, int(max(5, minutes_per_day)), 0, 5, key="whatif_minutes")
    avg_kw = group["kwh"].sum() / max(group["on_min"].sum() / 60.0, 1e-9)
    avg_rate = group["cost_rs"].sum() / max(group["kwh"].sum(), 1e-9)
    now_month = group["cost_rs"].sum() / days * 30
    saving = avg_kw * (cut / 60.0) * avg_rate * 30
    m1, m2, m3 = st.columns(3)
    m1.metric(f"{_name(key)} now, per month", f"Rs. {now_month:,.0f}")
    m2.metric("After the change, per month", f"Rs. {max(0, now_month - saving):,.0f}")
    m3.metric("You would save, per month", f"Rs. {saving:,.0f}")
    st.caption(f"How this is calculated: {avg_kw:.2f} kW while running x {cut} min a day "
               f"x Rs. {avg_rate:.2f} per unit (the rate at the hours you use it) x 30 days. "
               f"It runs about {minutes_per_day:.0f} min a day now.")
    if DEFAULT_PROFILES[key].category == "comfort":
        st.caption("For the AC this counts compressor minutes, which also fall when the set temperature is raised.")


# ------------------------------------------------------------------ report
def report_body(tariff_rate: float, owned, household_name: str) -> str:
    """Plain-text summary of the last 30 days; integrity.sign_report() seals it."""
    keys = _keys(owned, HISTORY_DAYS, tariff_rate)
    tot = _totals_cached(*keys)
    lines = [
        "EnergyPulse report",
        f"Household: {household_name}",
        f"Period: last {HISTORY_DAYS} days ending {keys[0]}",
        f"Data: simulated main-meter feed ({keys[1]})",
        f"Tariff: Rs. {tariff_rate:.2f} per unit, time-of-day {'on' if keys[5] else 'off'}",
        "",
        f"{'Appliance':<32}{'Units (kWh)':>12}{'Share %':>9}{'Cost (Rs.)':>12}",
    ]
    for _, row in tot.iterrows():
        lines.append(f"{_name(row['appliance']):<32}{row['kwh']:>12.1f}{row['share_pct']:>9.1f}{row['cost_rs']:>12.0f}")
    lines.append(f"{'Total':<32}{tot['kwh'].sum():>12.1f}{100.0:>9.1f}{tot['cost_rs'].sum():>12.0f}")
    return "\n".join(lines)
