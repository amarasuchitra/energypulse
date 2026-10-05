"""
Pages for the household toolkit (the sums are in toolkit.py).

Each render_* function takes one context dict built in app.py:
    pred, owned, switched, tariff, rate, db, household_id, user_name, people,
    section, metric_card, layout, ledger, brief
"""

from urllib.parse import quote

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import toolkit as tk
from appliance_profiles import DEFAULT_PROFILES
from daily_brief import _daily
from meter_sim import simulate_home
from meter_source import now_minute
from motion import notify
from sessions import extract_sessions, flag_long_runs
from shell_ui import current_theme, tc as _tc
from clock import local_now, to_local

_EXTRA = {"dark": {"alert": "#EF6A5B", "current": "#F5A83C"}, "light": {"alert": "#CF4630", "current": "#B96D05"}}


def tc(name: str) -> str:
    return _EXTRA[current_theme()].get(name) or _tc(name)

DEFAULTS = {"state": "Karnataka (BESCOM, domestic)", "sanctioned_kw": 3.0, "goal": 0.0,
            "schedules": {}, "co2": tk.GRID_KG_CO2_PER_KWH}


def _hhmm(minute: int) -> str:
    return f"{int(minute) // 60 % 24:02d}:{int(minute) % 60:02d}"


def settings(ctx) -> dict:
    return {**DEFAULTS, **(ctx["db"].get_store(ctx["household_id"], "toolkit", {}) or {})}


def save_settings(ctx, **changes) -> None:
    current = settings(ctx)
    if any(current.get(k) != v for k, v in changes.items()):
        ctx["db"].set_store(ctx["household_id"], "toolkit", {**current, **changes})


def _note(text: str) -> None:
    st.markdown(f"<p style='color:var(--mist);font-size:.84rem;margin:.2rem 0 .6rem'>{text}</p>",
                unsafe_allow_html=True)


def _rows(rows) -> None:
    """rows: (name, small text, right value, right small) tuples, drawn as the app's quiet table."""
    html = "".join(
        f"<tr><td class='n'>{a}<small>{b}</small></td><td class='v'>{c}<small>{d}</small></td></tr>"
        for a, b, c, d in rows)
    st.markdown(f"<table class='ep-rows'>{html}</table>", unsafe_allow_html=True)


def _status(ok: bool, text: str) -> None:
    st.markdown(f"<div class='ep-seal {'ok' if ok else 'bad'}'>{text}</div>", unsafe_allow_html=True)


def base(ctx) -> dict:
    """Daily units and cost, sessions and month totals, worked out once per page run."""
    if "_base" not in ctx:
        pred, owned, tariff = ctx["pred"], ctx["owned"], ctx["tariff"]
        kwh, cost = _daily(pred, owned, tariff)
        today = pd.to_datetime(pred["datetime"]).dt.normalize().iloc[-1]
        history = pred[pd.to_datetime(pred["datetime"]).dt.normalize() < today]
        days = max(1.0, len(history) / 1440.0)
        ctx["_base"] = {
            "kwh": kwh, "cost": cost, "history": history, "sessions": extract_sessions(history, tariff),
            # today is still in progress: only the minutes that have happened
            "today": pred[pd.to_datetime(pred["datetime"]).dt.normalize() == today].iloc[: now_minute() + 1],
            "month_kwh": {k: float(history[f"{k}_kw"].sum() / 60.0 / days * 30.0) for k in owned},
            "avg_rate": float(cost["total"].iloc[:-1].sum() / max(kwh["total"].iloc[:-1].sum(), 1e-9)),
        }
    return ctx["_base"]


# ====================================================================== bills
def _tariff_editor(ctx) -> dict:
    cfg = settings(ctx)
    names = list(tk.TARIFF_PRESETS)
    c1, c2 = st.columns([1.4, 1])
    state = c1.selectbox("Tariff", names, index=names.index(cfg["state"]) if cfg["state"] in names else 0,
                         key="tk_state")
    sanctioned = c2.number_input("Sanctioned load (kW)", 0.5, 50.0, float(cfg["sanctioned_kw"]), 0.5,
                                 key="tk_sanctioned", help="Printed on your bill as sanctioned or connected load.")
    save_settings(ctx, state=state, sanctioned_kw=float(sanctioned))
    preset = tk.TARIFF_PRESETS[state]
    with st.expander("Rates used (edit to match your bill)"):
        frame = pd.DataFrame([{"Up to units (blank = all above)": u, "Rs per unit": r} for u, r in preset["slabs"]])
        edited = st.data_editor(frame, num_rows="dynamic", key=f"tk_slabs_{state}", hide_index=True,
                                width="stretch")
        c3, c4, c5 = st.columns(3)
        fixed = c3.number_input("Fixed charge (Rs per kW)", 0.0, 2000.0,
                                float(tk.fixed_charge(sanctioned, preset["fixed"]) / max(sanctioned, 1e-9)), 5.0,
                                key=f"tk_fixed_{state}_{sanctioned}")
        duty = c4.number_input("Electricity duty or tax (%)", 0.0, 40.0, float(preset["duty_pct"]), 0.5,
                               key=f"tk_duty_{state}")
        surcharge = c5.number_input("Fuel or PPAC surcharge (%)", 0.0, 80.0, float(preset["surcharge_pct"]), 0.5,
                                    key=f"tk_sur_{state}")
        st.caption(preset["note"] + " These rates are a starting point; tariff orders change every year.")
    slabs = []
    for _, row in edited.iterrows():
        rate = pd.to_numeric(row.iloc[1], errors="coerce")
        if pd.isna(rate):
            continue
        upto = pd.to_numeric(row.iloc[0], errors="coerce")
        slabs.append((None if pd.isna(upto) else float(upto), float(rate)))
    slabs.sort(key=lambda s: float("inf") if s[0] is None else s[0])
    if not slabs:
        slabs = list(preset["slabs"])
    if slabs[-1][0] is not None:
        slabs.append((None, slabs[-1][1]))
    return {"slabs": slabs, "fixed": [(None, float(fixed))], "duty_pct": float(duty),
            "surcharge_pct": float(surcharge), "sanctioned_kw": float(sanctioned), "name": state}


def render_bill_tools(ctx, saved_bills=None) -> None:
    section, b = ctx["section"], base(ctx)
    section("Check, watch and explain your bill")
    with st.container(border=True):
        tariff = _tariff_editor(ctx)
    tab_check, tab_slab, tab_why = st.tabs(["Is my bill right?", "Slab watch", "Why it changed"])

    with tab_check:
        _note("Type the units and the amount from a bill. The bill is worked out again from the rates above "
              "and compared with what you were charged.")
        prefill_units, prefill_amount = 0.0, 0.0
        options = [x for x in (saved_bills or []) if x.get("details", {}).get("units")]
        if options:
            pick = st.selectbox("Use a saved bill", ["Type it myself"] + [x["name"] for x in options], key="tk_bill_pick")
            chosen = next((x for x in options if x["name"] == pick), None)
            if chosen:
                to_num = lambda v: float(str(v).replace(",", "") or 0) if v else 0.0
                prefill_units, prefill_amount = to_num(chosen["details"].get("units")), to_num(chosen["details"].get("amount"))
        c1, c2, c3 = st.columns(3)
        units = c1.number_input("Units on the bill (kWh)", 0.0, 20000.0, float(prefill_units), 1.0, key=f"tk_units_{prefill_units}")
        billed = c2.number_input("Amount charged (Rs)", 0.0, 500000.0, float(prefill_amount), 10.0, key=f"tk_billed_{prefill_amount}")
        other = c3.number_input("Other charges on the bill (Rs)", 0.0, 50000.0, 0.0, 10.0, key="tk_other",
                                help="Meter rent, arrears, late fee: anything not based on units.")
        if units > 0:
            res = tk.bill_check(units, tariff, tariff["sanctioned_kw"], billed or None, other)
            rows = [(f"Units {l['from']:.0f} to {l['to']:.0f}", f"{l['units']:.0f} units at Rs. {l['rate']:.2f}",
                     f"Rs. {l['amount']:.2f}", "") for l in res["lines"]]
            rows += [("Fixed charge", f"{tariff['sanctioned_kw']:.1f} kW sanctioned load", f"Rs. {res['fixed']:.2f}", "")]
            if res["surcharge"]:
                rows.append(("Fuel or PPAC surcharge", f"{tariff['surcharge_pct']:.1f}% of energy and fixed charges",
                             f"Rs. {res['surcharge']:.2f}", ""))
            rows.append(("Electricity duty or tax", f"{tariff['duty_pct']:.1f}% of energy charges", f"Rs. {res['duty']:.2f}", ""))
            if res["other"]:
                rows.append(("Other charges", "as entered", f"Rs. {res['other']:.2f}", ""))
            rows.append(("Worked-out total", tariff["name"], f"Rs. {res['total']:.2f}", ""))
            _rows(rows)
            if res["verdict"] == "matches":
                _status(True, f"Your bill of Rs. {billed:.0f} matches the worked-out total within 3%.")
            elif res["verdict"]:
                _status(False, f"You were charged Rs. {abs(res['difference']):.0f} "
                               f"({abs(res['difference_pct']):.0f}%) {'more' if res['difference'] > 0 else 'less'} than the "
                               f"worked-out total. Check the rates above against your bill first; if they are right, "
                               f"the Help page drafts a complaint with these figures.")
                st.session_state["tk_last_check"] = {"units": units, "billed": billed, "total": res["total"],
                                                     "difference": res["difference"], "tariff": tariff["name"]}
            else:
                st.caption("Enter the amount charged to compare.")

    with tab_slab:
        pos = tk.month_position(b["kwh"]["total"])
        warn = tk.slab_warning(pos, tariff["slabs"])
        c1, c2, c3 = st.columns(3)
        ctx["metric_card"](c1, label="Used this month", value=f"{pos['used']:.0f}", unit="units",
                           sub=f"{pos['days_done']} day(s) recorded")
        ctx["metric_card"](c2, label="Projected for the month", value=f"{pos['projected']:.0f}", unit="units",
                           sub=f"at {pos['pace']:.1f} units a day")
        est = tk.bill_check(pos["projected"], tariff, tariff["sanctioned_kw"])
        ctx["metric_card"](c3, label="Projected bill", value=f"Rs. {est['total']:.0f}",
                           sub=f"energy Rs. {est['energy']:.0f}, fixed Rs. {est['fixed']:.0f}, duty Rs. {est['duty']:.0f}")
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        if warn["flat"]:
            _status(True, f"This tariff charges one rate (Rs. {warn['rate_now']:.2f}) for every unit, so there is no slab to cross.")
        elif warn["boundary"] is None:
            _status(True, "You are already in the highest slab; every further unit costs "
                          f"Rs. {warn['rate_end']:.2f}.")
        elif warn["will_cross"]:
            _status(False, f"At this pace you cross {warn['boundary']:.0f} units around day {warn['cross_day']}. "
                           f"Units beyond that cost Rs. {warn['next_rate']:.2f} instead of Rs. {warn['rate_now']:.2f}, "
                           f"about Rs. {warn['extra_cost']:.0f} extra this month. To stay under, keep to "
                           f"{warn['allowance']:.1f} units a day for the remaining {pos['days_left']} days "
                           f"(you are using {pos['pace']:.1f}).")
        else:
            _status(True, f"On course to stay under {warn['boundary']:.0f} units. You have {warn['units_left']:.0f} units "
                          f"left in this slab, {warn['allowance']:.1f} a day; you are using {pos['pace']:.1f}.")
        st.caption("Month-to-date comes from the meter history; the rest of the month is projected from the last 14 days.")

    with tab_why:
        why = tk.explain_change(b["kwh"], b["cost"])
        if not why["span"]:
            st.info("A few more days of meter history are needed before two periods can be compared.")
        else:
            direction = "more" if why["delta_cost"] > 0 else "less"
            _note(f"The last {why['span']} days cost <b style='color:var(--plaster)'>Rs. {abs(why['delta_cost']):.0f} {direction}</b> "
                  f"than the {why['span']} days before ({why['delta_kwh']:+.1f} units). By cause, biggest first:")
            _rows([(r["name"],
                    ("used more" if r["delta_kwh"] > 0 else "used less") + f", {r['delta_kwh']:+.1f} units",
                    f"{'+' if r['delta_cost'] >= 0 else '-'}Rs. {abs(r['delta_cost']):.0f}",
                    f"{r['now_kwh']:.1f} units in the period") for r in why["rows"] if abs(r["delta_cost"]) >= 1])


# ===================================================================== safety
def render_safety(ctx) -> None:
    section, b, cfg = ctx["section"], base(ctx), settings(ctx)
    events = tk.health_events(b["history"], b["sessions"], ctx["owned"])
    peak = tk.peak_demand(b["history"], cfg["sanctioned_kw"], ctx["owned"])
    today_flags = flag_long_runs(extract_sessions(b["today"], ctx["tariff"])) if len(b["today"]) else []
    worst = peak["worst"]
    c1, c2, c3, c4 = st.columns(4)
    ctx["metric_card"](c1, label="Highest sustained load", value=f"{worst['kw']:.2f}" if worst else "0", unit="kW",
                       sub=f"{worst['when']:%a %d %b, %H:%M}" if worst else "")
    ctx["metric_card"](c2, label="Share of sanctioned load", value=f"{worst['pct']:.0f}" if worst else "0", unit="%",
                       sub=f"of {cfg['sanctioned_kw']:.1f} kW")
    ctx["metric_card"](c3, label="Faults and long runs", value=str(len(events)), sub="in the last 30 days")
    ctx["metric_card"](c4, label="Left on today", value=str(len(today_flags)),
                       sub="running longer than normal" if today_flags else "nothing unusual")
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    left, right = st.columns([1.1, 1], gap="medium")
    with left:
        with st.container(border=True):
            section("Overload check")
            new = st.number_input("Sanctioned load (kW)", 0.5, 50.0, float(cfg["sanctioned_kw"]), 0.5, key="tk_safe_sanc",
                                  help="From your bill. The main breaker is sized for this.")
            if new != cfg["sanctioned_kw"]:
                save_settings(ctx, sanctioned_kw=float(new))
                st.rerun()
            if peak["over"]:
                _status(False, f"The load went above the sanctioned {cfg['sanctioned_kw']:.1f} kW on {len(peak['over'])} day(s). "
                               "That is when a breaker trips, and the supplier may add a penalty.")
            elif peak["near"]:
                _status(False, f"The load came within 20% of the sanctioned {cfg['sanctioned_kw']:.1f} kW on "
                               f"{len(peak['near'])} day(s). Avoid starting another heavy appliance at those times.")
            else:
                _status(True, f"The load stayed under 80% of the sanctioned {cfg['sanctioned_kw']:.1f} kW on every day recorded.")
            if peak["days"]:
                frame = pd.DataFrame(peak["days"])
                fig = go.Figure(go.Bar(x=frame["day"], y=frame["kw"], marker_color=[
                    tc("alert") if p >= 100 else tc("current") if p >= 80 else tc("faint") for p in frame["pct"]],
                    hovertemplate="%{x|%a %d %b}<br>%{y:.2f} kW<extra></extra>"))
                fig.add_hline(y=cfg["sanctioned_kw"], line_dash="dot", line_color=tc("alert"),
                              annotation_text="sanctioned load", annotation_font_color=tc("mist"))
                fig.update_layout(**ctx["layout"](height=250, yaxis_title="kW, highest 15 minutes", showlegend=False,
                                                  margin=dict(l=40, r=20, t=16, b=30)))
                st.plotly_chart(fig, width="stretch", key="tk_peak_chart")
            if worst and worst["running"]:
                st.caption(f"At the highest point these were running together: {', '.join(worst['running'])}. "
                           f"If everything you listed ran at once it would draw about {peak['connected_kw']:.1f} kW.")
    with right:
        with st.container(border=True):
            section("Left on right now")
            if today_flags:
                for f in today_flags[:4]:
                    _status(False, f["message"])
                st.caption("Each of these is also sent to the household as a notification, once per run.")
            else:
                _status(True, "Nothing has been running longer than its normal time today.")
        with st.container(border=True):
            section("Fault history")
            if events:
                _rows([(e["name"], f"{e['when']:%a %d %b}" + (f", {e['when']:%H:%M}" if e["kind"] == "Long run" else ""),
                        e["kind"], e["text"]) for e in events[:8]])
                repeat = pd.Series([e["name"] for e in events]).value_counts()
                if repeat.iloc[0] >= 3:
                    st.caption(f"{repeat.index[0]} appears {repeat.iloc[0]} times. A fault that keeps coming back "
                               "is worth a service visit.")
            else:
                _status(True, "No faults or abnormal runs in the last 30 days.")


# =================================================================== upgrades
def render_upgrades(ctx) -> None:
    section, b = ctx["section"], base(ctx)
    rate = b["avg_rate"] or ctx["rate"]
    tab_replace, tab_solar, tab_backup = st.tabs(["Replace an appliance", "Rooftop solar", "Inverter and battery"])

    with tab_replace:
        _note("What a current 5-star model would save against what your appliance uses now, from your own "
              "meter history. Change the price or the saving to match a model you are looking at.")
        candidates = [k for k in tk.REPLACEMENTS if k in b["month_kwh"]]
        if not candidates:
            st.info("None of the appliances you listed has a replacement estimate.")
        overrides = {}
        for key in candidates:
            spec = tk.REPLACEMENTS[key]
            with st.expander(f"{DEFAULT_PROFILES[key].name}: {spec['what']}"):
                c1, c2 = st.columns(2)
                overrides[key] = {
                    "price": c1.number_input("Price (Rs)", 1000, 300000, int(spec["price"]), 500, key=f"tk_price_{key}"),
                    "saving": c2.slider("Energy saved (%)", 5, 60, int(spec["saving"] * 100), key=f"tk_save_{key}") / 100.0}
        rows = tk.replacement_table(b["month_kwh"], rate, overrides)
        if rows:
            _rows([(r["name"], f"uses {r['now_kwh']:.0f} units a month now; {r['what']} saves about {r['saving_pct']:.0f}%",
                    f"Rs. {r['saved_rs']:.0f}/month",
                    (f"pays back in {r['payback_months'] / 12:.1f} years" if r["payback_months"] and r["payback_months"] < 600
                     else "would not pay back")) for r in rows])
            best = rows[0]
            if best["payback_months"] and best["payback_months"] <= 60:
                _status(True, f"Best case: the {best['name'].lower()}. Rs. {best['price']:,} pays for itself in "
                              f"{best['payback_months'] / 12:.1f} years.")
            else:
                _status(True, "None of these pays for itself within five years on energy alone. Replace when the old one fails.")
            st.caption("Savings percentages are typical figures for 5-star models against older units, not measurements of a specific model.")

    with tab_solar:
        first = tk.solar_estimate(b["history"], rate)
        c1, c2, c3 = st.columns(3)
        kwp = c1.number_input("System size (kW)", 0.5, 20.0, float(first["suggested_kwp"]), 0.5, key="tk_kwp")
        cost_kw = c2.number_input("Installed cost (Rs per kW)", 30000, 120000, 60000, 1000, key="tk_cost_kw")
        net = c3.toggle("Net metering", value=True, key="tk_net",
                        help="On: units you export are credited against units you import. Off: only daytime use is offset.")
        s = tk.solar_estimate(b["history"], rate, kwp, cost_per_kwp=float(cost_kw), net_metering=net)
        c1, c2, c3, c4 = st.columns(4)
        ctx["metric_card"](c1, label="Generates", value=f"{s['generated']:.0f}", unit="units/month",
                           sub=f"needs about {s['roof_sqft']:.0f} sq ft of roof")
        ctx["metric_card"](c2, label="Covers", value=f"{s['covered_pct']:.0f}", unit="%",
                           sub=f"of your {s['month_kwh']:.0f} units a month")
        ctx["metric_card"](c3, label="Saves", value=f"Rs. {s['saving_month']:.0f}", unit="/month",
                           sub=f"at Rs. {rate:.2f} a unit")
        ctx["metric_card"](c4, label="Pays back in", value=f"{s['payback_years']:.1f}" if s["payback_years"] else "-",
                           unit="years", sub=f"Rs. {s['net_cost']:,.0f} after subsidy")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        _rows([("System cost", f"{s['kwp']:.1f} kW at Rs. {cost_kw:,} per kW", f"Rs. {s['cost']:,.0f}", ""),
               ("Central subsidy", "Rs. 30,000 per kW for the first 2 kW, Rs. 18,000 for the third, at most Rs. 78,000",
                f"- Rs. {s['subsidy']:,.0f}", ""),
               ("Your daytime use (9 am to 5 pm)", "what solar offsets directly without net metering",
                f"{s['day_kwh']:.0f} units/month", "")])
        st.caption(f"Assumes 4 units a day from each kW of panels, a typical figure for India; a suggested size for your "
                   f"use is {first['suggested_kwp']:.1f} kW. Confirm the subsidy and net-metering rules with your supplier before buying.")

    with tab_backup:
        _note("Pick what must keep running in a power cut and for how long. The inverter and battery are sized from "
              "the rated power of those items.")
        choices = [{"name": DEFAULT_PROFILES[k].name, "kw": DEFAULT_PROFILES[k].rated_kw} for k in ctx["owned"]] \
            + [{"name": d["name"], "kw": d["kw"]} for d in ctx["switched"]]
        if not choices:
            st.info("Add appliances in Settings to size a backup.")
            return
        default = [c["name"] for c in choices if c["kw"] <= 0.2][:6]
        picked = st.multiselect("Keep running", [c["name"] for c in choices], default=default, key="tk_backup_pick")
        hours = st.slider("For how many hours", 0.5, 8.0, 3.0, 0.5, key="tk_backup_hours")
        size = tk.backup_size([c for c in choices if c["name"] in picked], hours)
        c1, c2, c3 = st.columns(3)
        ctx["metric_card"](c1, label="Load to carry", value=f"{size['watts']:.0f}", unit="W", sub=f"{len(picked)} item(s)")
        ctx["metric_card"](c2, label="Inverter size", value=f"{size['standard_va'] or size['va']:.0f}", unit="VA",
                           sub=f"needs at least {size['va']:.0f} VA" if size["standard_va"] else "larger than home units; ask an installer")
        ctx["metric_card"](c3, label="Battery", value=f"{size['ah']:.0f}", unit="Ah at 12 V",
                           sub=f"{size['batteries']} x {size['battery_ah']:.0f} Ah")
        if size["heavy"]:
            _status(False, f"{', '.join(size['heavy'])}: 1 kW or more. Heating and cooling appliances drain a home "
                           "inverter in minutes and compressors need several times their rated power to start. "
                           "Leave them off the backup.")
        st.caption("Worked out as watts / 0.8 power factor, plus 25% headroom; battery as watts x hours / (12 V x 0.8).")


# ====================================================================== goals
@st.cache_data(show_spinner=False)
def _peer_days(owned: tuple, homes: int = 5) -> list:
    """Average daily units of simulated homes with the same appliances."""
    out = []
    for i in range(homes):
        df = simulate_home(days=7, seed=9100 + i, include=owned)
        out.append(float(df["mains_kw"].sum() / 60.0 / 7.0))
    return out


def render_goals(ctx) -> None:
    section, b, cfg = ctx["section"], base(ctx), settings(ctx)
    month_cost = float(b["cost"]["total"].iloc[:-1].tail(30).mean() * 30)
    with st.container(border=True):
        section("Monthly goal")
        target = st.number_input("Spend no more than (Rs a month)", 0.0, 100000.0,
                                 float(cfg["goal"] or round(month_cost * 0.9, -1)), 50.0, key="tk_goal",
                                 help="Starts at 10% under your recent monthly cost.")
        save_settings(ctx, goal=float(target))
        g = tk.goal_status(b["cost"]["total"], target)
        c1, c2, c3, c4 = st.columns(4)
        ctx["metric_card"](c1, label="Spent this month", value=f"Rs. {g['spent']:.0f}", sub=f"of Rs. {g['target']:.0f}")
        ctx["metric_card"](c2, label="Heading for", value=f"Rs. {g['projected']:.0f}",
                           sub="at the pace of the last 14 days")
        ctx["metric_card"](c3, label="You can spend", value=f"Rs. {g['allowance']:.0f}", unit="/day",
                           sub=f"for the remaining {g['days_left']} days")
        ctx["metric_card"](c4, label="Streak", value=str(g["streak"]), unit="days",
                           sub=f"at or under Rs. {g['per_day']:.0f} a day")
        st.progress(min(1.0, g["spent"] / max(g["target"], 1e-9)))
        if g["on_track"]:
            _status(True, f"On track: about Rs. {abs(g['gap']):.0f} under the goal at month end.")
        else:
            saving = (ctx.get("brief") or {}).get("saving_month", 0)
            _status(False, f"Heading about Rs. {g['gap']:.0f} over the goal. You are spending Rs. {g['pace']:.0f} a day; "
                           f"the goal allows Rs. {g['allowance']:.0f}. "
                           + (f"The changes on the Overview page are worth about Rs. {saving:.0f} a month." if saving else ""))

    left, right = st.columns(2, gap="medium")
    with left:
        with st.container(border=True):
            section("Homes like yours")
            mine = float(b["kwh"]["total"].iloc[:-1].tail(7).mean())
            if ctx["owned"]:
                peers = _peer_days(tuple(ctx["owned"]))
                avg = sum(peers) / len(peers)
                better = sum(1 for p in peers if mine < p)
                _rows([("Your home", "average of the last 7 days", f"{mine:.1f} units/day", ""),
                       ("Similar homes", f"{len(peers)} simulated homes with the same appliances", f"{avg:.1f} units/day",
                        f"range {min(peers):.1f} to {max(peers):.1f}")])
                _status(mine <= avg, f"You use {abs(mine - avg) / max(avg, 1e-9) * 100:.0f}% "
                                     f"{'less' if mine <= avg else 'more'} than the average, and less than "
                                     f"{better} of the {len(peers)}.")
                st.caption("The comparison homes are simulated, not real neighbours: same appliance list, different habits.")
            else:
                st.info("Add a meter-detected appliance to compare with similar homes.")
        with st.container(border=True):
            section("Carbon")
            factor = st.number_input("Grid factor (kg CO2 per unit)", 0.1, 1.5, float(cfg["co2"]), 0.01, key="tk_co2",
                                     help="India grid average is about 0.71 (Central Electricity Authority).")
            save_settings(ctx, co2=float(factor))
            month_units = float(b["kwh"]["total"].iloc[:-1].tail(30).mean() * 30)
            rows = [("This month, at your pace", f"{month_units:.0f} units", f"{tk.carbon_kg(month_units, factor):.0f} kg CO2", "")]
            for key in sorted(ctx["owned"], key=lambda k: -b["month_kwh"][k])[:3]:
                rows.append((DEFAULT_PROFILES[key].name, f"{b['month_kwh'][key]:.0f} units a month",
                             f"{tk.carbon_kg(b['month_kwh'][key], factor):.0f} kg", ""))
            saving = (ctx.get("brief") or {}).get("saving_month", 0)
            if saving:
                rows.append(("If you make the suggested changes", f"about {saving / max(b['avg_rate'], 1e-9):.0f} units less",
                             f"- {tk.carbon_kg(saving / max(b['avg_rate'], 1e-9), factor):.0f} kg", "every month"))
            _rows(rows)
    with right:
        with st.container(border=True):
            section("Schedules")
            _note("Set the hours you intend to run an appliance. The app cannot switch anything; it tells you "
                  "when a run fell outside your plan.")
            usable = [k for k in ctx["owned"] if DEFAULT_PROFILES[k].category != "always_on"]
            schedules = dict(cfg["schedules"])
            if not usable:
                st.info("None of your meter-detected appliances can be scheduled.")
            for key in usable:
                on = st.checkbox(DEFAULT_PROFILES[key].name, value=key in schedules, key=f"tk_sch_on_{key}")
                if on:
                    lo, hi = schedules.get(key, [360, 600])
                    span = st.slider("Allowed hours", 0, 24, (int(lo) // 60, max(int(hi) // 60, int(lo) // 60 + 1)),
                                     key=f"tk_sch_{key}", label_visibility="collapsed")
                    schedules[key] = [span[0] * 60, span[1] * 60]
                else:
                    schedules.pop(key, None)
            save_settings(ctx, schedules=schedules)
            report = tk.schedule_report(b["sessions"], {k: tuple(v) for k, v in schedules.items()})
            for r in report:
                window = f"{_hhmm(r['window'][0])} to {_hhmm(r['window'][1])}"
                if not r["runs"]:
                    st.caption(f"{r['name']}: no runs in the last 7 days.")
                elif not r["outside"]:
                    _status(True, f"{r['name']}: all {r['runs']} run(s) in the last 7 days were inside {window}.")
                else:
                    when = ", ".join(f"{o['start']:%a %H:%M}" for o in r["outside"][:4])
                    _status(False, f"{r['name']}: {len(r['outside'])} of {r['runs']} run(s) were outside {window} "
                                   f"({when}), costing Rs. {r['outside_cost']:.0f}.")


# ===================================================================== family
def render_family_extras(ctx) -> None:
    section, b, db = ctx["section"], base(ctx), ctx["db"]
    names = {k: DEFAULT_PROFILES[k].name for k in ctx["owned"]}
    names.update({d["key"]: d["name"] for d in ctx["switched"]})
    members = tk.member_activity(db.get_notification_log(ctx["household_id"], limit=500),
                                 {d["key"]: d["kw"] for d in ctx["switched"]}, names, b["avg_rate"] or ctx["rate"])
    left, right = st.columns([1.2, 1], gap="medium")
    with left:
        with st.container(border=True):
            section("Who switched what")
            if members:
                _rows([(m["name"], f"most used: {m['most_used']}" + (f"; last at {to_local(m['last']):%d %b %H:%M}" if m["last"] is not None else ""),
                        f"{m['switches']} switch(es)",
                        f"{m['minutes']:.0f} min on, Rs. {m['cost']:.2f}" if m["minutes"] else "") for m in members])
                st.caption("Counted from the switches used on the Home page. Time and cost are worked out for devices "
                           "shown from their switch.")
            else:
                st.info("Nobody has used a switch yet. Switch a device on the Home page and it will appear here.")
    with right:
        with st.container(border=True):
            section("Weekly family report")
            _note("A one-page PDF of the last seven days for the household. It is fingerprinted when created, so a "
                  "changed copy can be detected.")
            if st.button("Prepare this week's report", key="tk_weekly", type="primary"):
                kwh, cost = b["kwh"].iloc[:-1], b["cost"].iloc[:-1]
                week, before = kwh["total"].tail(7).sum(), kwh["total"].iloc[-14:-7].sum()
                parts = [{"name": "Everything else" if c == "other" else DEFAULT_PROFILES[c].name,
                          "kwh": float(kwh[c].tail(7).sum()), "cost": float(cost[c].tail(7).sum()),
                          "share": 100.0 * float(kwh[c].tail(7).sum()) / max(float(week), 1e-9)}
                         for c in kwh.columns if c != "total"]
                cutoff = kwh.index[-7]
                report = {
                    "household": ctx["user_name"], "prepared": f"{local_now():%d %b %Y}",
                    "period": f"{kwh.index[-7]:%d %b} to {kwh.index[-1]:%d %b %Y}",
                    "kwh": float(week), "cost": float(cost["total"].tail(7).sum()),
                    "change_pct": 100.0 * float(week - before) / max(float(before), 1e-9),
                    "appliances": sorted(parts, key=lambda r: -r["kwh"]),
                    "actions": (ctx.get("brief") or {}).get("actions", []),
                    "events": [e for e in tk.health_events(b["history"], b["sessions"], ctx["owned"]) if e["when"] >= cutoff],
                    "members": members, "co2": tk.carbon_kg(float(week), settings(ctx)["co2"]),
                }
                pdf = tk.weekly_report_pdf(report)
                name = f"energypulse_week_{local_now():%Y%m%d}.pdf"
                ctx["ledger"].record(ctx["household_id"], "weekly_report", name, pdf, {"period": report["period"]})
                st.session_state["tk_weekly_pdf"] = (name, pdf)
                notify("success", "Weekly report prepared and fingerprinted.")
            if "tk_weekly_pdf" in st.session_state:
                name, pdf = st.session_state["tk_weekly_pdf"]
                st.download_button("Download the PDF", pdf, file_name=name, mime="application/pdf", key="tk_weekly_dl")
            probe = st.file_uploader("Check a report file", type=["pdf"], key="tk_weekly_probe")
            if probe is not None:
                match = ctx["ledger"].find_by_content(ctx["household_id"], probe.getvalue())
                _status(bool(match), f"Unchanged. This is \"{match['ref']}\", created {match['created_at']}." if match
                        else "This file does not match any report created here. It was changed, or made elsewhere.")


def whatsapp_link(subject: str, body: str) -> str:
    """A link that opens WhatsApp with the forecast typed in, ready for the user to pick a chat."""
    lines = [l for l in body.splitlines() if l.strip()]
    plan = [l.strip() for l in lines if l.strip()[:2] in {"1.", "2.", "3."}]
    text = "\n".join(["EnergyPulse: " + subject] + lines[:1] + plan[:3])
    return "https://wa.me/?text=" + quote(text[:900])


# ======================================================================= help
def render_help(ctx) -> None:
    section, db, hid = ctx["section"], ctx["db"], ctx["household_id"]
    outages = db.get_store(hid, "outages", []) or []
    summary = tk.outage_summary(outages, local_now())
    left, right = st.columns(2, gap="medium")
    with left:
        with st.container(border=True):
            section("Power cut log")
            c1, c2, c3 = st.columns(3)
            ctx["metric_card"](c1, label="This month", value=f"{summary['hours_month']:.1f}", unit="hours",
                               sub=f"{summary.get('count_month', 0)} cut(s)")
            ctx["metric_card"](c2, label="Last 30 days", value=f"{summary['hours_30']:.1f}", unit="hours",
                               sub=f"{summary.get('count_30', 0)} cut(s)")
            ctx["metric_card"](c3, label="Longest", value=f"{summary['longest']}", unit="min", sub="single cut")
            with st.form("tk_outage_form", clear_on_submit=True):
                f1, f2, f3 = st.columns(3)
                day = f1.date_input("Date", value=local_now().date(), max_value=local_now().date())
                start = f2.time_input("Started at", value=pd.Timestamp("18:00").time())
                minutes = f3.number_input("Minutes without power", 1, 2880, 30, 5)
                note = st.text_input("Note (optional)", placeholder="For example: whole street, after rain")
                if st.form_submit_button("Add power cut", type="primary"):
                    outages.append({"date": str(day), "start": start.strftime("%H:%M"), "minutes": int(minutes),
                                    "note": note.strip()[:120]})
                    db.set_store(hid, "outages", outages[-500:])
                    notify("success", "Power cut recorded.")
                    st.rerun()
            if outages:
                recent = sorted(enumerate(outages), key=lambda x: (x[1]["date"], x[1]["start"]), reverse=True)[:8]
                for idx, o in recent:
                    cols = st.columns([4, 1], vertical_alignment="center")
                    cols[0].markdown(f"<div class='ep-row-name'>{pd.Timestamp(o['date']):%a %d %b %Y}, {o['start']}"
                                     f" &middot; {o['minutes']} min</div><div class='ep-row-kind'>{o.get('note') or 'no note'}</div>",
                                     unsafe_allow_html=True)
                    if cols[1].button("Remove", key=f"tk_out_del_{idx}"):
                        outages.pop(idx)
                        db.set_store(hid, "outages", outages)
                        st.rerun()
            else:
                st.caption("No power cuts recorded yet. A log with dates and durations is the evidence a supplier asks for.")
    with right:
        with st.container(border=True):
            section("Complaint letter")
            _note("Fills a formal complaint with your own figures. Nothing is sent from here: copy it or download it, "
                  "then submit it through your supplier's office, website or app.")
            kind = st.selectbox("What is the complaint about", list(tk.COMPLAINT_KINDS), key="tk_c_kind")
            c1, c2 = st.columns(2)
            consumer = c1.text_input("Consumer number", key="tk_c_no", placeholder="From your bill")
            supplier = c2.text_input("Supplier", key="tk_c_sup", placeholder="For example: BESCOM")
            address = st.text_input("Address (optional)", key="tk_c_addr")
            facts = []
            check = st.session_state.get("tk_last_check")
            if kind == "Bill amount looks wrong":
                if check:
                    facts += [f"Units billed: {check['units']:.0f}. Amount charged: Rs. {check['billed']:.0f}.",
                              f"Amount worked out from the published {check['tariff']} tariff: Rs. {check['total']:.0f}.",
                              f"Difference: Rs. {abs(check['difference']):.0f} "
                              f"{'more' if check['difference'] > 0 else 'less'} than expected."]
                else:
                    st.caption("Run 'Is my bill right?' on the My Bills page first and the figures are added here.")
            if kind == "Meter may be faulty":
                b = base(ctx)
                facts.append(f"My recorded use over the last 30 days is about {b['kwh']['total'].iloc[:-1].tail(30).sum():.0f} units.")
                if check:
                    facts.append(f"The latest bill shows {check['units']:.0f} units.")
            if kind == "Frequent power cuts":
                if summary["count"]:
                    facts.append(f"In the last 30 days there were {summary.get('count_30', 0)} power cuts totalling "
                                 f"{summary['hours_30']:.1f} hours; the longest lasted {summary['longest']} minutes.")
                    for o in sorted(outages, key=lambda o: o["date"], reverse=True)[:5]:
                        facts.append(f"{pd.Timestamp(o['date']):%d %b %Y}, from {o['start']}, {o['minutes']} minutes.")
                else:
                    st.caption("Record the cuts in the log on the left and they are listed in the letter.")
            extra = st.text_area("Anything to add", key="tk_c_extra", height=80)
            letter = tk.complaint_letter(kind, ctx["user_name"], consumer.strip(), address, supplier.strip(), facts, extra)
            st.text_area("Letter", letter, height=330, key=f"tk_letter_{hash(letter)}")
            st.download_button("Download the letter", letter, file_name="electricity_complaint.txt",
                               mime="text/plain", key="tk_letter_dl")
