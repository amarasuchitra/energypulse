"""Meter readings, slab bills and Gruha Jyothi (billing.py)."""
import pandas as pd
import pytest

from billing import (BadReading, BillPlan, add_reading, advice, bill, clean_readings, energy_charge,
                     month_status)

SLABS = [{"upto": 100.0, "rate": 4.0}, {"upto": 200.0, "rate": 6.0}, {"upto": None, "rate": 8.0}]


def _plan(**kw):
    return BillPlan.clean({"slabs": SLABS, "fixed_per_kw": 100, "sanctioned_kw": 2, **kw})


def _readings(pairs):
    raw = []
    for at, v in pairs:
        raw = add_reading(raw, v, at)
    return raw


def test_slab_charges_are_telescopic():
    assert energy_charge(50, SLABS)[0] == 200.0
    assert energy_charge(150, SLABS)[0] == 400 + 300
    assert energy_charge(250, SLABS)[0] == 400 + 600 + 400
    assert energy_charge(0, SLABS)[0] == 0.0
    b = bill(150, _plan())
    assert b["energy"] == 700 and b["fixed"] == 200 and b["payable"] == 900 and b["scheme"] is None


def test_gruha_jyothi_free_partly_free_and_above_200():
    p = _plan(gruha_jyothi=True, gj_average=120)
    assert p.free_limit == 132.0                                   # 120 + 10%
    assert bill(130, p)["payable"] == 0.0 and bill(130, p)["scheme"] == "free"
    partly = bill(150, p)                                          # pays for 18 units above 132, at 6.0
    assert partly["scheme"] == "partly free" and partly["payable"] == pytest.approx(18 * 6.0)
    assert bill(201, p)["payable"] == bill(201, _plan())["payable"]   # no benefit above 200
    assert _plan(gruha_jyothi=True, gj_average=190).free_limit == 200.0   # capped
    assert _plan(gruha_jyothi=True).free_limit is None             # no average given: not assumed


def test_plan_cleaning_rejects_bad_values_and_keeps_an_open_top_slab():
    p = BillPlan.clean({"slabs": [{"upto": 200, "rate": 6}, {"upto": 100, "rate": "4"}, {"upto": 50, "rate": "x"}],
                        "fixed_per_kw": -5, "sanctioned_kw": "abc"})
    assert [s["upto"] for s in p.slabs] == [100, 200, None] and p.slabs[-1]["rate"] == 6
    assert p.fixed_per_kw == 0 and p.sanctioned_kw == 2.0
    assert BillPlan.clean("junk").slabs == BillPlan().slabs


def test_readings_must_count_up_and_be_physically_possible():
    raw = _readings([("2026-10-01 08:00", 4500)])
    with pytest.raises(BadReading, match="only counts up"):
        add_reading(raw, 4400, "2026-10-02 08:00")
    with pytest.raises(BadReading, match="more than a home"):
        add_reading(raw, 4500 + 1000, "2026-10-02 08:00")         # 1000 units in a day
    with pytest.raises(BadReading):
        add_reading(raw, "abc", "2026-10-02 08:00")
    with pytest.raises(BadReading, match="already a reading"):
        add_reading(raw, 4501, "2026-10-01 08:00")
    with pytest.raises(BadReading, match="later reading"):
        add_reading(_readings([("2026-10-01", 4500), ("2026-10-05", 4540)]), 4600, "2026-10-03")
    assert len(add_reading(raw, 4507.5, "2026-10-02 08:00")) == 2


def test_month_figures_from_readings_across_the_first():
    raw = _readings([("2026-09-28 00:00", 1000), ("2026-10-02 00:00", 1024), ("2026-10-11 00:00", 1078)])
    s = month_status(raw, pd.Timestamp("2026-10-11 09:00"))
    # 6/day all along: the 1st falls at 1018 -> 60 units so far on 11 Oct
    assert s["units_so_far"] == pytest.approx(60.0)
    assert s["daily_avg"] == pytest.approx(6.0)
    assert s["projected"] == pytest.approx(31 * 6.0, abs=0.2)      # whole of October at 6/day


def test_month_figures_need_two_readings_and_never_invent_numbers():
    assert month_status([], pd.Timestamp("2026-10-11"))["projected"] is None
    one = _readings([("2026-10-01", 1000)])
    s = month_status(one, pd.Timestamp("2026-10-11"))
    assert s["units_so_far"] is None and s["projected"] is None
    assert advice(s, _plan()) is None
    assert clean_readings([{"at": "bad", "value": 1}, {"value": 2}, {"at": "2026-10-01", "value": -1}]) == []


def test_advice_says_how_much_to_cut_to_stay_free():
    p = _plan(gruha_jyothi=True, gj_average=120)                   # limit 132
    raw = _readings([("2026-10-01 00:00", 1000), ("2026-10-11 00:00", 1060)])   # 6/day, 60 so far
    a = advice(month_status(raw, pd.Timestamp("2026-10-11 00:00")), p)
    assert a["target"] == 132.0 and a["projected_bill"]["payable"] > 0
    assert a["per_day_needed"] == pytest.approx((132 - 60) / 21, abs=0.01)
    assert "stay within it" in a["message"] and "Rs." in a["message"]
    low = _readings([("2026-10-01 00:00", 1000), ("2026-10-11 00:00", 1030)])    # 3/day: on track
    assert advice(month_status(low, pd.Timestamp("2026-10-11")), p)["message"].startswith("On track")


def test_advice_without_the_scheme_points_at_the_next_slab():
    raw = _readings([("2026-10-01 00:00", 1000), ("2026-10-11 00:00", 1060)])
    a = advice(month_status(raw, pd.Timestamp("2026-10-11")), _plan())
    assert a["target"] == 100.0 and "dearer slab" in a["message"]


def test_reading_saved_through_the_dashboard():
    """The real app: a signed-in user types two readings; the month card appears."""
    import uuid
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    import os
    import catalog
    from auth import Accounts
    from db import DatabaseManager
    import billing
    email = f"meter-{uuid.uuid4().hex[:8]}@example.com"
    Accounts().sign_up(email, "Tr4il-Mango-Kettle", "Meter User")
    db = DatabaseManager()
    hid = db.ensure_user_login(email, "Meter User")["household_id"]
    db.save_home_details(hid, {"home_type": "Apartment", "occupants": 3, "appliances": [catalog.item_for("ac")]})
    now = pd.Timestamp.now().floor("min")
    db.set_store(hid, billing.READINGS_KEY, add_reading([], 1000, now - pd.Timedelta(days=3)))
    app = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "app.py")
    at = AppTest.from_file(app, default_timeout=600)
    at.run()
    at.text_input(key="login_email").input(email)
    at.text_input(key="login_pw").input("Tr4il-Mango-Kettle")
    at.checkbox(key="login_remember").uncheck()
    [b for b in at.button if b.label == "Sign In"][0].click().run()
    assert not at.exception
    box = [n for n in at.number_input if str(n.key).startswith("mr_value")]
    assert box, "no meter reading box on the dashboard"
    box[0].set_value(1020.5).run()
    at.button(key="mr_save").click().run()
    assert not at.exception
    saved = billing.clean_readings(db.get_store(hid, billing.READINGS_KEY))
    assert [r["value"] for r in saved] == [1000.0, 1020.5]
    text = " ".join(str(m.value) for m in list(at.markdown) + list(at.caption) + list(at.success) + list(at.warning))
    assert "Heading for" in text or "Estimated bill" in text or "units" in text
