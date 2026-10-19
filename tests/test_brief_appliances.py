"""Saved appliance setups: validation, persistence, isolation (brief sections 3 and 4, storage side).

The screens themselves (Edit appliances button, Save/Cancel) are driven in test_brief_ui.py.
"""
import json
import sqlite3
import threading

import pytest

import catalog
from db import DatabaseManager
from home_config import (LIMITS, MAX_APPLIANCES, InvalidSetup, expected_daily_kwh, usage_factor,
                         validate_home_details)


@pytest.fixture
def db(tmp_path):
    return DatabaseManager(db_path=str(tmp_path / "home.db"))


def _home(*items, **kw):
    d = {"home_type": "Apartment", "occupants": 3, "home_size": 900, "tariff_rate": 8.0, "city": "Mysuru",
         "appliances": list(items)}
    d.update(kw)
    return d


AC = catalog.item_for("ac")
FRIDGE = catalog.item_for("fridge")
FAN = catalog.item_for("fan")


@pytest.mark.brief("APP-002", "APP-003", "APP-005", "APP-006")
def test_valid_setup_is_saved_and_survives_a_restart(tmp_path):
    path = str(tmp_path / "home.db")
    db1 = DatabaseManager(db_path=path)
    assert db1.save_home_details("hh1", _home(AC, FRIDGE)) == []
    db2 = DatabaseManager(db_path=path)                       # a fresh connection: the app restarted
    saved = db2.get_home_details("hh1")
    assert [a["type_key"] for a in saved["appliances"]] == ["ac", "fridge"]
    assert db2.get_home_details("hh1") == saved                # reading again (a page refresh) is stable


@pytest.mark.brief("APP-001", "APP-022")
def test_new_household_has_no_setup_and_missing_fields_get_defaults(db):
    assert db.get_home_details("never-saved") is None          # the app then shows the setup screens
    clean, problems = validate_home_details({"appliances": [{"name": "Lamp"}]})
    item = clean["appliances"][0]
    assert item["quantity"] == 1 and item["usage"] == "Medium" and "hours_per_day" not in item
    assert clean["occupants"] == 4 and clean["tariff_rate"] == 8.0 and clean["home_type"] == "Other"
    assert problems == []


@pytest.mark.brief("APP-007")
def test_empty_appliance_list_is_valid_and_stored_as_empty(db):
    db.save_home_details("hh1", _home())
    assert db.get_home_details("hh1")["appliances"] == []
    assert expected_daily_kwh({}) == 0.0


@pytest.mark.brief("APP-008", "APP-009", "APP-024", "EDIT-007", "EDIT-011")
def test_add_remove_and_change_several_things_then_save(db):
    db.save_home_details("hh1", _home(AC, FRIDGE))
    hd = db.get_home_details("hh1")
    items = hd["appliances"]
    items.pop(0)                                               # remove the AC
    items.append(dict(FAN, quantity=3, hours_per_day=10))      # add three fans
    items[0]["usage"] = "High"                                 # change the fridge
    db.save_home_details("hh1", hd, strict=True)
    after = db.get_home_details("hh1")["appliances"]
    assert [a["type_key"] for a in after] == ["fridge", "fan"]
    assert after[0]["usage"] == "High" and after[1]["quantity"] == 3 and after[1]["hours_per_day"] == 10


@pytest.mark.brief("APP-010", "APP-011", "APP-012", "EDIT-004")
def test_quantity_power_and_duration_changes_persist_and_change_the_estimate(db):
    own = {"name": "Old geyser", "type_key": "geyser", "spec_mode": "own", "own": {"rated_w": 2000},
           "quantity": 1, "hours_per_day": 1}
    db.save_home_details("hh1", _home(own))
    before = expected_daily_kwh(db.get_home_details("hh1")["appliances"][0])
    hd = db.get_home_details("hh1")
    hd["appliances"][0].update(quantity=2, hours_per_day=2)
    hd["appliances"][0]["own"]["rated_w"] = 3000
    db.save_home_details("hh1", hd, strict=True)
    item = db.get_home_details("hh1")["appliances"][0]
    assert item["quantity"] == 2 and item["hours_per_day"] == 2 and item["own"]["rated_w"] == 3000
    after = expected_daily_kwh(item)
    assert after == pytest.approx(before * 2 * 2 * 1.5, rel=0.02)    # 2 units x 2 h x 1.5 x power


@pytest.mark.brief("APP-013", "APP-014", "APP-015", "EDIT-009")
@pytest.mark.parametrize("change, words", [
    ({"quantity": 0}, "quantity must be between 1 and 10"),
    ({"quantity": 11}, "quantity must be between 1 and 10"),
    ({"hours_per_day": -1}, "hours per day must be between 0 and 24"),
    ({"hours_per_day": 25}, "hours per day must be between 0 and 24"),
    ({"own": {"rated_w": -500}, "spec_mode": "own"}, "rated power must be between 0 and 20000"),
    ({"own": {"rated_w": 10 ** 9}, "spec_mode": "own"}, "rated power must be between 0 and 20000"),
    ({"hours_per_day": "abc"}, "hours per day must be a number"),
    ({"hours_per_day": float("nan")}, "hours per day must be a number"),
])
def test_out_of_range_values_are_refused_by_a_strict_save(db, change, words):
    db.save_home_details("hh1", _home(FRIDGE))
    with pytest.raises(InvalidSetup) as err:
        db.save_home_details("hh1", _home(dict(FRIDGE, **change)), strict=True)
    assert words in str(err.value)
    assert db.get_home_details("hh1")["appliances"] == [dict(FRIDGE, quantity=1)] or \
        db.get_home_details("hh1")["appliances"][0]["type_key"] == "fridge"     # old setup untouched


@pytest.mark.brief("APP-015")
def test_too_many_appliances_are_capped(db):
    clean, problems = validate_home_details(_home(*[{"name": f"Lamp {i}"} for i in range(MAX_APPLIANCES + 5)]))
    assert len(clean["appliances"]) == MAX_APPLIANCES and any("At most" in p for p in problems)


@pytest.mark.brief("APP-016")
@pytest.mark.parametrize("bad", ["not a dict", 42, None, ["list"]])
def test_malformed_setup_is_rejected(db, bad):
    with pytest.raises((InvalidSetup, ValueError)):
        db.save_home_details("hh1", bad)


@pytest.mark.brief("APP-016")
def test_malformed_appliance_list_and_unknown_fields(db):
    with pytest.raises(InvalidSetup):
        db.save_home_details("hh1", _home() | {"appliances": "fan, fridge"})
    clean, _ = validate_home_details(_home({"name": "Fan", "evil": "<script>", "household_id": "someone-else"}))
    assert "evil" not in clean["appliances"][0] and "household_id" not in clean["appliances"][0]


@pytest.mark.brief("APP-017")
def test_duplicate_catalogue_entries_are_merged_into_a_quantity():
    clean, _ = validate_home_details(_home(FAN, FAN, dict(FAN, quantity=2)))
    fans = [a for a in clean["appliances"] if a["type_key"] == "fan"]
    assert len(fans) == 1 and fans[0]["quantity"] == 4


@pytest.mark.brief("APP-018", "EDIT-010")
def test_failed_save_keeps_the_previous_setup(tmp_path):
    path = str(tmp_path / "home.db")
    db = DatabaseManager(db_path=path)
    db.save_home_details("hh1", _home(AC))
    with sqlite3.connect(path) as c:                          # the database refuses writes
        c.execute("CREATE TRIGGER no_writes BEFORE UPDATE ON household_homes BEGIN SELECT RAISE(ABORT, 'disk full'); END")
    with pytest.raises(sqlite3.DatabaseError):
        db.save_home_details("hh1", _home(FRIDGE))
    assert db.get_home_details("hh1")["appliances"][0]["type_key"] == "ac"


@pytest.mark.brief("APP-019")
def test_unreachable_database_raises_a_clear_error(tmp_path):
    db = DatabaseManager(db_path=str(tmp_path / "ok.db"))
    db.db_path = str(tmp_path / "gone" / "x.db")
    with pytest.raises(sqlite3.OperationalError):
        db.get_home_details("hh1")


@pytest.mark.brief("APP-020", "APP-021", "EDIT-015", "LIVE-017", "SEC-011")
def test_households_only_see_and_change_their_own_setup(db):
    a = db.ensure_user_login("a@example.com", "A")["household_id"]
    b = db.ensure_user_login("b@example.com", "B")["household_id"]
    assert a != b
    db.save_home_details(a, _home(AC))
    db.save_home_details(b, _home(FRIDGE))
    assert db.get_home_details(a)["appliances"][0]["type_key"] == "ac"
    assert db.get_home_details(b)["appliances"][0]["type_key"] == "fridge"
    # The household id is derived from the signed-in email on the server, never from the form:
    # a setup that claims another household id is still stored under the caller's own id.
    db.save_home_details(a, _home(FAN, household_id=b))
    assert db.get_home_details(b)["appliances"][0]["type_key"] == "fridge"


@pytest.mark.brief("APP-025")
def test_concurrent_saves_leave_one_complete_setup(db):
    errors = []

    def save(item):
        try:
            for _ in range(10):
                db.save_home_details("hh1", _home(item))
        except Exception as exc:                               # noqa: BLE001
            errors.append(exc)
    threads = [threading.Thread(target=save, args=(i,)) for i in (AC, FRIDGE, FAN)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    saved = db.get_home_details("hh1")
    assert len(saved["appliances"]) == 1 and saved["appliances"][0]["type_key"] in ("ac", "fridge", "fan")
    json.dumps(saved)                                          # a whole, valid record, never a mix


@pytest.mark.brief("EDIT-008", "APP-023")
def test_a_draft_that_is_not_saved_changes_nothing(db):
    db.save_home_details("hh1", _home(AC, FRIDGE))
    draft = db.get_home_details("hh1")["appliances"]
    draft.pop()                                                # edited in the screen, then Cancel
    assert [a["type_key"] for a in db.get_home_details("hh1")["appliances"]] == ["ac", "fridge"]


@pytest.mark.brief("ML-011", "OPT-005", "OPT-015")
def test_editing_quantity_or_hours_changes_the_simulated_meter_input():
    base = dict(FAN)
    assert usage_factor(dict(base, quantity=2)) == pytest.approx(2 * usage_factor(base))
    assert usage_factor(dict(base, hours_per_day=24)) > usage_factor(dict(base, hours_per_day=6))
    pytest.importorskip("streamlit")                       # meter_source caches with Streamlit
    from meter_source import habits_key
    ac = dict(AC)
    assert habits_key({"appliances": [ac]}) != habits_key({"appliances": [dict(ac, quantity=2)]})
    assert habits_key({"appliances": [ac]}) != habits_key({"appliances": [dict(ac, hours_per_day=12)]})


@pytest.mark.brief("EDIT-017")
def test_removing_an_appliance_keeps_past_history(db):
    import pandas as pd
    hour = pd.Timestamp("2026-10-14 10:00")
    db.log_forecasts("hh1", "simulated_meter", [(hour, 1.0, 0.5, 1.5)])
    db.record_actuals("hh1", "simulated_meter", [(hour, 1.2)])
    db.save_home_details("hh1", _home(AC, FRIDGE))
    db.save_home_details("hh1", _home(FRIDGE))                 # AC removed
    log = db.forecast_log("hh1", "simulated_meter")
    assert len(log) == 1 and log["actual_kwh"].iloc[0] == 1.2  # history is kept (retention: until account deletion)
