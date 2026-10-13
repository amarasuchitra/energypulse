"""Appliance configuration: validation, persistence, per-user isolation, and its effect on the meter."""
import pytest

import catalog
from db import DatabaseManager
from home_config import InvalidSetup, expected_daily_kwh, usage_factor, validate_home_details


def _home(**over):
    base = {"home_type": "Apartment", "occupants": 3, "home_size": 900, "tariff_rate": 7.5, "city": "Pune",
            "appliances": [catalog.item_for("fridge"), catalog.item_for("ac")]}
    base.update(over)
    return base


def test_valid_setup_passes_unchanged():
    clean, problems = validate_home_details(_home())
    assert problems == []
    assert [a["type_key"] for a in clean["appliances"]] == ["fridge", "ac"]
    assert all(a["quantity"] == 1 for a in clean["appliances"])


@pytest.mark.parametrize("field,value", [("occupants", 0), ("occupants", 99), ("tariff_rate", -3),
                                         ("home_size", 10 ** 7)])
def test_out_of_range_numbers_are_reported_and_clamped(field, value):
    clean, problems = validate_home_details(_home(**{field: value}))
    assert problems
    from home_config import LIMITS
    lo, hi = LIMITS[field]
    assert lo <= clean[field] <= hi


def test_bad_appliance_values_are_reported():
    ac = {**catalog.item_for("ac"), "quantity": 50, "hours_per_day": 30}
    clean, problems = validate_home_details(_home(appliances=[ac]))
    assert len(problems) == 2
    assert clean["appliances"][0]["quantity"] == 10 and clean["appliances"][0]["hours_per_day"] == 24


def test_duplicate_catalog_items_merge_into_a_quantity():
    fan = catalog.item_for("fan")
    clean, _ = validate_home_details(_home(appliances=[fan, dict(fan), dict(fan)]))
    assert len(clean["appliances"]) == 1 and clean["appliances"][0]["quantity"] == 3


def test_unknown_fields_are_dropped_and_names_required():
    clean, problems = validate_home_details(_home(appliances=[{"name": "  ", "evil": "<script>"}]))
    assert "evil" not in clean["appliances"][0]
    assert any("name" in p for p in problems)


def test_garbage_is_refused():
    with pytest.raises(InvalidSetup):
        validate_home_details("not a dict")
    with pytest.raises(InvalidSetup):
        validate_home_details(_home(appliances="fridge"))


def test_usage_and_quantity_change_the_meter_factor():
    ac = catalog.item_for("ac")
    typical = catalog.get_type("ac").hours_per_day
    assert usage_factor({**ac, "hours_per_day": typical}) == 1.0
    assert usage_factor({**ac, "hours_per_day": typical, "quantity": 2}) == 2.0
    assert usage_factor({**ac, "hours_per_day": 0}) == 0.2           # clipped, never zero
    assert expected_daily_kwh({**ac, "quantity": 2}) == pytest.approx(2 * expected_daily_kwh(ac), rel=1e-3)


def test_saved_setup_survives_a_new_connection_and_is_isolated(tmp_path):
    path = str(tmp_path / "h.db")
    db = DatabaseManager(db_path=path)
    a = _home(appliances=[{**catalog.item_for("ac"), "quantity": 2, "hours_per_day": 5}])
    b = _home(appliances=[catalog.item_for("fridge")], occupants=1)
    assert db.save_home_details("house-a", a) == []
    db.save_home_details("house-b", b)

    fresh = DatabaseManager(db_path=path)                 # like logging in again later
    got_a, got_b = fresh.get_home_details("house-a"), fresh.get_home_details("house-b")
    assert got_a["appliances"][0]["quantity"] == 2 and got_a["appliances"][0]["hours_per_day"] == 5
    assert [x["type_key"] for x in got_b["appliances"]] == ["fridge"] and got_b["occupants"] == 1
    assert fresh.get_home_details("house-c") is None

    # Removing an appliance in A leaves B alone.
    fresh.save_home_details("house-a", {**got_a, "appliances": []})
    assert DatabaseManager(db_path=path).get_home_details("house-a")["appliances"] == []
    assert len(DatabaseManager(db_path=path).get_home_details("house-b")["appliances"]) == 1


def test_save_needs_a_household(tmp_path):
    db = DatabaseManager(db_path=str(tmp_path / "x.db"))
    with pytest.raises(ValueError):
        db.save_home_details("", _home())
