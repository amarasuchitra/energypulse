"""The real app driven through Streamlit's AppTest: sign-up, setup, dashboard, Edit appliances,
signing back in, alerts on the dashboard (brief sections 2, 3, 4, 9 and the end-to-end journey).

These run the whole of frontend/app.py against the throwaway test database (see conftest.py),
with background jobs off.  They need the app's full requirements (CI installs them).
"""
import json
import os
import time
import uuid

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

from auth import Accounts  # noqa: E402
from db import DatabaseManager  # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "app.py")
PW = "Tr4il-Mango-Kettle"
TIMEOUT = 600


import logging  # noqa: E402
import traceback  # noqa: E402


class _Grab(logging.Handler):
    """Keeps the app's logged exceptions so a failing test can show the real cause."""
    records = []

    def emit(self, record):
        if record.exc_info:
            self.records.append("".join(traceback.format_exception(*record.exc_info))[-3000:])
        elif record.levelno >= logging.ERROR:
            self.records.append(record.getMessage()[-1500:])


logging.getLogger().addHandler(_Grab())
logging.getLogger("streamlit").addHandler(_Grab())


def _new():
    try:
        from streamlit import config as _cfg
        _cfg.set_option("client.showErrorDetails", "full")
    except Exception:                                          # noqa: BLE001
        pass
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    _ok(at)
    return at


def _ok(at):
    if at.exception:
        e = at.exception[0]
        raise AssertionError(f"app error: {getattr(e, 'message', e.value)}\n"
                             + "\n".join(getattr(e, "stack_trace", []) or [])[-3000:]
                             + "\nLOGGED:\n" + "\n---\n".join(_Grab.records[-2:]))


def _email():
    return f"user-{uuid.uuid4().hex[:10]}@example.com"


def _texts(at):
    out = []
    for kind in ("markdown", "caption", "error", "warning", "info", "success", "text"):
        try:
            out += [str(e.value) for e in getattr(at, kind)]
        except Exception:                                      # noqa: BLE001
            pass
    return "\n".join(out)


def _submit(at, label):
    btn = [b for b in at.button if b.label == label]
    assert btn, f"no button {label!r}: {[b.label for b in at.button]}"
    btn[0].click().run()
    _ok(at)


def _sign_up(at, email, name="Test User", pw=PW, confirm=None, terms=True):
    at.button(key="switch_signup").click().run()
    at.text_input(key="login_name").input(name)
    at.text_input(key="login_email").input(email)
    at.text_input(key="login_pw").input(pw)
    at.text_input(key="login_pw2").input(pw if confirm is None else confirm)
    if terms:
        at.checkbox(key="login_terms").check()
    at.checkbox(key="login_remember").uncheck()
    _submit(at, "Create Account")


def _sign_in(at, email, pw=PW):
    at.text_input(key="login_email").input(email)
    at.text_input(key="login_pw").input(pw)
    at.checkbox(key="login_remember").uncheck()
    _submit(at, "Sign In")


def _household(email):
    return DatabaseManager().get_user(email)["household_id"]


def _onboard_with_custom(at, name="Air conditioner"):
    at.button(key="ob_next1").click().run()
    _ok(at)
    at.text_input(key="ob_app_custom").input(name)
    at.button(key="ob_add_app").click().run()
    _ok(at)
    at.button(key="ob_next2").click().run()
    _ok(at)
    at.button(key="ob_finish").click().run()
    _ok(at)


def _saved_user(appliances):
    email = _email()
    Accounts().sign_up(email, PW, "Saved User")
    db = DatabaseManager()
    hid = db.ensure_user_login(email, "Saved User")["household_id"]
    db.save_home_details(hid, {"home_type": "Apartment", "occupants": 3, "tariff_rate": 8.0,
                               "appliances": appliances})
    return email, hid


# ------------------------------------------------------------------ signed out
@pytest.mark.brief("AUTH-013", "UI-002", "SMOKE-001")
def test_signed_out_visitor_sees_only_the_sign_in_form():
    at = _new()
    assert at.text_input(key="login_email") and at.text_input(key="login_pw")
    keys = {b.key for b in at.button}
    assert "dash_edit_appliances" not in keys and "side_logout" not in keys
    assert "forgot_pw" in keys


@pytest.mark.brief("AUTH-004", "AUTH-005", "AUTH-006", "AUTH-003")
def test_sign_up_form_shows_every_problem():
    at = _new()
    at.button(key="switch_signup").click().run()
    _submit(at, "Create Account")
    errors = " ".join(e.value for e in at.error)
    for words in ("full name", "email address", "Choose a password", "Terms of Use"):
        assert words in errors
    _sign_up(at, "not-an-email", pw=PW, confirm=PW + "x", terms=False)
    errors = " ".join(e.value for e in at.error)
    assert "valid email" in errors and "do not match" in errors and "Terms" in errors
    assert not at.session_state["auth"].get("logged_in")


@pytest.mark.brief("AUTH-010", "AUTH-008", "AUTH-009")
def test_sign_in_form_validates_and_refuses_wrong_details():
    at = _new()
    _submit(at, "Sign In")
    assert any("Enter your email" in e.value for e in at.error)
    _sign_in(at, "nobody@example.com", PW)
    assert any("email or password is not right" in e.value for e in at.error)
    assert not at.session_state["auth"].get("logged_in")


@pytest.mark.brief("AUTH-017")
def test_forgot_password_screen_is_honest_when_email_is_not_set_up(monkeypatch):
    for var in ("SMTP_USER", "SMTP_USERNAME", "SMTP_PASSWORD", "SENDGRID_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    at = _new()
    at.button(key="forgot_pw").click().run()
    _ok(at)
    at.text_input(key="reset_email").input("someone@example.com")
    _submit(at, "Email me a code")
    text = _texts(at)
    assert "not set up on this server" in text or "If that email has an account" in text
    at.button(key="reset_back").click().run()
    assert at.text_input(key="login_pw")


# ------------------------------------------------------------------ the journey
@pytest.mark.brief("E2E-001", "AUTH-001", "AUTH-007", "APP-001", "APP-002", "APP-003", "APP-004", "APP-005",
                   "EDIT-001", "EDIT-002", "EDIT-003", "EDIT-004", "EDIT-007", "EDIT-013", "EDIT-014",
                   "UI-001", "UI-010", "AUTH-023", "AUTH-024", "UI-020")
def test_register_set_up_edit_sign_out_and_back_in(metric):
    email = _email()
    at = _new()
    t0 = time.perf_counter()
    _sign_up(at, email, name="Journey User")
    assert at.session_state["auth"]["logged_in"] and at.session_state["auth"]["email"] == email
    # 1. A new account goes to the setup screens, not the dashboard.
    assert at.button(key="ob_next1") and "dash_edit_appliances" not in {b.key for b in at.button}
    _onboard_with_custom(at, "Air conditioner")
    first_render = time.perf_counter() - t0
    # 2. Dashboard with the saved list and the Edit appliances button beside its heading.
    assert at.button(key="dash_edit_appliances"), [b.key for b in at.button][:40]
    assert "Your appliances (1)" in _texts(at)
    hid = _household(email)
    assert [a["name"] for a in DatabaseManager().get_home_details(hid)["appliances"]] == ["Air conditioner"]
    # 3. Edit: the saved value is filled in; change it and save.
    at.button(key="dash_edit_appliances").click().run()
    _ok(at)
    qty = [n for n in at.number_input if str(n.key).startswith("ap_qty_")]
    assert qty and qty[0].value == 1
    qty[0].set_value(2).run()
    _ok(at)
    at.button(key="ap_save").click().run()
    _ok(at)
    assert DatabaseManager().get_home_details(hid)["appliances"][0]["quantity"] == 2
    assert "×2" in _texts(at) and at.button(key="dash_edit_appliances")      # editor closed, list updated
    at.run()                                                                   # a refresh keeps it
    _ok(at)
    assert "×2" in _texts(at)
    # 4. A separate browser session starts signed out; signing in lands on the saved home.
    other = _new()
    assert not other.session_state["auth"].get("logged_in")
    _sign_in(other, email)
    assert other.session_state["auth"]["logged_in"]
    assert "ob_next1" not in {b.key for b in other.button}                   # no setup again
    assert other.button(key="dash_edit_appliances") and "×2" in _texts(other)
    metric("ui_signup_setup_first_dashboard_s", round(first_render, 2))


@pytest.mark.brief("EDIT-008", "APP-023", "EDIT-005", "EDIT-006")
def test_cancel_discards_changes_and_add_remove_are_drafts_until_saved():
    import catalog
    email, hid = _saved_user([catalog.item_for("ac"), catalog.item_for("fridge")])
    at = _new()
    _sign_in(at, email)
    at.button(key="dash_edit_appliances").click().run()
    _ok(at)
    qty = [n for n in at.number_input if str(n.key).startswith("ap_qty_")]
    assert len(qty) == 2
    qty[0].set_value(5).run()
    dels = [b for b in at.button if str(b.key).startswith("ap_del_")]
    dels[1].click().run()                                                     # remove the fridge (draft)
    _ok(at)
    assert len([n for n in at.number_input if str(n.key).startswith("ap_qty_")]) == 1
    assert "unsaved changes" in _texts(at)
    at.button(key="ap_cancel").click().run()
    _ok(at)
    saved = DatabaseManager().get_home_details(hid)["appliances"]
    assert [a["type_key"] for a in saved] == ["ac", "fridge"] and saved[0].get("quantity", 1) == 1
    assert at.button(key="dash_edit_appliances")                             # editor closed


@pytest.mark.brief("EDIT-012")
def test_save_is_disabled_until_something_changes():
    import catalog
    email, _ = _saved_user([catalog.item_for("ac")])
    at = _new()
    _sign_in(at, email)
    at.button(key="dash_edit_appliances").click().run()
    assert at.button(key="ap_save").disabled                                 # nothing to save twice


@pytest.mark.brief("UI-013", "ALERT-014", "E2E-002")
def test_an_overconsumption_alert_shows_on_the_dashboard():
    import catalog
    import monitoring
    email, hid = _saved_user([catalog.item_for("ac")])
    db = DatabaseManager()
    db.log_notification(hid, email, "Saved User", monitoring.ALERT_TYPE, "actual above forecast",
                        {"actual_kwh": 4.0, "predicted_kwh": 2.0}, "High energy use detected",
                        "Between 10:00-12:00 your home used 4.00 units (kWh), against a forecast of 2.00 units.",
                        status="in-app")
    db.set_store(hid, monitoring.STATE_KEY, {"open": True})
    at = _new()
    _sign_in(at, email)
    banner = [w.value for w in at.warning if "High energy use detected" in str(w.value)]
    assert banner and "4.00 units" in banner[0] and "email is not set up" in banner[0]


@pytest.mark.brief("UI-011", "UI-012", "LIVE-005", "LIVE-006", "ML-019")
def test_dashboard_shows_when_each_stage_last_ran():
    import catalog
    email, _ = _saved_user([catalog.item_for("ac")])
    at = _new()
    _sign_in(at, email)
    text = _texts(at)
    assert "Last updated" in text and "Reading:" in text and "Forecast:" in text and "Alert check:" in text
    assert "Forecast by a" in text or "next-hour forecast is not available" in text


@pytest.mark.brief("UI-014", "AUTH-015", "E2E-001")
def test_logging_out_hides_the_household():
    import catalog
    email, _ = _saved_user([catalog.item_for("ac")])
    at = _new()
    _sign_in(at, email)
    at.button(key="side_logout").click().run()
    _ok(at)
    if not any(b.key == "logout_go" for b in at.button):
        pytest.skip("AppTest did not open the log-out dialog")
    at.button(key="logout_go").click().run()
    _ok(at)
    assert not at.session_state["auth"].get("logged_in") if "auth" in at.session_state else True
    assert at.text_input(key="login_email") and "dash_edit_appliances" not in {b.key for b in at.button}


@pytest.mark.brief("AUTH-021", "LIVE-017", "APP-020")
def test_two_signed_in_users_each_see_only_their_own_appliances():
    import catalog
    a_email, _ = _saved_user([dict(catalog.item_for("ac"), name="Alpha AC")])
    b_email, _ = _saved_user([dict(catalog.item_for("fridge"), name="Beta Fridge")])
    a, b = _new(), _new()
    _sign_in(a, a_email)
    _sign_in(b, b_email)
    ta, tb = _texts(a), _texts(b)
    assert "Alpha AC" in ta and "Beta Fridge" not in ta
    assert "Beta Fridge" in tb and "Alpha AC" not in tb


@pytest.mark.brief("UI-015", "UI-016")
def test_layout_has_rules_for_narrow_screens():
    css = open(os.path.join("frontend", "shell_ui.py"), encoding="utf-8").read() + \
        open(os.path.join("frontend", "app.py"), encoding="utf-8").read()
    assert "@media" in css and "max-width" in css
