"""Sign-up, sign-in, sessions and password reset (brief section 2).  Each test uses its own database."""
import logging
import sqlite3

import pytest

import auth
from auth import (Accounts, LOCK_SECONDS, MAX_FAILURES, RESET_MAX_TRIES, RESET_PER_HOUR, RESET_SECONDS,
                  session_expired, validate_login_form, validate_signup_form)

PW = "Tr4il-Mango-Kettle"
NEW_PW = "Br1ght-Ocean-Lamp"


@pytest.fixture
def acc(tmp_path):
    return Accounts(db_path=str(tmp_path / "auth.db"))


def _good_form(**kw):
    form = dict(name="Amara Suchitra", email="amara@example.com", password=PW, confirm=PW, accepted_terms=True)
    form.update(kw)
    return form


@pytest.mark.brief("AUTH-001")
def test_register_with_valid_details_creates_account(acc):
    assert validate_signup_form(**_good_form()) == []
    ok, name = acc.sign_up("amara@example.com", PW, "Amara Suchitra", terms_version=auth.TERMS_VERSION)
    assert ok and name == "Amara Suchitra" and acc.exists("amara@example.com")
    with sqlite3.connect(acc.db_path) as c:
        assert c.execute("SELECT terms_accepted FROM credentials").fetchone()[0] == auth.TERMS_VERSION


@pytest.mark.brief("AUTH-002", "APP-017")
def test_duplicate_email_is_rejected_case_and_space_insensitive(acc):
    assert acc.sign_up("amara@example.com", PW, "A")[0]
    ok, msg = acc.sign_up("  AMARA@Example.com ", NEW_PW, "Someone else")
    assert not ok and "already exists" in msg
    assert acc.sign_in("amara@example.com", PW) == (True, "A")          # original password unchanged


@pytest.mark.brief("AUTH-003")
@pytest.mark.parametrize("email", ["plainaddress", "a@b", "@example.com", "amara@", "a b@example.com",
                                   "x@guest.energypulse.local", "a" * 250 + "@example.com"])
def test_invalid_email_is_rejected(acc, email):
    assert any("valid email" in p for p in validate_signup_form(**_good_form(email=email)))
    assert not acc.sign_up(email, PW, "A")[0]


@pytest.mark.brief("AUTH-004")
def test_empty_registration_form_lists_every_required_field():
    problems = validate_signup_form("", "", "", "", False)
    text = " ".join(problems)
    for needed in ("full name", "email address", "Choose a password", "Terms of Use"):
        assert needed in text


@pytest.mark.brief("AUTH-005")
def test_password_and_confirmation_must_match():
    assert any("do not match" in p for p in validate_signup_form(**_good_form(confirm=PW + "x")))


@pytest.mark.brief("AUTH-006")
@pytest.mark.parametrize("weak", ["short1!A", "alllowercase1!", "ALLUPPERCASE1!", "NoDigits!!xx",
                                  "NoSymbol1234ab", "Password123!", "amaraXyz1!abc"])
def test_weak_passwords_are_rejected_with_guidance(acc, weak):
    ok, msg = acc.sign_up("amara@example.com", weak, "A")
    assert not ok and "Still needed" in msg
    assert not acc.exists("amara@example.com")


@pytest.mark.brief("AUTH-007")
def test_login_with_valid_credentials(acc):
    acc.sign_up("amara@example.com", PW, "Amara")
    assert acc.sign_in(" Amara@Example.COM ", PW) == (True, "Amara")


@pytest.mark.brief("AUTH-008")
def test_wrong_password_is_denied_with_tries_left(acc):
    acc.sign_up("amara@example.com", PW, "Amara")
    ok, msg, code = acc.sign_in_detail("amara@example.com", PW + "x")
    assert not ok and code == Accounts.WRONG_PASSWORD
    assert "Wrong password" in msg and f"{MAX_FAILURES - 1} tries left" in msg and "Forgot password" in msg


@pytest.mark.brief("AUTH-009")
def test_unknown_email_is_told_to_create_an_account(acc):
    ok, msg, code = acc.sign_in_detail("nobody@example.com", PW)
    assert not ok and code == Accounts.NO_ACCOUNT
    assert "no account for nobody@example.com" in msg and "create an account" in msg
    ok, msg, code = acc.sign_in_detail("not-an-email", PW)
    assert not ok and code == Accounts.INVALID_EMAIL and "not a valid email" in msg


@pytest.mark.brief("AUTH-010")
def test_empty_login_fields_are_validated():
    assert validate_login_form("", "") == ["Enter your email address.", "Enter your password."]
    assert validate_login_form("a@example.com", "") == ["Enter your password."]
    assert validate_login_form("a@example.com", "x") == []


@pytest.mark.brief("AUTH-011", "AUTH-020", "SEC-001")
@pytest.mark.parametrize("bad", ["' OR '1'='1", "amara@example.com'; DROP TABLE credentials;--",
                                 "<script>alert(1)</script>@x.com", "\x00\x01", "a" * 5000, "ü@ü.ü", None])
def test_malformed_and_injection_input_is_handled_safely(acc, bad):
    acc.sign_up("amara@example.com", PW, "Amara")
    ok, _ = acc.sign_in(bad, bad)
    assert not ok
    assert acc.sign_in("amara@example.com", PW)[0]                      # table intact, account still works
    assert not acc.sign_up(bad if isinstance(bad, str) else "", PW, "x")[0]
    assert validate_signup_form(**_good_form(email=bad or ""))           # the form refuses it too


@pytest.mark.brief("AUTH-020", "SEC-002")
def test_script_in_name_is_stripped_before_storing(acc):
    ok, name = acc.sign_up("amara@example.com", PW, "<script>alert(1)</script> Amara\x07")
    assert ok and "<" not in name and ">" not in name and "\x07" not in name
    assert acc.sign_in("amara@example.com", PW)[1] == name


@pytest.mark.brief("AUTH-012")
def test_repeated_failures_lock_the_account_then_it_unlocks(acc):
    acc.sign_up("amara@example.com", PW, "Amara")
    for i in range(MAX_FAILURES):
        ok, msg, code = acc.sign_in_detail("amara@example.com", "Wrong-Pass-123", now=1000.0)
    assert code == Accounts.LOCKED and "locked for 5 minutes" in msg          # the 5th wrong try says so
    ok, msg, code = acc.sign_in_detail("amara@example.com", PW, now=1001.0)   # even the right password waits
    assert not ok and code == Accounts.LOCKED and "5 more minute(s)" in msg
    assert acc.sign_in("amara@example.com", PW, now=1000.0 + LOCK_SECONDS + 1)[0]


@pytest.mark.brief("AUTH-012", "SEC-003")
def test_unknown_emails_are_never_counted_or_locked(acc):
    for _ in range(MAX_FAILURES + 2):
        ok, msg, code = acc.sign_in_detail("nobody@example.com", "x", now=1000.0)
    assert code == Accounts.NO_ACCOUNT                                       # still told to create one
    import sqlite3
    with sqlite3.connect(acc.db_path) as c:
        assert c.execute("SELECT COUNT(*) FROM login_throttle").fetchone()[0] == 0


@pytest.mark.brief("AUTH-015", "AUTH-016", "SEC-004")
def test_logout_revokes_token_and_it_cannot_be_reused(acc):
    acc.sign_up("amara@example.com", PW, "Amara")
    token = acc.issue_token("amara@example.com")
    assert acc.check_token(token)[0] == "amara@example.com"
    acc.revoke_token(token)
    assert acc.check_token(token) is None


@pytest.mark.brief("AUTH-023", "AUTH-024", "SEC-005")
def test_session_lifetime_and_independent_browsers(acc):
    acc.sign_up("amara@example.com", PW, "Amara")
    t1, t2 = acc.issue_token("amara@example.com", now=0.0), acc.issue_token("amara@example.com", now=0.0)
    assert t1 != t2
    acc.revoke_token(t1)                                                # logging out one browser ...
    assert acc.check_token(t2, now=1.0) is not None                     # ... leaves the other signed in
    assert acc.check_token(t2, now=auth.REMEMBER_SECONDS + 1) is None   # remembered sign-in expires
    assert not session_expired(0.0, now=auth.SESSION_SECONDS - 1)
    assert session_expired(0.0, now=auth.SESSION_SECONDS + 1)
    assert session_expired(None)


@pytest.mark.brief("SEC-006")
def test_only_token_hashes_are_stored(acc):
    acc.sign_up("amara@example.com", PW, "Amara")
    token = acc.issue_token("amara@example.com")
    with sqlite3.connect(acc.db_path) as c:
        stored = [r[0] for r in c.execute("SELECT token_hash FROM login_tokens")]
        cred = c.execute("SELECT password_hash, salt FROM credentials").fetchone()
    assert token not in stored and len(stored[0]) == 64
    assert PW not in cred[0] and len(cred[0]) == 64 and len(cred[1]) == 32      # PBKDF2 hash + random salt


@pytest.mark.brief("AUTH-017")
def test_reset_code_flow_changes_password_and_signs_out_everywhere(acc):
    acc.sign_up("amara@example.com", PW, "Amara")
    remembered = acc.issue_token("amara@example.com")
    code = acc.request_reset("amara@example.com", now=100.0)
    assert code and len(code) == auth.RESET_CODE_LEN
    with sqlite3.connect(acc.db_path) as c:
        assert code not in str(c.execute("SELECT * FROM password_resets").fetchall())     # only a hash kept
    ok, msg = acc.reset_password("amara@example.com", code.lower(), NEW_PW, now=200.0)
    assert ok, msg
    assert acc.sign_in("amara@example.com", NEW_PW)[0]
    assert not acc.sign_in("amara@example.com", PW)[0]
    assert acc.check_token(remembered) is None


@pytest.mark.brief("AUTH-017", "AUTH-009")
def test_reset_request_for_unknown_email_gives_no_code(acc):
    assert acc.request_reset("nobody@example.com") is None


@pytest.mark.brief("AUTH-018")
def test_expired_reset_code_is_rejected(acc):
    acc.sign_up("amara@example.com", PW, "Amara")
    code = acc.request_reset("amara@example.com", now=0.0)
    ok, msg = acc.reset_password("amara@example.com", code, NEW_PW, now=RESET_SECONDS + 1)
    assert not ok and "expired" in msg and "Ask for a new code" in msg
    assert acc.sign_in("amara@example.com", PW)[0]


@pytest.mark.brief("AUTH-019")
def test_reset_code_works_only_once(acc):
    acc.sign_up("amara@example.com", PW, "Amara")
    code = acc.request_reset("amara@example.com", now=0.0)
    assert acc.reset_password("amara@example.com", code, NEW_PW, now=1.0)[0]
    ok, msg = acc.reset_password("amara@example.com", code, "An0ther-Good-Pass", now=2.0)
    assert not ok and "already been used" in msg
    assert acc.sign_in("amara@example.com", NEW_PW)[0]


@pytest.mark.brief("AUTH-019", "SEC-007")
def test_reset_code_guessing_is_limited_and_requests_are_rate_limited(acc):
    acc.sign_up("amara@example.com", PW, "Amara")
    code = acc.request_reset("amara@example.com", now=0.0)
    for _ in range(RESET_MAX_TRIES):
        assert not acc.reset_password("amara@example.com", "WRONGCOD", NEW_PW, now=1.0)[0]
    assert not acc.reset_password("amara@example.com", code, NEW_PW, now=2.0)[0]    # burnt after 5 wrong tries
    for _ in range(RESET_PER_HOUR - 1):
        assert acc.request_reset("amara@example.com", now=10.0)
    assert acc.request_reset("amara@example.com", now=11.0) is None                 # 3 an hour
    assert acc.request_reset("amara@example.com", now=10.0 + 3601) is not None


@pytest.mark.brief("AUTH-018")
def test_new_code_replaces_older_ones(acc):
    acc.sign_up("amara@example.com", PW, "Amara")
    old = acc.request_reset("amara@example.com", now=0.0)
    new = acc.request_reset("amara@example.com", now=1.0)
    assert not acc.reset_password("amara@example.com", old, NEW_PW, now=2.0)[0]
    assert acc.reset_password("amara@example.com", new, NEW_PW, now=3.0)[0]


@pytest.mark.brief("AUTH-006")
def test_reset_still_enforces_password_strength(acc):
    acc.sign_up("amara@example.com", PW, "Amara")
    code = acc.request_reset("amara@example.com")
    ok, msg = acc.reset_password("amara@example.com", code, "weak")
    assert not ok and "Still needed" in msg


@pytest.mark.brief("AUTH-022")
def test_auth_store_failure_raises_cleanly_for_the_ui_to_report(tmp_path):
    acc = Accounts(db_path=str(tmp_path / "a.db"))
    acc.db_path = str(tmp_path / "missing-dir" / "x.db")              # database becomes unreachable
    with pytest.raises(sqlite3.OperationalError):
        acc.sign_in("amara@example.com", PW)
    # app.render_login_screen wraps this call and shows a plain message (checked in test_brief_ui).


@pytest.mark.brief("AUTH-025", "SEC-008")
def test_passwords_and_codes_never_reach_the_logs(acc, caplog):
    caplog.set_level(logging.DEBUG)
    acc.sign_up("amara@example.com", PW, "Amara")
    acc.sign_in("amara@example.com", PW)
    acc.sign_in("amara@example.com", "Wrong-Pass-123")
    code = acc.request_reset("amara@example.com")
    acc.reset_password("amara@example.com", code, NEW_PW)
    text = caplog.text
    for secret in (PW, NEW_PW, code, "Wrong-Pass-123"):
        assert secret not in text


@pytest.mark.brief("SEC-009")
def test_legacy_user_rows_never_hold_a_guessable_password(tmp_path):
    from db import DatabaseManager
    db = DatabaseManager(db_path=str(tmp_path / "u.db"))
    user = db.ensure_user_login("amara@example.com", "Amara")
    assert db.authenticate_user("amara@example.com", "amara@example.com" + user["household_id"]) is None
    # A row written the old way is replaced on the next sign-in.
    salt = bytes(16)
    old = db._hash_password("amara@example.com" + user["household_id"], salt, db.PBKDF2_ALGO, db.PBKDF2_ROUNDS)
    with db.get_connection() as c:
        c.execute("UPDATE users SET password_hash = ?, salt = ? WHERE email = ?", (old, salt.hex(), "amara@example.com"))
    db.ensure_user_login("amara@example.com", "Amara")
    assert db.authenticate_user("amara@example.com", "amara@example.com" + user["household_id"]) is None


@pytest.mark.brief("AUTH-002", "SEC-010")
def test_simultaneous_signups_for_one_email_create_one_account(acc):
    import threading
    results = []

    def go(pw):
        results.append(acc.sign_up("race@example.com", pw, "R")[0])
    threads = [threading.Thread(target=go, args=(p,)) for p in (PW, NEW_PW, "Thr3e-Good-Pass!")]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results.count(True) == 1
    with sqlite3.connect(acc.db_path) as c:
        assert c.execute("SELECT COUNT(*) FROM credentials WHERE email='race@example.com'").fetchone()[0] == 1


@pytest.mark.brief("AUTH-011", "SEC-005")
@pytest.mark.parametrize("token", [None, "", 123, b"bytes", ["list"], "x" * 500, "not-a-real-token"])
def test_odd_sign_in_cookies_are_refused_without_crashing(acc, token):
    assert acc.check_token(token) is None
    acc.revoke_token(token)                                             # no error either


@pytest.mark.brief("AUTH-017", "AUTH-019")
def test_reset_errors_say_exactly_what_is_wrong(acc):
    assert "no account" in acc.reset_password("nobody@example.com", "ABCDEFGH", NEW_PW)[1]
    acc.sign_up("amara@example.com", PW, "Amara")
    assert "No reset code was asked" in acc.reset_password("amara@example.com", "ABCDEFGH", NEW_PW)[1]
    acc.request_reset("amara@example.com", now=0.0)
    msg = acc.reset_password("amara@example.com", "WRONGCOD", NEW_PW, now=1.0)[1]
    assert "code is wrong" in msg and "4 tries left" in msg
    for _ in range(4):
        msg = acc.reset_password("amara@example.com", "WRONGCOD", NEW_PW, now=1.0)[1]
    assert "No tries left" in msg
    assert "typed wrong 5 times" in acc.reset_password("amara@example.com", "WRONGCOD", NEW_PW, now=2.0)[1]
    assert "8-character code" in acc.reset_password("amara@example.com", "", NEW_PW)[1]
