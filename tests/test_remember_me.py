"""Keep-me-signed-in tokens: 30 days, hashed at rest, revocable."""
import time

from auth import REMEMBER_SECONDS, SESSION_SECONDS, Accounts, session_expired
from db import DatabaseManager

PW = "Monsoon#Kettle42"


def _accounts(tmp_path):
    acc = Accounts(db_path=str(tmp_path / "a.db"))
    assert acc.sign_up("ravi@example.com", PW, "Ravi")[0]
    return acc


def test_token_signs_in_and_is_stored_only_as_a_hash(tmp_path):
    acc = _accounts(tmp_path)
    token = acc.issue_token("Ravi@Example.com")
    assert acc.check_token(token)[:2] == ("ravi@example.com", "Ravi")
    with acc._conn() as conn:
        stored = [r["token_hash"] for r in conn.execute("SELECT token_hash FROM login_tokens")]
    assert stored and token not in stored


def test_token_expires_after_30_days(tmp_path):
    acc = _accounts(tmp_path)
    now = time.time()
    token = acc.issue_token("ravi@example.com", now=now)
    assert acc.check_token(token, now=now + REMEMBER_SECONDS - 60)
    assert acc.check_token(token, now=now + REMEMBER_SECONDS + 60) is None


def test_logout_revokes_the_token(tmp_path):
    acc = _accounts(tmp_path)
    token = acc.issue_token("ravi@example.com")
    acc.revoke_token(token)
    assert acc.check_token(token) is None


def test_sign_out_everywhere_and_unknown_tokens(tmp_path):
    acc = _accounts(tmp_path)
    t1, t2 = acc.issue_token("ravi@example.com"), acc.issue_token("ravi@example.com")
    assert acc.revoke_all_tokens("ravi@example.com") == 2
    assert acc.check_token(t1) is None and acc.check_token(t2) is None
    assert acc.check_token("") is None and acc.check_token("not-a-token") is None
    assert acc.check_token("x" * 500) is None


def test_session_length_depends_on_remember_me():
    now = time.time()
    assert session_expired(now - SESSION_SECONDS - 1, now=now)
    assert not session_expired(now - SESSION_SECONDS - 1, now=now, remember=True)
    assert session_expired(now - REMEMBER_SECONDS - 1, now=now, remember=True)
    assert session_expired(None, now=now, remember=True)


def test_deleting_a_household_removes_its_tokens(tmp_path):
    path = str(tmp_path / "shared.db")
    db = DatabaseManager(db_path=path)
    acc = Accounts(db_path=path)
    acc.sign_up("ravi@example.com", PW, "Ravi")
    db.create_user("ravi@example.com", "Ravi", PW, household_id="hh9")
    token = acc.issue_token("ravi@example.com")
    db.delete_household("hh9")
    assert acc.check_token(token) is None
