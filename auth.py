"""
Accounts and access
===================
Who may sign in, and which household's data they may see.

Rules
-----
1. Passwords are never stored.  Each account keeps a random salt and a
   PBKDF2-HMAC-SHA256 hash (310,000 rounds); sign-in recomputes and compares
   in constant time.
2. An email can be registered once.  Signing up again with someone else's
   email does not give access to their household.
3. Five wrong passwords lock the account for five minutes.
4. Every page reads data by the household id of the SIGNED-IN account.  The
   id comes from the session, never from anything typed into a form.
5. Each guest gets a private, throwaway household, so guests cannot see one
   another's bills, alerts or family list.
6. Saved bill files are encrypted on disk (see protect()/unprotect()).

Not covered: family members listed on the Family page do not get their own
sign-in; one account owns each household.
"""

import base64
import hashlib
import hmac
import re
import secrets
import sqlite3
import time
from contextlib import contextmanager
from typing import Optional, Tuple

from db import DB_PATH

ROUNDS = 310_000
MIN_PASSWORD = 8
MAX_FAILURES = 5
LOCK_SECONDS = 300
SESSION_SECONDS = 8 * 3600
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
GUEST_DOMAIN = "guest.energypulse.local"


def _hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ROUNDS).hex()


class Accounts:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS credentials (
                    email TEXT PRIMARY KEY,
                    salt TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    name TEXT NOT NULL,
                    failures INTEGER NOT NULL DEFAULT 0,
                    locked_until REAL NOT NULL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )""")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def normalise(email: str) -> str:
        return (email or "").strip().lower()

    def exists(self, email: str) -> bool:
        with self._conn() as conn:
            return conn.execute("SELECT 1 FROM credentials WHERE email = ?",
                                (self.normalise(email),)).fetchone() is not None

    def sign_up(self, email: str, password: str, name: str = "") -> Tuple[bool, str]:
        email = self.normalise(email)
        if not _EMAIL.match(email) or email.endswith("@" + GUEST_DOMAIN):
            return False, "Enter a valid email address."
        if len(password or "") < MIN_PASSWORD:
            return False, f"Use a password of at least {MIN_PASSWORD} characters."
        if self.exists(email):
            return False, "An account with this email already exists. Sign in instead."
        salt = secrets.token_bytes(16)
        display = (name or "").strip()[:80] or email.split("@")[0]
        with self._conn() as conn:
            conn.execute("INSERT INTO credentials (email, salt, password_hash, name) VALUES (?, ?, ?, ?)",
                         (email, salt.hex(), _hash(password, salt), display))
        return True, display

    def sign_in(self, email: str, password: str, now: Optional[float] = None) -> Tuple[bool, str]:
        """(True, display name) or (False, reason).  The reason never says which part was wrong."""
        email = self.normalise(email)
        now = time.time() if now is None else now
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM credentials WHERE email = ?", (email,)).fetchone()
            if row is None:
                _hash(password or "", b"0" * 16)          # same work whether or not the account exists
                return False, "The email or password is not right."
            if row["locked_until"] > now:
                minutes = int((row["locked_until"] - now) // 60) + 1
                return False, f"Too many wrong attempts. Try again in about {minutes} minute(s)."
            good = hmac.compare_digest(row["password_hash"],
                                       _hash(password or "", bytes.fromhex(row["salt"])))
            if good:
                conn.execute("UPDATE credentials SET failures = 0, locked_until = 0 WHERE email = ?", (email,))
                return True, row["name"]
            failures = row["failures"] + 1
            locked = now + LOCK_SECONDS if failures >= MAX_FAILURES else 0
            conn.execute("UPDATE credentials SET failures = ?, locked_until = ? WHERE email = ?",
                         (0 if locked else failures, locked, email))
            return False, "The email or password is not right."


def new_guest_email() -> str:
    """A private identity for one guest session."""
    return f"guest-{secrets.token_hex(8)}@{GUEST_DOMAIN}"


def is_guest_email(email: str) -> bool:
    return (email or "").endswith("@" + GUEST_DOMAIN)


def session_expired(signed_in_at: Optional[float], now: Optional[float] = None) -> bool:
    now = time.time() if now is None else now
    return signed_in_at is None or now - signed_in_at > SESSION_SECONDS


# ----------------------------------------------------------------- files at rest
def _fernet():
    from cryptography.fernet import Fernet
    from integrity import _key
    return Fernet(base64.urlsafe_b64encode(hashlib.sha256(b"bills|" + _key()).digest()))


def protect(data: bytes) -> bytes:
    """Encrypt a file before it is written to disk."""
    return _fernet().encrypt(data)


def unprotect(blob: bytes) -> Optional[bytes]:
    """Decrypt a stored file; None if it was altered or the key is wrong."""
    try:
        return _fernet().decrypt(blob)
    except Exception:
        return None
