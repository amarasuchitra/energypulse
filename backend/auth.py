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
   A new password must be strong: 10+ characters with upper and lower case,
   a digit and a symbol, and not a common password or the email name.
3. Five wrong passwords for an email lock it for five minutes.  The count is
   kept for unknown emails too, so the lock message never reveals whether an
   account exists.
7. Forgotten password: a one-time 8-character code is emailed.  Only its hash
   is stored; it expires after 30 minutes, works once, allows 5 wrong tries,
   and at most 3 codes an hour can be requested per email.  A successful
   reset signs the account out on every browser.
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
MIN_PASSWORD = 10
MAX_FAILURES = 5
LOCK_SECONDS = 300
RESET_SECONDS = 30 * 60
RESET_MAX_TRIES = 5
RESET_PER_HOUR = 3
RESET_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"     # no 0/O or 1/I to misread
RESET_CODE_LEN = 8
MAX_NAME = 80
MAX_EMAIL = 254
MAX_PASSWORD = 256
TERMS_VERSION = "2026-10"
SESSION_SECONDS = 8 * 3600
# "Keep me signed in": a random token in a browser cookie, stored here only as
# a SHA-256 hash, valid for 30 days and revoked on log-out.
REMEMBER_DAYS = 30
REMEMBER_SECONDS = REMEMBER_DAYS * 24 * 3600
COOKIE_NAME = "ep_session"
# Plain ASCII addresses: letters, digits and . _ % + - before the @, a dotted domain after.
_EMAIL = re.compile(r"^[A-Za-z0-9._%+'-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}$")
GUEST_DOMAIN = "guest.energypulse.local"


_COMMON = {"password", "passw0rd", "qwerty", "letmein", "welcome", "admin", "iloveyou",
           "abc123", "123456", "12345678", "111111", "energypulse"}

PASSWORD_RULES = [
    ("length", f"At least {MIN_PASSWORD} characters"),
    ("upper", "An upper-case letter (A-Z)"),
    ("lower", "A lower-case letter (a-z)"),
    ("digit", "A number (0-9)"),
    ("symbol", "A symbol such as ! @ # $ %"),
    ("personal", "Not a common password, and not your email name"),
]


def password_checks(password: str, email: str = "") -> dict:
    """Which strength rules a password meets: {rule key: True/False}."""
    pw = password or ""
    low = pw.lower()
    name = (email or "").split("@")[0].strip().lower()
    letters = re.sub(r"[^a-z]", "", low)
    weak = any(c in low for c in _COMMON) or (len(name) >= 3 and name in low) \
        or len(set(pw)) < 5 or letters in _COMMON
    return {
        "length": len(pw) >= MIN_PASSWORD,
        "upper": bool(re.search(r"[A-Z]", pw)),
        "lower": bool(re.search(r"[a-z]", pw)),
        "digit": bool(re.search(r"\d", pw)),
        "symbol": bool(re.search(r"[^A-Za-z0-9\s]", pw)),
        "personal": bool(pw) and not weak,
    }


def password_problems(password: str, email: str = "") -> list:
    """Plain-language list of the rules a password still fails (empty = strong)."""
    checks = password_checks(password, email)
    return [label for key, label in PASSWORD_RULES if not checks[key]]


_CONTROL = re.compile(r"[\x00-\x1f\x7f<>]")


def clean_name(name: str) -> str:
    """A display name with control characters and angle brackets removed, trimmed to 80 characters."""
    return re.sub(r"\s+", " ", _CONTROL.sub("", name or "")).strip()[:MAX_NAME]


def validate_signup_form(name: str, email: str, password: str, confirm: str, accepted_terms: bool) -> list:
    """
    Everything wrong with a sign-up form, in plain language (empty list = OK to submit).
    Checked before any account is created; Accounts.sign_up checks email and password again.
    """
    problems = []
    if not clean_name(name):
        problems.append("Enter your full name.")
    em = (email or "").strip().lower()
    if not em:
        problems.append("Enter your email address.")
    elif len(em) > MAX_EMAIL or not _EMAIL.match(em) or em.endswith("@" + GUEST_DOMAIN):
        problems.append("Enter a valid email address.")
    if not password:
        problems.append("Choose a password.")
    else:
        if len(password) > MAX_PASSWORD:
            problems.append(f"The password can be at most {MAX_PASSWORD} characters.")
        missing = password_problems(password, em)
        if missing:
            problems.append("Choose a stronger password. Still needed: " + "; ".join(missing) + ".")
        if password != (confirm or ""):
            problems.append("The two passwords do not match. Type the same password in both boxes.")
    if not accepted_terms:
        problems.append("Accept the Terms of Use and Privacy Policy to create an account.")
    return problems


def validate_login_form(email: str, password: str) -> list:
    problems = []
    if not (email or "").strip():
        problems.append("Enter your email address.")
    if not password:
        problems.append("Enter your password.")
    return problems


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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS login_tokens (
                    token_hash TEXT PRIMARY KEY,
                    email TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                )""")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS login_throttle (
                    email TEXT PRIMARY KEY,
                    failures INTEGER NOT NULL DEFAULT 0,
                    locked_until REAL NOT NULL DEFAULT 0
                )""")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS password_resets (
                    code_hash TEXT PRIMARY KEY,
                    email TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    tries INTEGER NOT NULL DEFAULT 0,
                    used INTEGER NOT NULL DEFAULT 0
                )""")
            cols = {r[1] for r in conn.execute("PRAGMA table_info(credentials)")}
            if "terms_accepted" not in cols:
                conn.execute("ALTER TABLE credentials ADD COLUMN terms_accepted TEXT")

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

    def sign_up(self, email: str, password: str, name: str = "",
                terms_version: Optional[str] = None) -> Tuple[bool, str]:
        email = self.normalise(email)
        if len(email) > MAX_EMAIL or not _EMAIL.match(email) or email.endswith("@" + GUEST_DOMAIN):
            return False, "Enter a valid email address."
        if not isinstance(password, str) or len(password) > MAX_PASSWORD:
            return False, f"The password can be at most {MAX_PASSWORD} characters."
        problems = password_problems(password, email)
        if problems:
            return False, "Choose a stronger password. Still needed: " + "; ".join(problems) + "."
        if self.exists(email):
            return False, "An account with this email already exists. Sign in instead."
        salt = secrets.token_bytes(16)
        display = clean_name(name) or email.split("@")[0]
        try:
            with self._conn() as conn:
                conn.execute("INSERT INTO credentials (email, salt, password_hash, name, terms_accepted) "
                             "VALUES (?, ?, ?, ?, ?)",
                             (email, salt.hex(), _hash(password, salt), display, terms_version))
        except sqlite3.IntegrityError:                    # a second request for the same email won the race
            return False, "An account with this email already exists. Sign in instead."
        return True, display

    def sign_in(self, email: str, password: str, now: Optional[float] = None) -> Tuple[bool, str]:
        """(True, display name) or (False, reason).  The reason never says which part was wrong,
        and never whether the email has an account."""
        email = self.normalise(email)[:MAX_EMAIL]
        now = time.time() if now is None else now
        if not isinstance(password, str) or len(password) > MAX_PASSWORD:
            password = ""
        with self._conn() as conn:
            th = conn.execute("SELECT failures, locked_until FROM login_throttle WHERE email = ?",
                              (email,)).fetchone()
            if th is not None and th["locked_until"] > now:
                minutes = int((th["locked_until"] - now) // 60) + 1
                return False, f"Too many wrong attempts. Try again in about {minutes} minute(s)."
            row = conn.execute("SELECT * FROM credentials WHERE email = ?", (email,)).fetchone()
            if row is None:
                _hash(password or "", b"0" * 16)          # same work whether or not the account exists
                good = False
            else:
                good = hmac.compare_digest(row["password_hash"],
                                           _hash(password or "", bytes.fromhex(row["salt"])))
            if good:
                conn.execute("DELETE FROM login_throttle WHERE email = ?", (email,))
                return True, row["name"]
            failures = (th["failures"] if th is not None else 0) + 1
            locked = now + LOCK_SECONDS if failures >= MAX_FAILURES else 0
            conn.execute("""INSERT INTO login_throttle (email, failures, locked_until) VALUES (?, ?, ?)
                            ON CONFLICT(email) DO UPDATE SET failures = excluded.failures,
                                                             locked_until = excluded.locked_until""",
                         (email, 0 if locked else failures, locked))
            return False, "The email or password is not right."

    # ------------------------------------------------------- forgotten password
    @staticmethod
    def _code_hash(email: str, code: str) -> str:
        return hashlib.sha256(("reset|" + email + "|" + (code or "").strip().upper()).encode()).hexdigest()

    def request_reset(self, email: str, now: Optional[float] = None) -> Optional[str]:
        """
        A new one-time reset code for this account, to be emailed; None when the email has
        no account or too many codes were asked for in the last hour.  The caller shows the
        same message either way.
        """
        email = self.normalise(email)
        now = time.time() if now is None else now
        if not _EMAIL.match(email) or not self.exists(email):
            return None
        with self._conn() as conn:
            conn.execute("DELETE FROM password_resets WHERE expires_at < ?", (now - 3600,))
            recent = conn.execute("SELECT COUNT(*) FROM password_resets WHERE email = ? AND created_at > ?",
                                  (email, now - 3600)).fetchone()[0]
            if recent >= RESET_PER_HOUR:
                return None
            conn.execute("UPDATE password_resets SET used = 1 WHERE email = ?", (email,))   # older codes stop working
            code = "".join(secrets.choice(RESET_ALPHABET) for _ in range(RESET_CODE_LEN))
            conn.execute("INSERT INTO password_resets (code_hash, email, created_at, expires_at) VALUES (?, ?, ?, ?)",
                         (self._code_hash(email, code), email, now, now + RESET_SECONDS))
        return code

    def reset_password(self, email: str, code: str, new_password: str,
                       now: Optional[float] = None) -> Tuple[bool, str]:
        """Set a new password with a valid code.  Every failure gives the same answer about the code."""
        email = self.normalise(email)
        now = time.time() if now is None else now
        bad = (False, "That code is not valid or has expired. Ask for a new one.")
        if not code or len(code) > 40:
            return bad
        if not isinstance(new_password, str) or len(new_password) > MAX_PASSWORD:
            return False, f"The password can be at most {MAX_PASSWORD} characters."
        problems = password_problems(new_password, email)
        if problems:
            return False, "Choose a stronger password. Still needed: " + "; ".join(problems) + "."
        with self._conn() as conn:
            live = conn.execute("SELECT * FROM password_resets WHERE email = ? AND used = 0 AND expires_at >= ? "
                                "ORDER BY created_at DESC LIMIT 1", (email, now)).fetchone()
            if live is None:
                return bad
            if not hmac.compare_digest(live["code_hash"], self._code_hash(email, code)):
                tries = live["tries"] + 1
                conn.execute("UPDATE password_resets SET tries = ?, used = ? WHERE code_hash = ?",
                             (tries, 1 if tries >= RESET_MAX_TRIES else 0, live["code_hash"]))
                return bad
            salt = secrets.token_bytes(16)
            conn.execute("UPDATE credentials SET salt = ?, password_hash = ? WHERE email = ?",
                         (salt.hex(), _hash(new_password, salt), email))
            conn.execute("UPDATE password_resets SET used = 1 WHERE email = ?", (email,))
            conn.execute("DELETE FROM login_tokens WHERE email = ?", (email,))      # signed out everywhere
            conn.execute("DELETE FROM login_throttle WHERE email = ?", (email,))
        return True, "Your password has been changed. Sign in with the new password."

    # ------------------------------------------------------- remember-me tokens
    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(("login|" + (token or "")).encode()).hexdigest()

    def issue_token(self, email: str, now: Optional[float] = None,
                    seconds: int = REMEMBER_SECONDS) -> str:
        """A new sign-in token for this account.  Only its hash is kept."""
        email = self.normalise(email)
        now = time.time() if now is None else now
        token = secrets.token_urlsafe(32)
        with self._conn() as conn:
            conn.execute("DELETE FROM login_tokens WHERE expires_at < ?", (now,))
            conn.execute("INSERT INTO login_tokens (token_hash, email, created_at, expires_at) "
                         "VALUES (?, ?, ?, ?)", (self._token_hash(token), email, now, now + seconds))
        return token

    def check_token(self, token: str, now: Optional[float] = None) -> Optional[Tuple[str, str, float]]:
        """(email, display name, issued at) for a valid, unexpired token; otherwise None."""
        if not isinstance(token, str) or not token or len(token) > 200:
            return None
        now = time.time() if now is None else now
        with self._conn() as conn:
            row = conn.execute("""
                SELECT t.email, t.created_at, t.expires_at, c.name
                FROM login_tokens t JOIN credentials c ON c.email = t.email
                WHERE t.token_hash = ?""", (self._token_hash(token),)).fetchone()
        if row is None or row["expires_at"] < now:
            return None
        return row["email"], row["name"], float(row["created_at"])

    def revoke_token(self, token: str) -> None:
        if isinstance(token, str) and token:
            with self._conn() as conn:
                conn.execute("DELETE FROM login_tokens WHERE token_hash = ?", (self._token_hash(token),))

    def revoke_all_tokens(self, email: str) -> int:
        """Sign this account out on every browser."""
        with self._conn() as conn:
            return conn.execute("DELETE FROM login_tokens WHERE email = ?",
                                (self.normalise(email),)).rowcount


def new_guest_email() -> str:
    """A private identity for one guest session."""
    return f"guest-{secrets.token_hex(8)}@{GUEST_DOMAIN}"


def is_guest_email(email: str) -> bool:
    return (email or "").endswith("@" + GUEST_DOMAIN)


def session_expired(signed_in_at: Optional[float], now: Optional[float] = None,
                    remember: bool = False) -> bool:
    """A normal sign-in lasts 8 hours; with "keep me signed in", 30 days."""
    now = time.time() if now is None else now
    limit = REMEMBER_SECONDS if remember else SESSION_SECONDS
    return signed_in_at is None or now - signed_in_at > limit


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
