"""
Permanent storage for the app's database on a host whose disk is wiped on every restart
(Streamlit Community Cloud).

The app keeps working on its local SQLite file, exactly as before.  When DATABASE_URL (a Postgres
connection string, e.g. a free Neon database) is set in the app's Secrets:

  on start      the latest saved copy is downloaded and written to the local file BEFORE anything
                opens it, so accounts, homes, meter readings, bills and alerts come back
  while running a background thread saves a consistent copy (SQLite backup API) to Postgres
                within SAVE_EVERY seconds of any change, and once more when the app stops

The copy is one row in table energypulse_snapshot (id, saved_at, size, data).  Nothing else in the
app changes, and without DATABASE_URL this module does nothing.
"""
import atexit
import hashlib
import os
import sqlite3
import threading
import time
from typing import Optional

from ep_log import get_logger

log = get_logger("cloud_store")

SAVE_EVERY = 15                     # seconds between checks for changes
TABLE = "energypulse_snapshot"
_state = {"started": False, "last_hash": None, "last_error": None, "restored": None, "saved_at": None}
_lock = threading.Lock()


def database_url() -> str:
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        try:
            import streamlit as st
            url = str(st.secrets.get("DATABASE_URL", "") or "").strip()
        except Exception:
            url = ""
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url


def _connect(url: str):
    try:
        import psycopg2
        return psycopg2.connect(url, connect_timeout=10)
    except ImportError:
        import psycopg
        return psycopg.connect(url, connect_timeout=10)


def _ensure_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(f"CREATE TABLE IF NOT EXISTS {TABLE} (id INTEGER PRIMARY KEY, saved_at TIMESTAMPTZ "
                    "DEFAULT now(), size INTEGER, sha256 TEXT, data BYTEA NOT NULL)")
    conn.commit()


def snapshot_bytes(db_path: str) -> Optional[bytes]:
    """A consistent copy of the SQLite database (safe while the app is writing)."""
    if not os.path.exists(db_path):
        return None
    src = sqlite3.connect(db_path, timeout=10)
    try:
        mem = sqlite3.connect(":memory:")
        src.backup(mem)
        data = mem.serialize()
        mem.close()
        return bytes(data)
    finally:
        src.close()


def restore(db_path: str, url: Optional[str] = None, connect=None) -> str:
    """Write the saved copy to `db_path`.  Returns 'restored', 'none saved yet', 'off' or 'error: ...'."""
    url = url if url is not None else database_url()
    if not url:
        return "off"
    connect = connect or _connect
    try:
        conn = connect(url)
        try:
            _ensure_table(conn)
            with conn.cursor() as cur:
                cur.execute(f"SELECT data FROM {TABLE} WHERE id = 1")
                row = cur.fetchone()
        finally:
            conn.close()
    except Exception as exc:
        log.warning("Could not reach the permanent database: %s", type(exc).__name__)
        return f"error: {type(exc).__name__}"
    if not row or not row[0]:
        return "none saved yet"
    data = bytes(row[0])
    if data[:16] != b"SQLite format 3\x00":
        log.warning("The saved copy is not a database; ignored")
        return "error: bad copy"
    tmp = db_path + ".restore"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, db_path)
    _state["last_hash"] = hashlib.sha256(data).hexdigest()
    log.info("Database restored from permanent storage (%d bytes)", len(data))
    return "restored"


def save(db_path: str, url: Optional[str] = None, connect=None, force: bool = False) -> str:
    """Save a copy if it changed.  Returns 'saved', 'unchanged', 'off' or 'error: ...'."""
    url = url if url is not None else database_url()
    if not url:
        return "off"
    with _lock:
        try:
            data = snapshot_bytes(db_path)
        except Exception as exc:
            return f"error: {type(exc).__name__}"
        if data is None:
            return "unchanged"
        digest = hashlib.sha256(data).hexdigest()
        if digest == _state["last_hash"] and not force:
            return "unchanged"
        try:
            conn = (connect or _connect)(url)
            try:
                _ensure_table(conn)
                with conn.cursor() as cur:
                    cur.execute(f"INSERT INTO {TABLE} (id, size, sha256, data) VALUES (1, %s, %s, %s) "
                                "ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data, size = EXCLUDED.size, "
                                "sha256 = EXCLUDED.sha256, saved_at = now()", (len(data), digest, data))
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            _state["last_error"] = type(exc).__name__
            log.warning("Could not save to the permanent database: %s", type(exc).__name__)
            return f"error: {type(exc).__name__}"
        _state["last_hash"], _state["last_error"], _state["saved_at"] = digest, None, time.time()
        return "saved"


def _loop(db_path: str) -> None:
    while True:
        time.sleep(SAVE_EVERY)
        try:
            save(db_path)
        except Exception:
            log.exception("Permanent save failed")


def start(db_path: str) -> dict:
    """Restore once, then keep saving in the background.  Safe to call on every rerun."""
    with _lock:
        if _state["started"]:
            return status()
        _state["started"] = True
    if not database_url():
        _state["restored"] = "off"
        return status()
    _state["restored"] = restore(db_path)
    if _state["restored"] == "none saved yet":
        save(db_path, force=True)
    threading.Thread(target=_loop, args=(db_path,), daemon=True, name="cloud-store").start()
    atexit.register(lambda: save(db_path))
    return status()


def status() -> dict:
    return {"on": bool(database_url()), "restored": _state["restored"], "last_error": _state["last_error"],
            "saved_at": _state["saved_at"]}
