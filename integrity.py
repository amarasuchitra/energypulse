"""
Tamper evidence for bills and reports
=====================================
Makes later changes to a saved bill, or to a report the app produced,
detectable.

How it works
------------
1. Fingerprint.  When a bill is saved, the app stores a SHA-256 fingerprint
   of the exact file and of the figures read from it.  Change one pixel or
   one digit afterwards and the fingerprint no longer matches.
2. Chain.  Every record also stores the fingerprint of the record before
   it.  Deleting, reordering or rewriting an old record breaks every link
   after it, so one edit cannot be hidden by also editing its fingerprint.
3. Seal.  Each record and each exported report is signed with a secret key
   kept on the server (HMAC-SHA256).  Without the key a forged record or an
   edited report cannot be given a valid seal.

What it does not do
-------------------
- It cannot tell whether a bill was genuine BEFORE it was uploaded.
- Someone who holds both the database and the key file can rewrite history.
  Keep the key (ENERGYPULSE_SECRET or data/.integrity_key) off shared machines.

Usage:
    from integrity import Ledger
    ledger = Ledger()
    ledger.record(household, "bill", "oct.pdf", file_bytes, {"amount": "1840"})
    ledger.verify_chain(household)            # -> [] when untouched
    text = sign_report(body)                  # export
    verify_report(text)                       # -> (True, "...") / (False, "...")
"""

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from db import DB_PATH

KEY_FILE = os.path.join("data", ".integrity_key")
BILL_DIR = os.path.join("data", "bills")
GENESIS = "0" * 64
SEAL_MARK = "-----ENERGYPULSE SEAL-----"


def _key() -> bytes:
    env = os.environ.get("ENERGYPULSE_SECRET")
    if env:
        return env.encode()
    if not os.path.exists(KEY_FILE):
        os.makedirs(os.path.dirname(KEY_FILE), exist_ok=True)
        with open(KEY_FILE, "w") as fh:
            fh.write(secrets.token_hex(32))
        try:
            os.chmod(KEY_FILE, 0o600)
        except OSError:
            pass
    with open(KEY_FILE) as fh:
        return fh.read().strip().encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _seal(text: str) -> str:
    return hmac.new(_key(), text.encode(), hashlib.sha256).hexdigest()


def _canonical(payload: Dict[str, Any]) -> str:
    return json.dumps(payload or {}, sort_keys=True, separators=(",", ":"), default=str)


class Ledger:
    """Append-only, hash-chained record of bills and reports for each household."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS integrity_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    household_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    ref TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    prev_hash TEXT NOT NULL,
                    entry_hash TEXT NOT NULL,
                    seal TEXT NOT NULL
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
    def _entry_hash(prev_hash, household_id, kind, ref, content_hash, payload_json, created_at) -> str:
        joined = "|".join([prev_hash, household_id, kind, ref, content_hash, payload_json, created_at])
        return sha256(joined.encode())

    def entries(self, household_id: str, kind: Optional[str] = None) -> List[dict]:
        query = "SELECT * FROM integrity_ledger WHERE household_id = ?"
        args = [household_id]
        if kind:
            query += " AND kind = ?"
            args.append(kind)
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(query + " ORDER BY id ASC", args).fetchall()]

    def record(self, household_id: str, kind: str, ref: str, content: bytes,
               payload: Optional[Dict[str, Any]] = None) -> dict:
        payload_json = _canonical(payload)
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content_hash = sha256(content)
        with self._conn() as conn:
            last = conn.execute(
                "SELECT entry_hash FROM integrity_ledger WHERE household_id = ? ORDER BY id DESC LIMIT 1",
                (household_id,)).fetchone()
            prev_hash = last["entry_hash"] if last else GENESIS
            entry_hash = self._entry_hash(prev_hash, household_id, kind, ref, content_hash,
                                          payload_json, created_at)
            cur = conn.execute(
                "INSERT INTO integrity_ledger (household_id, kind, ref, content_hash, payload_json, "
                "created_at, prev_hash, entry_hash, seal) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (household_id, kind, ref, content_hash, payload_json, created_at, prev_hash,
                 entry_hash, _seal(entry_hash)))
            entry_id = cur.lastrowid
        return {"id": entry_id, "content_hash": content_hash, "entry_hash": entry_hash,
                "created_at": created_at}

    def verify_chain(self, household_id: str) -> List[str]:
        """Problems found in the household's records; an empty list means untouched."""
        problems, prev = [], GENESIS
        for e in self.entries(household_id):
            expected = self._entry_hash(e["prev_hash"], e["household_id"], e["kind"], e["ref"],
                                        e["content_hash"], e["payload_json"], e["created_at"])
            if e["prev_hash"] != prev:
                problems.append(f"Record {e['id']} ({e['ref']}): a record before it was removed or reordered.")
            if e["entry_hash"] != expected:
                problems.append(f"Record {e['id']} ({e['ref']}): its details were changed after saving.")
            if not hmac.compare_digest(e["seal"], _seal(e["entry_hash"])):
                problems.append(f"Record {e['id']} ({e['ref']}): the seal does not match this server's key.")
            prev = e["entry_hash"]
        return problems

    def check_content(self, entry: dict, content: bytes, payload: Optional[dict] = None) -> Tuple[bool, str]:
        """Is this file (and these figures) exactly what was saved?"""
        if sha256(content) != entry["content_hash"]:
            return False, "The file is different from the one that was saved."
        if payload is not None and _canonical(payload) != entry["payload_json"]:
            return False, "The figures shown are different from the ones that were saved."
        return True, f"Unchanged since it was saved on {entry['created_at']}."

    def find_by_content(self, household_id: str, content: bytes) -> Optional[dict]:
        digest = sha256(content)
        for e in self.entries(household_id):
            if e["content_hash"] == digest:
                return e
        return None


# ----------------------------------------------------------------- bill files
def bill_path(household_id: str, content_hash: str, ext: str) -> str:
    safe_ext = "".join(c for c in ext.lower() if c.isalnum())[:5] or "bin"
    return os.path.join(BILL_DIR, household_id, f"{content_hash}.{safe_ext}")


def store_bill_file(household_id: str, data: bytes, ext: str) -> str:
    path = bill_path(household_id, sha256(data), ext)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        from auth import protect            # bills hold personal details: encrypted on disk
        with open(path, "wb") as fh:
            fh.write(protect(data))
    return path


def read_bill_file(path: str) -> Optional[bytes]:
    """The original file, decrypted.  None if it is missing or was altered on disk."""
    from auth import unprotect
    try:
        with open(path, "rb") as fh:
            return unprotect(fh.read())
    except OSError:
        return None


# ----------------------------------------------------------------- reports
def sign_report(body: str) -> str:
    """Append a seal to a report.  Any later edit to the body invalidates it."""
    body = body.rstrip("\n") + "\n"
    return f"{body}{SEAL_MARK}\n{_seal(body)}\n"


def verify_report(text: str) -> Tuple[bool, str]:
    text = text.replace("\r\n", "\n")
    if SEAL_MARK not in text:
        return False, "This file has no EnergyPulse seal, so it cannot be checked."
    body, _, tail = text.rpartition(SEAL_MARK + "\n")
    given = tail.strip()
    if hmac.compare_digest(given, _seal(body)):
        return True, "The seal matches. This report is exactly as EnergyPulse produced it."
    return False, "The seal does not match. The report was edited after it was produced, or it came from another system."
