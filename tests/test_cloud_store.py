"""Permanent storage: the SQLite database is saved to Postgres and brought back after a restart."""
import sqlite3

import cloud_store as cs


class FakePg:
    """Just enough of a Postgres connection for cloud_store (one table, one row)."""
    rows = {}

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, args=None):
        self.last = None
        if sql.startswith("INSERT"):
            FakePg.rows[1] = args[2]
        elif sql.startswith("SELECT"):
            self.last = (FakePg.rows[1],) if 1 in FakePg.rows else None

    def fetchone(self):
        return self.last

    def commit(self):
        pass

    def close(self):
        pass


def _db(path, value):
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE IF NOT EXISTS t (v TEXT)")
    c.execute("INSERT INTO t VALUES (?)", (value,))
    c.commit()
    c.close()


def test_saved_database_comes_back_after_a_restart(tmp_path):
    FakePg.rows = {}
    cs._state["last_hash"] = None
    db = str(tmp_path / "a.db")
    _db(db, "account-1")
    assert cs.save(db, url="postgresql://x", connect=lambda u: FakePg()) == "saved"
    assert cs.save(db, url="postgresql://x", connect=lambda u: FakePg()) == "unchanged"
    fresh = str(tmp_path / "wiped.db")                     # the host's disk after a restart
    assert cs.restore(fresh, url="postgresql://x", connect=lambda u: FakePg()) == "restored"
    assert sqlite3.connect(fresh).execute("SELECT v FROM t").fetchall() == [("account-1",)]


def test_off_without_url_and_errors_are_reported(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert cs.restore(str(tmp_path / "x.db"), url="") == "off"

    def boom(url):
        raise ConnectionError
    assert cs.restore(str(tmp_path / "x.db"), url="postgresql://x", connect=boom) == "error: ConnectionError"
    FakePg.rows = {}
    assert cs.restore(str(tmp_path / "x.db"), url="postgresql://x", connect=lambda u: FakePg()) == "none saved yet"


def test_postgres_scheme_is_normalised(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://u:p@h/db")
    assert cs.database_url() == "postgresql://u:p@h/db"
