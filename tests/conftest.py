"""Make backend/ and frontend/ importable for the tests, as frontend/app.py does for the app."""
import os
import sys
import tempfile

# Tests never touch the real database, start background jobs or load TensorFlow.
# These must be set before any module reads them (paths.DB_PATH is read at import).
os.environ.setdefault("ENERGYPULSE_DB", os.path.join(tempfile.mkdtemp(prefix="ep-test-"), "test.db"))
os.environ.setdefault("ENERGYPULSE_NO_SCHEDULER", "1")
os.environ.setdefault("ENERGYPULSE_DISABLE_LSTM", "1")
os.environ.setdefault("ENERGYPULSE_NO_WARMUP", "1")
# The deployed app hides error details from visitors; tests need to see them.
os.environ.setdefault("STREAMLIT_CLIENT_SHOW_ERROR_DETAILS", "full")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in (os.path.join(ROOT, "frontend"), os.path.join(ROOT, "backend")):
    if d not in sys.path:
        sys.path.insert(0, d)
os.chdir(ROOT)          # a few tests read data/ and models/ by relative path


# ------------------------------------------------------------------ brief-ID report
import json
import time

import pytest

_RESULTS = {}          # nodeid -> {"ids": [...], "outcome": str, "message": str, "seconds": float}
_METRICS = {}          # measured figures quoted in the test report


@pytest.fixture
def metric():
    """metric(name, value): record a measured figure (timing, error) for tests/reports/metrics.json."""
    def record(name, value):
        _METRICS[name] = value
    return record


def pytest_configure(config):
    config.addinivalue_line("markers", "brief(*ids): test-case IDs from the testing brief this test covers")


def pytest_collection_modifyitems(session, config, items):
    for item in items:
        ids = []
        for m in item.iter_markers("brief"):
            ids.extend(m.args)
        if ids:
            _RESULTS[item.nodeid] = {"ids": sorted(set(ids)), "outcome": "not run", "message": "", "seconds": 0.0}


def pytest_runtest_logreport(report):
    r = _RESULTS.get(report.nodeid)
    if r is None:
        return
    if report.when == "setup" and report.skipped:
        r["outcome"] = "skipped"
        r["message"] = str(report.longrepr[-1]) if isinstance(report.longrepr, tuple) else str(report.longrepr)[:300]
    elif report.when == "setup" and report.failed:
        r["outcome"], r["message"] = "error", str(report.longrepr)[-600:]
    elif report.when == "call":
        r["seconds"] = round(report.duration, 3)
        if report.passed:
            r["outcome"] = "passed"
        elif report.skipped:
            r["outcome"] = "skipped"
            r["message"] = str(report.longrepr[-1]) if isinstance(report.longrepr, tuple) else str(report.longrepr)[:300]
        else:
            r["outcome"] = "failed"
            r["message"] = (report.longreprtext or "")[-800:]
    elif report.when == "teardown" and report.failed and r["outcome"] == "passed":
        r["outcome"], r["message"] = "error", (report.longreprtext or "")[-600:]


def pytest_sessionfinish(session, exitstatus):
    if not _RESULTS:
        return
    from brief_ids import CATEGORIES, category_of
    by_id = {}
    for nodeid, r in _RESULTS.items():
        for i in r["ids"]:
            by_id.setdefault(i, []).append({"test": nodeid, "outcome": r["outcome"], "message": r["message"],
                                            "seconds": r["seconds"]})
    summary = {}
    for cat, ids in CATEGORIES.items():
        row = {"ids": len(ids), "executed": 0, "passed": 0, "failed": 0, "blocked": 0, "not_automated": 0}
        for i in ids:
            outs = [t["outcome"] for t in by_id.get(i, [])]
            if not outs:
                row["not_automated"] += 1
            elif any(o in ("failed", "error") for o in outs):
                row["executed"] += 1
                row["failed"] += 1
            elif all(o == "skipped" or o == "not run" for o in outs):
                row["blocked"] += 1
            else:
                row["executed"] += 1
                row["passed"] += 1
        summary[cat] = row
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "summary": summary,
           "by_id": {i: by_id[i] for i in sorted(by_id)},
           "untagged_ids": sorted(i for ids in CATEGORIES.values() for i in ids if i not in by_id),
           "unknown_ids": sorted(i for i in by_id if category_of(i) == "Other")}
    os.makedirs(os.path.join(ROOT, "tests", "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "tests", "reports", "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(_METRICS, fh, indent=1, default=str)
    with open(os.path.join(ROOT, "tests", "reports", "brief_results.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
