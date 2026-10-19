"""The test-case IDs from the testing brief, by report category.

Each test is tagged with the IDs it covers: @pytest.mark.brief("AUTH-001", ...).
conftest.py writes tests/reports/brief_results.json: every ID with the outcome of
each test tagged with it.  An ID with no test is reported as "not automated",
never as passed.
"""

def _ids(prefix, n):
    return [f"{prefix}-{i:03d}" for i in range(1, n + 1)]


CATEGORIES = {
    "Authentication": _ids("AUTH", 25),
    "Appliance management": _ids("APP", 25) + _ids("EDIT", 17),
    "Real-time monitoring": _ids("LIVE", 20),
    "Prediction": _ids("ML", 20),
    "Optimization": _ids("OPT", 20),
    "Notifications": _ids("ALERT", 20),
    "Dashboard": _ids("UI", 20),
    "Security": _ids("SEC", 12),
    "Integration": _ids("INT", 12),
    "End-to-end": _ids("E2E", 3),
    "Deployment smoke": _ids("SMOKE", 6),
}


def category_of(test_id: str) -> str:
    for cat, ids in CATEGORIES.items():
        if test_id in ids:
            return cat
    return "Other"
