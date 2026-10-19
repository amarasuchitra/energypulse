"""Turn tests/reports/*.json into GitHub annotations (CI step; not a test module).

Prints ::notice lines with the per-category table, every brief ID's outcome and the
measured coverage, and ::error lines for each failed test.  Exit code 1 if anything failed.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(HERE, "reports")


def esc(text: str) -> str:
    return str(text).replace("%", "%25").replace("\r", "").replace("\n", "%0A")


def emit(kind: str, title: str, body: str, chunk: int = 3500) -> None:
    parts = [body[i:i + chunk] for i in range(0, len(body), chunk)] or [""]
    for n, part in enumerate(parts, 1):
        t = title if len(parts) == 1 else f"{title} ({n}/{len(parts)})"
        print(f"::{kind} title={t}::{esc(part)}")


def main() -> int:
    path = os.path.join(REPORTS, "brief_results.json")
    if not os.path.exists(path):
        print("::error title=Brief report::tests/reports/brief_results.json was not written")
        return 1
    data = json.load(open(path, encoding="utf-8"))
    rows = ["category | ids | executed | passed | failed | blocked | not automated"]
    for cat, r in data["summary"].items():
        rows.append(f"{cat} | {r['ids']} | {r['executed']} | {r['passed']} | {r['failed']} | {r['blocked']} | "
                    f"{r['not_automated']}")
    emit("notice", "Brief summary", "\n".join(rows))
    code = {"passed": "P", "failed": "F", "error": "E", "skipped": "S", "not run": "N"}
    ids = []
    for i, tests in data["by_id"].items():
        outs = "".join(sorted({code.get(t["outcome"], "?") for t in tests}))
        ids.append(f"{i}:{outs}")
    emit("notice", "Brief IDs", " ".join(ids) + "\nNOT AUTOMATED: " + " ".join(data["untagged_ids"]))
    skipped = sorted({(t["test"], t["message"][:160]) for tests in data["by_id"].values() for t in tests
                      if t["outcome"] == "skipped"})
    if skipped:
        emit("notice", "Skipped", "\n".join(f"{t} :: {m}" for t, m in skipped))
    cov = os.path.join(REPORTS, "coverage.json")
    if os.path.exists(cov):
        c = json.load(open(cov, encoding="utf-8"))
        files = sorted(((f, v["summary"]["percent_covered"]) for f, v in c["files"].items()), key=lambda x: x[0])
        emit("notice", "Coverage", f"TOTAL line coverage {c['totals']['percent_covered']:.1f}% "
             f"({c['totals']['covered_lines']}/{c['totals']['num_statements']} statements)\n"
             + "\n".join(f"{f}: {p:.0f}%" for f, p in files))
    timings = sorted(((t["seconds"], t["test"]) for tests in data["by_id"].values() for t in tests), reverse=True)
    emit("notice", "Slowest", "\n".join(f"{s:.1f}s {n}" for s, n in list(dict.fromkeys(timings))[:15]))
    failed = {}
    for tests in data["by_id"].values():
        for t in tests:
            if t["outcome"] in ("failed", "error"):
                failed[t["test"]] = t["message"]
    for name, msg in list(failed.items())[:40]:
        emit("error", f"FAILED {name.split('::')[-1][:80]}", f"{name}\n{msg[-1500:]}")
    lastfail = os.path.join(REPORTS, "junit.xml")
    if os.path.exists(lastfail):
        import xml.etree.ElementTree as ET
        root = ET.parse(lastfail).getroot()
        suite = root if root.tag == "testsuite" else root.find("testsuite")
        emit("notice", "All tests (junit)", f"tests {suite.get('tests')} failures {suite.get('failures')} "
             f"errors {suite.get('errors')} skipped {suite.get('skipped')} time {suite.get('time')}s")
        others = []
        for case in suite.iter("testcase"):
            bad = case.find("failure") if case.find("failure") is not None else case.find("error")
            if bad is not None and f"{case.get('classname')}::{case.get('name')}" not in failed:
                others.append(f"{case.get('classname')}::{case.get('name')}\n{(bad.get('message') or '')[:600]}")
        for o in others[:20]:
            emit("error", "FAILED (untagged test)", o)
        if others:
            return 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
