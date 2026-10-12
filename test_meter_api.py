"""Meter API data source: payload formats, network guard, and failures."""
import json

import pandas as pd

from data_source import (check_api_url, fetch_meter_api, normalize_uploaded,
                         parse_api_payload, source_from_api)


def _readings(n=48):
    when = pd.date_range("2026-01-01", periods=n, freq="h")
    return [{"timestamp": t.isoformat(), "kw": 0.5 + (i % 5) / 10} for i, t in enumerate(when)]


class FakeResponse:
    def __init__(self, status=200, body=b"", ctype="application/json"):
        self.status_code, self._body, self.headers = status, body, {"Content-Type": ctype}

    def iter_content(self, size):
        for i in range(0, len(self._body), size):
            yield self._body[i:i + size]

    def close(self):
        pass


class FakeSession:
    def __init__(self, response):
        self.response, self.calls = response, []

    def get(self, url, **kw):
        self.calls.append((url, kw))
        return self.response


def test_json_shapes_are_understood():
    rows = _readings(5)
    for payload in (rows, {"readings": rows}, {"data": {"items": rows}},
                    {"columns": ["timestamp", "kw"], "data": [[r["timestamp"], r["kw"]] for r in rows]}):
        frame, err = parse_api_payload(json.dumps(payload).encode())
        assert err is None and len(frame) == 5 and "timestamp" in frame.columns


def test_csv_reply_and_bad_replies():
    frame, err = parse_api_payload(b"timestamp,kw\n2026-01-01 00:00,1.2\n2026-01-01 01:00,1.0\n", "text/csv")
    assert err is None and len(frame) == 2
    assert parse_api_payload(b"")[1]
    assert parse_api_payload(b"{not json")[1]
    assert parse_api_payload(json.dumps({"status": "ok"}).encode())[1]


def test_private_and_unsafe_addresses_are_refused():
    for url in ("http://localhost/readings", "https://127.0.0.1/r", "https://169.254.169.254/latest",
                "https://10.0.0.5/r", "ftp://example.com/r", "not a url",
                "https://user:pw@127.0.0.1/r"):
        host, reason = check_api_url(url, allow_private=False)
        assert host is None and reason, url
    # http is refused for public use too
    assert check_api_url("http://example.com/r", allow_private=False)[1]


def test_local_gateway_allowed_when_enabled():
    host, reason = check_api_url("http://127.0.0.1:8080/readings", allow_private=True)
    assert reason is None and host == "127.0.0.1"


def test_fetch_end_to_end_with_token():
    body = json.dumps({"readings": _readings(48)}).encode()
    session = FakeSession(FakeResponse(200, body))
    frame, err, host = fetch_meter_api("http://127.0.0.1/readings", token="abc",
                                       allow_private=True, session=session)
    assert err is None and host == "127.0.0.1"
    _, kw = session.calls[0]
    assert kw["headers"]["Authorization"] == "Bearer abc" and kw["allow_redirects"] is False
    df, report = normalize_uploaded(frame, power_unit="kW")
    assert report["ok"] and len(df) == 48
    src = source_from_api(df, host, report)
    assert src.kind == "api" and not src.is_sample


def test_fetch_failures_are_explained():
    url = "http://127.0.0.1/readings"
    for status, word in ((401, "token"), (403, "token"), (302, "redirect"), (500, "500")):
        frame, err, _ = fetch_meter_api(url, allow_private=True, session=FakeSession(FakeResponse(status)))
        assert frame is None and word in err
    big = FakeResponse(200, b"x" * 2048)
    frame, err, _ = fetch_meter_api(url, allow_private=True, session=FakeSession(big), max_bytes=1024)
    assert frame is None and "larger" in err
