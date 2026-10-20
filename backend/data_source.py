"""
Data source handling — flexible CSV import and honest provenance
================================================================
The app ships with a prepared sample dataset derived from the UCI
"Individual household electric power consumption" measurements. Those
readings are from 2006-2010, so the dashboard shifts their timestamps
forward to look recent. That is a demo convenience, not live data, and this
module keeps the distinction explicit everywhere:

  * :func:`normalize_uploaded` maps an arbitrary user CSV onto the canonical
    schema used by the rest of the app and reports exactly what it could and
    could not map, instead of rejecting the file.
  * :class:`DataSource` carries the frame together with its origin and the
    capabilities that are actually supported by the data, so the UI can say
    "not available" rather than inventing numbers.

Canonical schema (hourly rows):
    datetime, Global_active_power (kW), Global_reactive_power (kvar),
    Voltage (V), Global_intensity (A), Sub_metering_1..3 (Wh per hour)
"""

import re
import unicodedata

import pandas as pd

TARGET = "Global_active_power"
METER_COLS = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]


# Everything the UI can compute, and the columns each capability needs.
CAPABILITY_REQUIREMENTS = {
    # The forecast needs total power only; meter columns improve it when present.
    "prediction": [TARGET],
    "appliances": METER_COLS,
    "optimize": METER_COLS,
    "costs": [TARGET],
    "trends": [TARGET],
}

# Accepted spellings for each canonical column. Keys are compared after
# normalization (lowercase, non-alphanumerics removed, accents folded), so
# "Global Active Power", "global_active_power" and "globalActivePower" all
# collapse to the same key.
COLUMN_ALIASES = {
    "datetime": [
        "datetime", "date", "timestamp", "datetimestamp", "time", "dt",
        "recordedat", "readingtime", "readingdatetime", "eventtime",
        "measuredat", "observationdate", "day", "dateandtime",
    ],
    TARGET: [
        "globalactivepower", "globalactive", "activepower", "totalactivepower",
        "totalpower", "power", "powerkw", "kw", "globalpower", "consumption",
        "totalconsumption", "energy", "use", "usage", "kwh", "load",
        "total", "totalkw", "totalw", "demand", "gridpower", "mainpower",
        "housepower", "overallpower", "totalusage", "reading", "value",
    ],
    "Global_reactive_power": [
        "globalreactivepower", "reactivepower", "globalreactive", "reactive",
        "reactivekvar", "var",
    ],
    "Voltage": ["voltage", "volt", "volts", "v", "gridvoltage"],
    "Global_intensity": [
        "globalintensity", "intensity", "globalcurrent", "current", "amps",
        "amperes", "globalintensitya",
    ],
    "Sub_metering_1": [
        "submetering1", "submeter1", "submetera", "sub1", "kitchen",
        "kitchenwh", "meteredkitchen", "circuit1", "appliance1",
    ],
    "Sub_metering_2": [
        "submetering2", "submeter2", "submeterb", "sub2", "laundry",
        "laundrywh", "laundryroom", "meteredlaundry", "circuit2", "appliance2",
    ],
    "Sub_metering_3": [
        "submetering3", "submeter3", "submeterc", "sub3", "waterheater",
        "waterheaterandac", "waterheaterac", "ac", "airconditioner", "hvac",
        "waterheaterwh", "meteredac", "circuit3", "appliance3",
    ],
}

# UCI exports split the timestamp across two columns.
DATE_COL_KEYS = {"date", "day", "datetime", "datetimestamp", "timestamp", "dt",
                 "recordedat", "readingdatetime", "measuredat", "observationdate",
                 "dateandtime"}
TIME_COL_KEYS = {"time", "clock", "hour", "timeofday"}

# Plausible unit conversions for the power column, applied as a divisor.
POWER_UNITS = {
    "kW": 1.0,
    "W": 1000.0,
    "Wh": 1000.0,
}

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


_NOTE_TEXT = {
    "src_note_empty": "The file is empty.",
    "src_note_no_timestamp": "No timestamp column was found. One is required to order the readings.",
    "src_note_no_power": ("No total power column was recognised, so consumption, cost "
                          "and forecast cannot be computed from this file."),
    "src_note_no_valid_rows": "No row had both a readable timestamp and a numeric power value.",
    "src_note_unit_unsure": ("The power column could be kW or Wh for this sampling interval, "
                             "so it was read as Wh per interval. Use the unit selector if wrong."),
    "src_note_resampled": "The file was sampled every {minutes} minute(s) and has been aggregated to hourly rows ({how} per hour).",
    "src_note_wh_converted": "The power columns held energy in Wh and were summed per hour and divided by 1000, giving kW.",
    "src_note_w_converted": "The power columns were read in watts and divided by 1000, giving kW.",
    "src_note_missing_cols": "No column was recognised for: {cols}.",
    "src_note_date_order": "Dates were read as month-first (03/01/2024 = 3 March). Change the date column to ISO YYYY-MM-DD if that is wrong.",
}


def normalize_key(name) -> str:
    """Lowercase, strip accents and drop everything that is not a letter/digit."""
    if name is None:
        return ""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    # A trailing unit suffix should not stop an otherwise exact match.
    for suffix in ("kwh", "kw", "kvar", "wh", "w", "v", "a", "amps", "amperes"):
        if text.endswith(suffix) and text != suffix and len(text) > len(suffix) + 2:
            text = text[: -len(suffix)]
            break
    return text


def _alias_index():
    index = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        index.setdefault(normalize_key(canonical), canonical)
        for alias in aliases:
            index.setdefault(normalize_key(alias), canonical)
    return index


ALIAS_INDEX = _alias_index()


def detect_power_unit(series: pd.Series, minutes: float = 60.0) -> tuple:
    """
    Guess the unit of a power column from its magnitude and sampling interval.

    Household demand is normally well under 50 kW, so a column whose median runs
    into the hundreds or thousands is almost certainly watts or watt-hours.

    Sub-minute/hourly energy exports are genuinely ambiguous: 0.15 in a 15
    minute row is 0.6 kW if it is kW and 0.6 kW if it is Wh, and both readings
    are plausible. Guessing silently there would fabricate a number, so the
    caller is told the guess is uncertain and must let the user override it.

    Returns
    -------
    (unit, confident) where unit is one of "kW", "W", "Wh".
    """
    values = (pd.to_numeric(series, errors="coerce").abs()
              .replace([float("inf")], pd.NA).dropna())
    if values.empty:
        return "kW", True
    median = float(values.median())

    if median > 500:
        return "W", True
    if median > 50:
        return "Wh", True
    if minutes > 60.5:
        # A value this small is only believable as watts.
        return "W", True
    if abs(minutes - 60.0) > 1.0:
        # Sub-hourly and small: could be kW or Wh per interval. Assume energy,
        # which is the common meter-export shape, but flag it.
        return "Wh", False
    return "kW", True


def _date_order_code(text):
    """
    Report how a column's dates had to be read: "iso", "day-first" or "month-first".

    A slash/dash date whose first field never exceeds 12 is genuinely
    ambiguous - "03/01/2024" is 3 January in the US and 3 March elsewhere - so
    month-first is assumed and the caller tells the user, rather than silently
    picking and hoping.
    """
    sample = text.dropna()
    if sample.empty:
        return "iso"
    head = sample.iloc[0]
    if re.match(r"^\s*\d{4}-\d{2}-\d{2}", head):
        return "iso"
    leading = (sample.str.extract(r"^\s*(\d{1,2})[/\-.]", expand=False)
               .astype("float64"))
    return "day-first" if bool((leading > 12).any()) else "month-first"


def _to_datetime(values):
    """
    Parse timestamps without guessing the day/month order.

    ``dayfirst=True`` alone silently rewrites "2024-03-05" (5 March) into
    "2024-05-03" (3 May) because it swaps the month and day fields, corrupting
    every date in the file. So ISO 8601 is tried first.

    For slash/dot/dash dates the order is inferred from the column as a whole:
    a value whose first field exceeds 12 can only be a day, so the file is
    day-first, otherwise it is month-first. That is what a spreadsheet does and
    it avoids rewriting "01/02/2024" from 2 January to 1 February.
    """
    text = values if values.dtype == object else values.astype(str).str.strip()
    parsed = pd.to_datetime(text, errors="coerce", format="ISO8601")
    unresolved = text.notna() & parsed.isna()
    if not unresolved.any():
        return parsed

    rest = text.where(unresolved)
    month_first = pd.to_datetime(rest, errors="coerce", dayfirst=False)
    day_first = pd.to_datetime(rest, errors="coerce", dayfirst=True)

    # Does any unresolved value start with a number above 12?
    leading = (rest.str.extract(r"^\s*(\d{1,2})[/\-.]", expand=False)
               .astype("float64"))
    day_first_file = bool((leading > 12).any())

    chosen = day_first if day_first_file else month_first
    # Fall back to the other order for rows the preferred one could not read.
    if chosen.isna().any() and not day_first_file:
        chosen = chosen.fillna(day_first)
    elif chosen.isna().any():
        chosen = chosen.fillna(month_first)
    return parsed.fillna(chosen)


def _parse_datetime(df, date_col, time_col):
    """
    Build a timestamp from a date column, optionally joined with a time column.

    A numeric "time" column is only treated as a clock when its values look like
    hours (0-23). A column counting 0, 1, 2 ... across days is a row index, not
    a time of day, and joining it would destroy every timestamp.
    """
    if time_col is not None:
        raw_time = df[time_col]
        numeric = pd.to_numeric(raw_time, errors="coerce")
        if numeric.notna().all() and numeric.between(0, 23).all():
            # A bare hour number. "2024-05-01" + " 0" parses as 5 January 2024
            # under day-first inference, so the clock is added explicitly.
            clock = numeric.astype(int).astype(str).str.zfill(2) + ":00:00"
            parsed = (_to_datetime(df[date_col].astype(str).str.strip())
                      + pd.to_timedelta(clock, errors="coerce"))
            if parsed.notna().mean() > 0.7:
                return parsed

        as_text = raw_time.astype(str).str.strip()
        if as_text.str.contains(":").any():
            combined = df[date_col].astype(str).str.strip() + " " + as_text
            parsed = _to_datetime(combined)
            if parsed.notna().mean() > 0.7:
                return parsed
    return _to_datetime(df[date_col].astype(str).str.strip())


def normalize_uploaded(raw: pd.DataFrame, power_unit=None, resample=True):
    """
    Map an arbitrary uploaded CSV onto the canonical hourly schema.

    Parameters
    ----------
    raw : pd.DataFrame
        The uploaded file, read without assuming anything about it.
    power_unit : str, optional
        One of "kW", "W", "Wh". Auto-detected when omitted.
    resample : bool
        Aggregate to hourly rows when the source is more granular.

    Returns
    -------
    (df, report) where ``report`` is a dict with:
        ok          : bool - usable for the basic dashboard
        columns     : {original name: canonical name} for everything matched
        missing     : canonical columns the file did not provide
        ignored     : columns present in the file but not used
        detected_unit / applied_unit : the power unit guess actually applied
        rows_in / rows_out : row counts before and after resampling
        interval    : detected sampling interval in minutes
        notes       : human readable strings describing what happened
    """
    report = {
        "ok": False,
        "columns": {},
        "missing": [],
        "ignored": [],
        "detected_unit": None,
        "applied_unit": None,
        "rows_in": int(len(raw)) if raw is not None else 0,
        "rows_out": 0,
        "interval": None,
        "notes": [],
        "note_codes": [],
    }

    def note(code, **params):
        """Record an explanatory note twice: text for logs, code for the UI."""
        report["note_codes"].append((code, params))
        report["notes"].append(_NOTE_TEXT.get(code, code).format(**params))

    if raw is None or raw.empty:
        note("src_note_empty")
        return pd.DataFrame(), report

    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # Drop fully empty columns and unnamed index columns like "Unnamed: 0".
    drop = [c for c in df.columns
            if c.lower().startswith("unnamed") or df[c].isna().all()]
    if drop:
        report["ignored"].extend(drop)
        df = df.drop(columns=drop)

    keys = {c: normalize_key(c) for c in df.columns}
    used = set()

    date_col = next((c for c in df.columns if keys[c] in DATE_COL_KEYS
                     and keys[c] not in ("day",)), None)
    if date_col is None:
        date_col = next((c for c in df.columns if "date" in keys[c] or keys[c].endswith("time")), None)
    if date_col is None:
        note("src_note_no_timestamp")
        return pd.DataFrame(), report
    time_col = next((c for c in df.columns
                     if c != date_col and keys[c] in TIME_COL_KEYS), None)
    if time_col:
        used.update({date_col, time_col})
    datetime = _parse_datetime(df, date_col, time_col)
    report["columns"][date_col] = "datetime"
    if time_col:
        report["columns"][time_col] = "datetime"
    report["date_order"] = _date_order_code(
        df[date_col].astype(str) if time_col is None
        else (df[date_col].astype(str).str.strip() + " "
              + df[time_col].astype(str).str.strip()))
    if report["date_order"] == "month-first":
        note("src_note_date_order")

    out = pd.DataFrame({"datetime": datetime})

    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical == "datetime":
            continue
        match = None
        for col in df.columns:
            if col in used:
                continue
            if keys[col] in {normalize_key(a) for a in aliases}:
                match = col
                break
        if match is not None:
            out[canonical] = pd.to_numeric(df[match], errors="coerce")
            report["columns"][match] = canonical
            used.add(match)

    report["ignored"].extend(c for c in df.columns if c not in used)

    if TARGET not in out.columns:
        note("src_note_no_power")
        return pd.DataFrame(), report

    out = out.dropna(subset=["datetime", TARGET]).sort_values("datetime")
    if out.empty:
        note("src_note_no_valid_rows")
        return pd.DataFrame(), report

    # Sampling interval, decided before the unit because it decides the unit.
    diffs = out["datetime"].diff().dropna()
    if not diffs.empty:
        minutes = float(diffs.median().total_seconds() / 60.0)
        report["interval"] = round(minutes, 3)
    else:
        minutes = 60.0

    detected, confident = detect_power_unit(out[TARGET], minutes)
    applied = power_unit or detected
    if applied not in POWER_UNITS:
        applied = "kW"
    report["detected_unit"] = detected
    report["applied_unit"] = applied
    report["unit_confident"] = bool(confident or power_unit is not None)
    divisor = POWER_UNITS[applied]

    if not report["unit_confident"]:
        note("src_note_unit_unsure")

    if resample and minutes > 0 and abs(minutes - 60.0) > 1e-6:
        # Energy columns must be summed across the interval; power columns are
        # averaged. Getting this backwards is what made 15-minute meter data
        # come out 4x too small.
        how = "sum" if applied == "Wh" else "mean"
        agg = {}
        for col in [TARGET, "Global_reactive_power", "Global_intensity", *METER_COLS]:
            if col in out.columns:
                agg[col] = how
        hourly = (out.dropna(subset=["datetime"]).set_index("datetime")
                      .resample("h").agg(agg).dropna(subset=[TARGET]))
        out = hourly.reset_index()
        note("src_note_resampled", minutes=f"{minutes:g}", how=how)

    if divisor != 1.0:
        for col in [TARGET, *METER_COLS]:
            if col in out.columns:
                out[col] = out[col] / divisor
        if applied == "Wh":
            note("src_note_wh_converted")
        else:
            note("src_note_w_converted")

    out = add_time_features(out)
    report["rows_out"] = int(len(out))
    report["missing"] = [c for c in [TARGET] + METER_COLS
                         if c not in out.columns]
    report["ok"] = len(out) > 0
    if report["missing"]:
        note("src_note_missing_cols", cols=", ".join(report["missing"]))
    return out, report


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the calendar features the models and charts rely on."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["day_of_month"] = df["datetime"].dt.day
    df["week_of_year"] = df["datetime"].dt.isocalendar().week.astype(int)
    return df.reset_index(drop=True)


def capabilities_of(df: pd.DataFrame) -> dict:
    """
    Which features the given frame can honestly support.

    A capability is enabled only when every column it needs is present with
    usable values, so the UI never fabricates a number it cannot compute.
    """
    present = set(df.columns) if df is not None else set()
    caps = {}
    for name, required in CAPABILITY_REQUIREMENTS.items():
        missing = [c for c in required if c not in present]
        caps[name] = {"enabled": not missing, "missing": missing}
    return caps


class DataSource:
    """
    A dataset plus everything the UI needs to describe it honestly.

    Attributes
    ----------
    df : pd.DataFrame
        Canonical hourly frame.
    kind : str
        "sample" for the bundled UCI-derived demo data, "upload" for a user
        CSV, "api" for readings fetched from a meter API. Anything else is rejected.
    filename : str
        Display name of the upload (empty for the sample).
    original_range : tuple or None
        For the sample data, the real measurement window before the demo
        shift, so the UI can say what is actually being replayed.
    shifted : bool
        True when the timestamps were moved forward for the demo.
    """

    def __init__(self, df, kind, filename="", label="", original_range=None,
                 shifted=False, report=None):
        if kind not in ("sample", "upload", "api", "meter"):
            raise ValueError("kind must be 'sample', 'upload', 'api' or 'meter'")
        self.df = add_time_features(df) if df is not None else pd.DataFrame()
        self.kind = kind
        self.filename = filename
        self.label = label or filename
        self.original_range = original_range
        self.shifted = shifted
        self.report = report or {}

    @property
    def is_sample(self):
        return self.kind == "sample"

    @property
    def capabilities(self):
        return capabilities_of(self.df)

    def can(self, name):
        return bool(self.capabilities.get(name, {}).get("enabled"))

    def age_hours(self):
        """How far behind wall-clock time the newest reading is."""
        if self.df.empty or "datetime" not in self.df.columns:
            return None
        latest = self.df["datetime"].max()
        if pd.isna(latest):
            return None
        return (pd.Timestamp.now() - latest).total_seconds() / 3600.0

    def __len__(self):
        return len(self.df)


def source_from_sample(df: pd.DataFrame, filename: str, original_range=None,
                       shifted=True):
    return DataSource(df, "sample", filename=filename,
                      original_range=original_range, shifted=shifted)


def source_from_meter(df: pd.DataFrame, label: str = "Simulated meter"):
    """The household's own simulated main meter, hour by hour up to now (real timestamps)."""
    return DataSource(df, "meter", filename=label, label=label, shifted=False)


def meter_hourly(minute_df: pd.DataFrame, upto) -> pd.DataFrame:
    """
    1-minute simulated meter readings -> the hourly columns every page uses, finished hours only.
      Global_active_power  average kW in the hour (= kWh in the hour)
      Sub_metering_1/2/3   Wh in the hour for kitchen / laundry and fridge / water heater and AC
      Global_intensity     A at 230 V;  Voltage 230
    """
    df = minute_df[pd.to_datetime(minute_df["datetime"]) < pd.Timestamp(upto).floor("h")].copy()
    if df.empty:
        return pd.DataFrame(columns=["datetime", TARGET])
    df["datetime"] = pd.to_datetime(df["datetime"])
    groups = {"Sub_metering_1": ("microwave", "dishwasher", "induction", "kettle", "mixer", "otg"),
              "Sub_metering_2": ("washing_machine", "fridge", "iron"),
              "Sub_metering_3": ("geyser", "ac", "water_pump", "heater")}
    out = pd.DataFrame({"datetime": df["datetime"].dt.floor("h"), TARGET: df["mains_kw"].astype(float)})
    for col, keys in groups.items():
        cols = [f"{k}_kw" for k in keys if f"{k}_kw" in df.columns]
        out[col] = df[cols].astype(float).sum(axis=1) * 1000.0 if cols else 0.0
    hourly = out.groupby("datetime").agg({TARGET: "mean", "Sub_metering_1": "mean", "Sub_metering_2": "mean",
                                          "Sub_metering_3": "mean"})
    counts = out.groupby("datetime").size()
    hourly = hourly[counts >= 45].reset_index()
    hourly["Voltage"] = 230.0
    hourly["Global_intensity"] = hourly[TARGET] * 1000.0 / 230.0
    return hourly


def source_from_upload(df: pd.DataFrame, filename: str, report=None):
    return DataSource(df, "upload", filename=filename, shifted=False,
                      report=report or {})


def source_from_api(df: pd.DataFrame, host: str, report=None):
    return DataSource(df, "api", filename=host, label=host, shifted=False,
                      report=report or {})


# ---------------------------------------------------------------- meter API
# Readings can also come from a smart-meter or utility API that returns JSON or
# CSV over HTTPS.  The server fetches the URL, so addresses on private or
# internal networks are refused unless ENERGYPULSE_ALLOW_PRIVATE_API=1 is set
# (useful when the app runs at home next to a local meter gateway).
API_TIMEOUT = 15
_LIST_KEYS = ("readings", "data", "items", "results", "records", "values", "rows", "measurements")


def _private_allowed() -> bool:
    import os
    return os.environ.get("ENERGYPULSE_ALLOW_PRIVATE_API", "").strip().lower() in ("1", "true", "yes")


def check_api_url(url: str, allow_private=None):
    """(host, None) when the URL may be fetched, otherwise (None, reason)."""
    import ipaddress
    import socket
    from urllib.parse import urlparse
    allow_private = _private_allowed() if allow_private is None else allow_private
    url = (url or "").strip()
    parts = urlparse(url)
    if parts.scheme not in ("https", "http") or not parts.hostname:
        return None, "Enter a full http(s):// address."
    if parts.username or parts.password:
        return None, "Put the access token in the token box, not in the address."
    if parts.scheme == "http" and not allow_private:
        return None, "Use an https:// address."
    host = parts.hostname
    try:
        infos = socket.getaddrinfo(host, parts.port or (443 if parts.scheme == "https" else 80))
    except (socket.gaierror, UnicodeError):
        return None, f"The address {host} could not be found."
    if not allow_private:
        for info in infos:
            ip = ipaddress.ip_address(info[4][0].split("%")[0])
            if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                    or ip.is_multicast or ip.is_unspecified):
                return None, f"{host} is on a private network, which this server does not contact."
    return host, None


def _records_from_json(obj, depth=0):
    """Find the list of readings inside a JSON reply."""
    if isinstance(obj, list):
        if obj and all(isinstance(r, dict) for r in obj):
            return pd.json_normalize(obj)
        if obj and all(isinstance(r, (list, tuple)) for r in obj):
            return pd.DataFrame(obj)
        return None
    if isinstance(obj, dict) and depth < 3:
        if isinstance(obj.get("columns"), list) and isinstance(obj.get("data"), list):
            return pd.DataFrame(obj["data"], columns=obj["columns"])
        keys = [k for k in _LIST_KEYS if k in obj] + [k for k in obj if k not in _LIST_KEYS]
        for key in keys:
            found = _records_from_json(obj[key], depth + 1)
            if found is not None and not found.empty:
                return found
    return None


def parse_api_payload(body: bytes, content_type: str = ""):
    """(frame, None) or (None, reason) for a JSON or CSV reply."""
    import io
    import json
    if not body:
        return None, "The API returned nothing."
    text_start = body.lstrip()[:1]
    if "json" in (content_type or "").lower() or text_start in (b"[", b"{"):
        try:
            frame = _records_from_json(json.loads(body.decode("utf-8-sig")))
        except (ValueError, UnicodeDecodeError):
            return None, "The reply looked like JSON but could not be read."
        if frame is None or frame.empty:
            return None, "No list of readings was found in the JSON reply."
        return frame, None
    for kwargs in ({"sep": None, "engine": "python"}, {"sep": ","}, {"sep": ";"}):
        try:
            frame = pd.read_csv(io.BytesIO(body), **kwargs)
        except Exception:
            continue
        if frame is not None and not frame.empty and frame.shape[1] > 1:
            return frame, None
    return None, "The reply was neither JSON readings nor a CSV table."


def fetch_meter_api(url: str, token: str = None, timeout: float = API_TIMEOUT,
                    max_bytes: int = MAX_UPLOAD_BYTES, allow_private=None, session=None):
    """
    Download readings from a meter API.
    Returns (frame, None, host) on success or (None, reason, host).
    """
    import requests
    host, reason = check_api_url(url, allow_private)
    if reason:
        return None, reason, host
    headers = {"Accept": "application/json, text/csv;q=0.9, */*;q=0.5",
               "User-Agent": "EnergyPulse/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token.strip()}"
    getter = session or requests
    try:
        # Redirects are not followed: a redirect could lead to an address
        # that the check above would have refused.
        resp = getter.get(url.strip(), headers=headers, timeout=timeout,
                          allow_redirects=False, stream=True)
    except requests.exceptions.Timeout:
        return None, f"No reply within {int(timeout)} seconds.", host
    except requests.exceptions.RequestException as exc:
        return None, f"Could not connect ({type(exc).__name__}).", host
    try:
        if 300 <= resp.status_code < 400:
            return None, "The API answered with a redirect; use the final address.", host
        if resp.status_code in (401, 403):
            return None, "The API refused access. Check the token.", host
        if resp.status_code >= 400:
            return None, f"The API answered with error {resp.status_code}.", host
        chunks, size = [], 0
        for chunk in resp.iter_content(64 * 1024):
            size += len(chunk)
            if size > max_bytes:
                return None, f"The reply is larger than {max_bytes // (1024 * 1024)} MB.", host
            chunks.append(chunk)
        body = b"".join(chunks)
        frame, reason = parse_api_payload(body, resp.headers.get("Content-Type", ""))
        return frame, reason, host
    finally:
        try:
            resp.close()
        except Exception:
            pass
