"""
Background scheduler for EnergyPulse.

Three jobs run inside the app's own process (APScheduler, daemon thread):

  daily_brief     07:00  today's forecast and savings plan for every household
  bill_alerts     09:00  warn when next month's predicted bill passes the threshold
  weekly_summary  Mon 10:00  last week's units and cost against the week before
  guest_cleanup   hourly     remove guest homes older than a day

Each job works from what is saved in the database (home setup, family members,
replayed meter readings), so it does not need anyone to have the app open.
A failure for one household is logged and never stops the others.

Note: on a free host that sleeps when idle, jobs only run while the app is awake.
"""
import atexit
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

DEFAULT_RATE = 8.0

_scheduler = None


def get_scheduler():
    global _scheduler
    if _scheduler is None:
        _scheduler = BackgroundScheduler(daemon=True)
        _scheduler.start()
        atexit.register(lambda: _scheduler.shutdown(wait=False))
    return _scheduler


# ----------------------------------------------------------------- helpers
def registered_households(db) -> List[Dict]:
    """Every signed-up (non-guest) household with its primary member."""
    with db.get_connection() as conn:
        rows = conn.execute("""
            SELECT household_id, email, name, preferred_language
            FROM users WHERE is_guest = 0
        """).fetchall()
    return [dict(r) for r in rows]


def household_rate(db, household_id: str) -> float:
    """Tariff in Rs./kWh from the saved home setup, or the default."""
    home = db.get_home_details(household_id) or {}
    try:
        rate = float(home.get("tariff_rate") or DEFAULT_RATE)
    except (TypeError, ValueError):
        rate = DEFAULT_RATE
    return rate if rate > 0 else DEFAULT_RATE


def household_people(db, household_id: str, name: str, email: str) -> List[Dict]:
    people = [{"name": name, "email": email}]
    for m in db.get_family_members(household_id):
        people.append({"name": m.get("name") or m.get("email"),
                       "email": m.get("email"), "phone": m.get("phone")})
    return people


def meter_history(db, household_id: str):
    """Replayed meter readings for a household, one row per timestamp."""
    df = db.get_simulation_history(household_id)
    if df is None or df.empty or "datetime" not in df.columns:
        return None
    df = (df.drop_duplicates(subset=["datetime"], keep="last")
            .sort_values("datetime").reset_index(drop=True))
    return df


# ----------------------------------------------------------------- jobs
def run_daily_briefs(db=None, service=None, date: Optional[str] = None) -> Dict[str, int]:
    """Build and send today's brief for every household. Returns counts."""
    from daily_brief import make_brief, send_daily_brief
    from meter_source import SCENARIOS, owned_appliances, today_str, with_habits

    if db is None:
        from db import get_db
        db = get_db()
    if service is None:
        from notifications import get_notification_service
        service = get_notification_service()
    date = date or today_str()
    counts = {"sent": 0, "skipped": 0, "failed": 0}
    for h in registered_households(db):
        hid = h["household_id"]
        try:
            home = db.get_home_details(hid)
            if not home:                       # not set up yet: nothing to forecast
                counts["skipped"] += 1
                continue
            owned = owned_appliances(home)
            scenario = with_habits(list(SCENARIOS)[0], home)
            brief = make_brief(date, scenario, owned, household_rate(db, hid), True)
            if not brief:
                counts["failed"] += 1
                continue
            sent = send_daily_brief(db, hid, household_people(db, hid, h["name"], h["email"]),
                                    brief, h.get("preferred_language") or "en", service)
            counts["sent" if sent else "skipped"] += 1
        except Exception as e:
            counts["failed"] += 1
            print(f"Daily brief failed for {hid}: {e}")
    return counts


def _forecast_for_alerts():
    """Next month's forecast, moved onto the present like the dashboard does."""
    import pandas as pd
    from data import remap_to_current_dates, shift_forecast_to_current_dates
    from model import load_data

    path = os.path.join("data", "next_month_forecast.csv")
    if not os.path.exists(path):
        return None
    forecast_df = pd.read_csv(path, parse_dates=["datetime"])
    try:
        raw = load_data()
        if raw is not None and not raw.empty and "datetime" in raw.columns:
            current = remap_to_current_dates(raw, last_n_days=90)
            if current is not None and not current.empty:
                forecast_df = shift_forecast_to_current_dates(forecast_df, raw, current)
    except Exception as e:
        print(f"Forecast date shift skipped: {e}")
    return forecast_df


def check_bill_alerts(db=None, forecast_df=None) -> Dict[str, int]:
    from cost import next_month_cost
    from notifications import dispatch_household_alerts

    if db is None:
        from db import get_db
        db = get_db()
    if forecast_df is None:
        forecast_df = _forecast_for_alerts()
    if forecast_df is None or forecast_df.empty:
        print("Forecast not available, skipping bill alerts")
        return {"checked": 0, "failed": 0}
    counts = {"checked": 0, "failed": 0}
    # Only the calendar month after the newest reading counts as "next month".
    from forecast import month_slice
    newest = pd.Timestamp.now() if forecast_df.empty else pd.to_datetime(forecast_df["datetime"]).min()
    nxt = newest.normalize().replace(day=1) + pd.DateOffset(months=1)
    month_df = month_slice(forecast_df, nxt.year, nxt.month)
    if month_df.empty:
        month_df = forecast_df
    for h in registered_households(db):
        hid = h["household_id"]
        try:
            rate = household_rate(db, hid)
            info = next_month_cost(month_df, rate)
            dispatch_household_alerts(
                household_id=hid,
                primary_email=h["email"],
                predicted_cost=info.get("total_cost") or 0,
                predicted_kwh=info.get("total_kwh") or 0,
                month=info.get("month") or "N/A",
                tariff_rate=rate,
                history_df=meter_history(db, hid),
                language=h.get("preferred_language") or "en",
            )
            counts["checked"] += 1
        except Exception as e:
            counts["failed"] += 1
            print(f"Bill alert failed for {hid}: {e}")
    return counts


def send_weekly_summaries(db=None, service=None, now: Optional[datetime] = None) -> Dict[str, int]:
    from cost import weekly_cost

    if db is None:
        from db import get_db
        db = get_db()
    if service is None:
        from notifications import get_notification_service
        service = get_notification_service()
    now = now or datetime.now()
    counts = {"sent": 0, "skipped": 0, "failed": 0}
    for h in registered_households(db):
        hid = h["household_id"]
        try:
            settings = db.get_household_settings(hid)
            last = settings.get("last_weekly_summary_at")
            if last:
                try:
                    if datetime.fromisoformat(str(last)) > now - timedelta(days=6):
                        counts["skipped"] += 1        # already sent this week
                        continue
                except ValueError:
                    pass
            history = meter_history(db, hid)
            if history is None:
                counts["skipped"] += 1
                continue
            latest = history["datetime"].dt.date.iloc[-1]
            week = weekly_cost(history, str(latest - timedelta(days=6)), household_rate(db, hid))
            service.send_weekly_summary(
                household_id=hid,
                primary_email=h["email"],
                week_kwh=week.get("total_kwh") or 0,
                week_cost=week.get("total_cost") or 0,
                pct_change=week.get("pct_change") or 0,
                language=h.get("preferred_language") or "en",
            )
            db.update_household_settings(hid, {"last_weekly_summary_at": now.isoformat()})
            counts["sent"] += 1
        except Exception as e:
            counts["failed"] += 1
            print(f"Weekly summary failed for {hid}: {e}")
    return counts


def purge_guests(db=None, max_age_hours: float = 24) -> int:
    """Remove guest homes left behind by visitors who closed the tab without logging out."""
    if db is None:
        from db import get_db
        db = get_db()
    try:
        return db.purge_stale_guests(max_age_hours)
    except Exception as e:
        print(f"Guest cleanup failed: {e}")
        return 0


# ----------------------------------------------------------------- schedule
def schedule_daily_brief(hour=7, minute=0):
    get_scheduler().add_job(run_daily_briefs, CronTrigger(hour=hour, minute=minute),
                            id="daily_brief", replace_existing=True,
                            misfire_grace_time=3600, coalesce=True)


def schedule_bill_alerts(hour=9, minute=0):
    get_scheduler().add_job(check_bill_alerts, CronTrigger(hour=hour, minute=minute),
                            id="bill_alerts", replace_existing=True,
                            misfire_grace_time=3600, coalesce=True)


def schedule_weekly_summary(hour=10, minute=0, day_of_week="mon"):
    get_scheduler().add_job(send_weekly_summaries,
                            CronTrigger(day_of_week=day_of_week, hour=hour, minute=minute),
                            id="weekly_summary", replace_existing=True,
                            misfire_grace_time=3600, coalesce=True)


def schedule_guest_cleanup(minutes=60):
    get_scheduler().add_job(purge_guests, "interval", minutes=minutes,
                            id="guest_cleanup", replace_existing=True, coalesce=True)


def init_scheduler():
    """Register all jobs (safe to call more than once)."""
    schedule_daily_brief()
    schedule_bill_alerts()
    schedule_weekly_summary()
    schedule_guest_cleanup()
    print("Scheduler initialized: daily brief, bill alerts, weekly summary, guest cleanup")


if __name__ == "__main__":
    import time
    init_scheduler()
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Scheduler stopped")
