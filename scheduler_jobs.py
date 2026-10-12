"""
Background scheduler for EnergyPulse
Runs daily briefs, bill alerts, weekly summaries
"""
import os
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
import atexit

_scheduler = None

def get_scheduler():
    global _scheduler
    if _scheduler is None:
        _scheduler = BackgroundScheduler(daemon=True)
        _scheduler.start()
        atexit.register(lambda: _scheduler.shutdown(wait=False))
    return _scheduler

def schedule_daily_brief(hour=7, minute=0):
    """Schedule daily brief generation for all households"""
    sched = get_scheduler()
    
    def run_daily_briefs():
        from db import get_db
        from daily_brief import build_brief, send_daily_brief
        from notifications import get_notification_service
        
        db = get_db()
        # Get all households with notification enabled
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT u.household_id, u.email, u.name, u.preferred_language
                FROM users u
                JOIN household_settings hs ON u.household_id = hs.household_id
                WHERE u.is_guest = 0
            """)
            households = cursor.fetchall()
        
        for h in households:
            try:
                send_daily_brief(h["household_id"], h["email"], h["name"], h["preferred_language"])
            except Exception as e:
                print(f"Daily brief failed for {h['household_id']}: {e}")
    
    sched.add_job(
        run_daily_briefs,
        CronTrigger(hour=hour, minute=minute),
        id="daily_brief",
        replace_existing=True
    )

def schedule_bill_alerts(hour=9, minute=0):
    """Schedule bill threshold alerts"""
    sched = get_scheduler()
    
    def check_bill_alerts():
        from db import get_db
        from notifications import get_notification_service, dispatch_household_alerts
        from model import load_data
        from data import remap_to_current_dates, shift_forecast_to_current_dates
        from cost import next_month_cost
        import pandas as pd
        import os
        
        db = get_db()
        notif = get_notification_service()
        
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT u.household_id, u.email, u.name, u.preferred_language,
                       hs.bill_threshold, hs.tariff_rate
                FROM users u
                JOIN household_settings hs ON u.household_id = hs.household_id
                WHERE u.is_guest = 0
            """)
            households = cursor.fetchall()
        
        # Load forecast data once
        forecast_path = os.path.join("data", "next_month_forecast.csv")
        if not os.path.exists(forecast_path):
            print("Forecast file not found, skipping bill alerts")
            return
        forecast_df = pd.read_csv(forecast_path, parse_dates=["datetime"])
        raw = load_data()
        if raw is not None and not raw.empty and "datetime" in raw.columns:
            current = remap_to_current_dates(raw, last_n_days=90)
            if current is not None and not current.empty:
                forecast_df = shift_forecast_to_current_dates(forecast_df, raw, current)
        
        for h in households:
            try:
                tariff_rate = float(h.get("tariff_rate") or 8.0)
                info = next_month_cost(forecast_df, tariff_rate)
                predicted_cost = info.get("total_cost") or 0
                predicted_kwh = info.get("total_kwh") or 0
                month = info.get("month") or "N/A"
                
                # Load history for weekly/optimization alerts
                history_df = db.get_usage_history(h["household_id"])
                
                dispatch_household_alerts(
                    household_id=h["household_id"],
                    primary_email=h["email"],
                    predicted_cost=predicted_cost,
                    predicted_kwh=predicted_kwh,
                    month=month,
                    tariff_rate=tariff_rate,
                    history_df=history_df,
                    language=h["preferred_language"] or "en"
                )
            except Exception as e:
                print(f"Bill alert failed for {h['household_id']}: {e}")
    
    sched.add_job(
        check_bill_alerts,
        CronTrigger(hour=hour, minute=minute),
        id="bill_alerts",
        replace_existing=True
    )

def schedule_weekly_summary(hour=10, minute=0, day_of_week="mon"):
    """Schedule weekly summary emails"""
    sched = get_scheduler()
    
    def send_weekly_summaries():
        from db import get_db
        from notifications import get_notification_service
        from cost import weekly_cost
        import pandas as pd
        from datetime import datetime, timedelta
        
        db = get_db()
        notif = get_notification_service()
        
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT u.household_id, u.email, u.name, u.preferred_language,
                       hs.tariff_rate
                FROM users u
                JOIN household_settings hs ON u.household_id = hs.household_id
                WHERE u.is_guest = 0
            """)
            households = cursor.fetchall()
        
        for h in households:
            try:
                history_df = db.get_usage_history(h["household_id"])
                if history_df is None or history_df.empty:
                    continue
                
                tariff_rate = float(h.get("tariff_rate") or 8.0)
                latest = history_df["datetime"].dt.date.iloc[-1]
                week_start = latest - timedelta(days=6)
                week_info = weekly_cost(history_df, str(week_start), tariff_rate)
                
                notif.send_weekly_summary(
                    household_id=h["household_id"],
                    primary_email=h["email"],
                    week_kwh=week_info.get("total_kwh") or 0,
                    week_cost=week_info.get("total_cost") or 0,
                    pct_change=week_info.get("pct_change") or 0,
                    language=h["preferred_language"] or "en"
                )
            except Exception as e:
                print(f"Weekly summary failed for {h['household_id']}: {e}")
    
    sched.add_job(
        send_weekly_summaries,
        CronTrigger(day_of_week=day_of_week, hour=hour, minute=minute),
        id="weekly_summary",
        replace_existing=True
    )

def init_scheduler():
    """Initialize all scheduled jobs"""
    schedule_daily_brief()
    schedule_bill_alerts()
    schedule_weekly_summary()
    print("Scheduler initialized with daily brief, bill alerts, and weekly summary")

if __name__ == "__main__":
    init_scheduler()
    # Keep running
    import time
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Scheduler stopped")