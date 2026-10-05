"""
Notification System for EnergyPulse
===================================
Sends automated notifications (email, optional SMS) to the primary user
and currently enrolled family members only.

Notification types (always grounded in computed figures):
  1. Bill alerts — predicted monthly cost from next_month_cost / forecast
  2. Weekly summary — weekly_cost on stored usage
  3. Optimization tips — detect_anomalies on stored usage

Removed family members are never recipients: the recipient list is always
re-queried from family_members at send time.
"""

import os
import re
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

from db import get_db
from i18n import t_lang, format_localized_month, localized_appliance_label

log = logging.getLogger(__name__)

# Defaults so that a user who only fills in a username and an app password
# gets working Gmail SMTP without having to know these variable names.
DEFAULT_SMTP_HOST = "smtp.gmail.com"
DEFAULT_SMTP_PORT = 587
DEFAULT_SMTP_FROM = "noreply@energypulse.local"

# Secrets may live in .streamlit/secrets.toml. Streamlit supports both a flat
# file and grouped sections, so both layouts are searched.
_SECRET_SECTIONS = (None, "notifications", "email", "sms")


def _secret_lookup(name: str) -> Optional[str]:
    """
    Read one value from st.secrets, or None when secrets.toml is absent.

    Streamlit raises (or warns) outside a script run and when no secrets file
    exists, so every access is guarded: a missing secrets.toml is a normal
    setup, not an error.
    """
    try:
        import streamlit as st
        secrets = st.secrets
    except Exception:
        return None
    for section in _SECRET_SECTIONS:
        try:
            source = secrets if section is None else secrets[section]
        except Exception:
            continue
        try:
            value = source[name]
        except Exception:
            continue
        if value is None:
            continue
        return str(value)
    return None


def _read_setting(*names: str) -> Optional[str]:
    """
    First non-empty value among `names`, from the environment first and then
    st.secrets. Environment variables win so that a stale secrets.toml cannot
    silently override a .env the user just edited.
    """
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    for name in names:
        value = _secret_lookup(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _redact(text: str, secrets: Tuple[Optional[str], ...] = ()) -> str:
    """
    Replace any configured credential found in `text` with ****.

    Server error strings are echoed to the user and written to the log table,
    so a password that a provider happens to include in its reply must never
    travel any further than this function.
    """
    cleaned = str(text)
    for secret in secrets:
        if secret and len(str(secret).strip()) >= 4:
            cleaned = cleaned.replace(str(secret), "****")
    return cleaned


def _round2(value: float) -> float:
    return round(float(value), 2)


class NotificationService:
    """Sends notifications via email and optional SMS. Never raises to callers."""

    # Values that ship in .env.example (or appear in the in-app setup guide)
    # and must NOT count as "configured".
    _PLACEHOLDERS = {
        "your_sendgrid_api_key_here",
        "your_app_password_here",
        "your_account_sid_here",
        "your_auth_token_here",
        "your_email@gmail.com",
        "your.email@gmail.com",
        "the-16-character-app-password",
        "noreply@energypulse.local",
    }
    # Belt-and-braces: reject anything that is still clearly an example value.
    _PLACEHOLDER_RE = re.compile(r"(your[_.]|changeme|placeholder|_here$)", re.IGNORECASE)

    def __init__(self):
        # Config is read at construction time (fresh on every app restart), so a
        # user can add credentials to .env and see "Configured" after restarting.
        # SMTP_USER is the documented name; SMTP_USERNAME is kept as an alias so
        # older .env files and the setup guide keep working.
        self.SENDGRID_API_KEY = _read_setting("SENDGRID_API_KEY")
        self.SMTP_HOST = _read_setting("SMTP_HOST") or DEFAULT_SMTP_HOST
        self.SMTP_PORT, self.SMTP_PORT_ERROR = self._parse_port(
            _read_setting("SMTP_PORT")
        )
        self.SMTP_USERNAME = _read_setting("SMTP_USER", "SMTP_USERNAME")
        self.SMTP_PASSWORD = _read_setting("SMTP_PASSWORD")
        self.SMTP_FROM = (
            _read_setting("SMTP_FROM")
            or self.SMTP_USERNAME
            or DEFAULT_SMTP_FROM
        )
        self.SMTP_STARTTLS = self._truthy(_read_setting("SMTP_STARTTLS"), default=True)
        self.SMTP_USE_SSL = self._truthy(_read_setting("SMTP_USE_SSL"), default=False)
        # Port 465 is implicit TLS: there is no STARTTLS step to perform.
        if self.SMTP_PORT == 465 and _read_setting("SMTP_USE_SSL") is None:
            self.SMTP_USE_SSL = True
        self.TWILIO_ACCOUNT_SID = _read_setting("TWILIO_ACCOUNT_SID")
        self.TWILIO_AUTH_TOKEN = _read_setting("TWILIO_AUTH_TOKEN")
        self.TWILIO_FROM_NUMBER = _read_setting("TWILIO_FROM_NUMBER")
        self.db = get_db()
        self.email_backend = None
        self.sms_backend = None
        self.sg_client = None
        self.twilio_client = None
        # Credentials present is not proof of delivery. These stay unknown until
        # a real send succeeds or fails, so the UI never claims a connection it
        # has not made.
        self.email_verified = False
        self.last_email_error: Optional[str] = None
        self._init_email_backend()
        self._init_sms_backend()

    @staticmethod
    def _truthy(value: Optional[str], default: bool = False) -> bool:
        if value is None:
            return default
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    @property
    def _secret_values(self) -> Tuple[Optional[str], ...]:
        """Everything that must never be echoed back to the UI or a log."""
        return (
            self.SMTP_PASSWORD,
            self.SENDGRID_API_KEY,
            self.TWILIO_AUTH_TOKEN,
        )

    def _clean_error(self, exc: BaseException) -> str:
        """Readable one-line error text with any credential masked out."""
        message = _redact(str(exc), self._secret_values).strip()
        if not message:
            return exc.__class__.__name__
        if message == exc.__class__.__name__:
            return message
        return f"{exc.__class__.__name__}: {message}"

    def _log_notification(self, **kwargs) -> bool:
        """
        Write one history row, never raising.

        A logging failure must not be able to abort the send loop or reach the
        user as a send error, so the real outcome is returned separately.
        """
        try:
            self.db.log_notification(**kwargs)
            return True
        except Exception as e:
            log.error("Could not write notification history: %s", self._clean_error(e))
            return False

    @staticmethod
    def _parse_port(raw: Optional[str]) -> Tuple[int, Optional[str]]:
        """
        Read SMTP_PORT without letting a typo take down the whole app.

        int("smtp.gmail.com") raises ValueError, and this runs during
        NotificationService construction, which every tab calls. A mistyped
        port therefore used to blank out all eight tabs instead of only
        notifications.
        """
        text = (raw or "").strip()
        if not text:
            return DEFAULT_SMTP_PORT, None
        try:
            port = int(text)
        except ValueError:
            return DEFAULT_SMTP_PORT, f"SMTP_PORT={text!r} is not a number; using {DEFAULT_SMTP_PORT}"
        if not 1 <= port <= 65535:
            return DEFAULT_SMTP_PORT, f"SMTP_PORT={port} is out of range 1-65535; using {DEFAULT_SMTP_PORT}"
        return port, None

    @staticmethod
    def _real(value: Optional[str]) -> bool:
        """True only when a value is present AND not an .env.example placeholder."""
        if not value:
            return False
        s = str(value).strip()
        if s in NotificationService._PLACEHOLDERS:
            return False
        if NotificationService._PLACEHOLDER_RE.search(s):
            return False
        return True

    def _init_email_backend(self):
        key = (self.SENDGRID_API_KEY or "").strip()
        if self._real(key):
            try:
                from sendgrid import SendGridAPIClient
                self.email_backend = "sendgrid"
                self.sg_client = SendGridAPIClient(key)
                return
            except ImportError:
                self.email_backend = None
        host = (self.SMTP_HOST or "").strip()
        user = (self.SMTP_USERNAME or "").strip()
        password = (self.SMTP_PASSWORD or "").strip()
        if host and self._real(user) and self._real(password):
            self.email_backend = "smtp"

    def email_missing(self) -> List[str]:
        """
        Names of the variables still needed before email can be sent.

        The UI shows these so a user can see which value to add instead of
        only being told "not configured". SMTP_HOST and SMTP_PORT are left out
        because they fall back to Gmail/587, and SMTP_FROM is left out because
        it defaults to SMTP_USER.
        """
        if self.email_backend:
            return []
        missing = []
        if not self._real(self.SMTP_USERNAME):
            missing.append("SMTP_USER")
        if not self._real(self.SMTP_PASSWORD):
            missing.append("SMTP_PASSWORD")
        if not (self.SMTP_HOST or "").strip():
            missing.append("SMTP_HOST")
        return missing

    def email_configured(self) -> bool:
        return bool(self.email_backend)

    def email_status(self) -> str:
        """
        Honest delivery state: not_configured, configured, verified or failed.

        "configured" only means credentials were found. Nothing here opens a
        socket, so it must never be reported as a live connection.
        """
        if not self.email_backend:
            return "not_configured"
        if self.last_email_error:
            return "failed"
        if self.email_verified:
            return "verified"
        return "configured"

    def _init_sms_backend(self):
        sid = (self.TWILIO_ACCOUNT_SID or "").strip()
        token = (self.TWILIO_AUTH_TOKEN or "").strip()
        if self._real(sid) and self._real(token):
            try:
                from twilio.rest import Client
                self.sms_backend = "twilio"
                self.twilio_client = Client(sid, token)
            except ImportError:
                self.sms_backend = None

    def sms_missing(self) -> List[str]:
        """Twilio variables still needed. SMS is optional, so this is a hint."""
        if self.sms_backend:
            return []
        missing = []
        if not self._real(self.TWILIO_ACCOUNT_SID):
            missing.append("TWILIO_ACCOUNT_SID")
        if not self._real(self.TWILIO_AUTH_TOKEN):
            missing.append("TWILIO_AUTH_TOKEN")
        if not self._real(self.TWILIO_FROM_NUMBER):
            missing.append("TWILIO_FROM_NUMBER")
        return missing

    def sms_configured(self) -> bool:
        return bool(self.sms_backend)

    def recipients_for(self, household_id: str, primary_email: str,
                       preference_field: str, default_language: str) -> List[Tuple[str, str, str, Optional[str]]]:
        """
        Live recipient list: primary user plus family members who still exist
        and opted into preference_field. Removed members cannot appear here.
        Returns (email, name, language, phone).
        """
        recipients = []
        primary_email = (primary_email or "").strip().lower()
        primary_user = self.db.get_user(primary_email)
        if primary_user:
            lang = primary_user.get("preferred_language") or default_language or "en"
            recipients.append((
                primary_email,
                primary_user.get("name") or "User",
                lang,
                None,
            ))
        for member in self.db.get_family_members(household_id):
            if not member.get(preference_field):
                continue
            email = (member.get("email") or "").strip().lower()
            if not email or email == primary_email:
                continue
            lang = member.get("preferred_language") or default_language or "en"
            recipients.append((
                email,
                member.get("name") or email,
                lang,
                member.get("phone"),
            ))
        return recipients

    def send_bill_alert(self, household_id: str, primary_email: str,
                        month: str, predicted_cost: float, threshold: Optional[float] = None,
                        language: str = "en", predicted_kwh: Optional[float] = None) -> List[str]:
        predicted_cost = _round2(predicted_cost)
        predicted_kwh = _round2(predicted_kwh) if predicted_kwh is not None else None
        threshold = _round2(threshold) if threshold is not None else None
        trigger_data = {
            "type": "bill_alert",
            "month": month,
            "predicted_cost": predicted_cost,
            "predicted_kwh": predicted_kwh,
            "threshold": threshold,
            "triggered_at": datetime.now().isoformat(),
        }
        recipients = self.recipients_for(
            household_id, primary_email, "notification_bill_alerts", language
        )
        sent_to = []
        for email, name, lang, phone in recipients:
            month_lbl = format_localized_month(month, lang)
            if threshold is not None:
                body_core = t_lang(
                    "email_bill_body", lang, month=month_lbl, cost=predicted_cost, threshold=threshold
                )
            else:
                body_core = t_lang("email_bill_body_no_th", lang, month=month_lbl, cost=predicted_cost)
            subject = t_lang("email_bill_subject", lang, month=month_lbl)
            body = self._compose(name, body_core, lang)
            ok, err = self._deliver(email, subject, body, phone)
            self._log_notification(
                household_id=household_id,
                recipient_email=email,
                recipient_name=name,
                notification_type="bill_alert",
                triggered_by="predicted_monthly_cost",
                trigger_data=trigger_data,
                subject=subject,
                body_text=body,
                language=lang,
                status="sent" if ok else "failed",
                error_message=err,
            )
            if ok:
                sent_to.append(email)
        return sent_to

    def send_weekly_summary(self, household_id: str, primary_email: str,
                            week_kwh: float, week_cost: float, pct_change: float,
                            language: str = "en") -> List[str]:
        week_kwh = _round2(week_kwh)
        week_cost = _round2(week_cost)
        pct_change = round(float(pct_change), 1)
        trigger_data = {
            "type": "weekly_summary",
            "week_kwh": week_kwh,
            "week_cost": week_cost,
            "pct_change": pct_change,
            "triggered_at": datetime.now().isoformat(),
        }
        recipients = self.recipients_for(
            household_id, primary_email, "notification_bill_alerts", language
        )
        sent_to = []
        for email, name, lang, phone in recipients:
            subject = t_lang("email_weekly_subject", lang)
            body_core = t_lang(
                "email_weekly_body", lang, kwh=week_kwh, cost=week_cost, pct=pct_change
            )
            body = self._compose(name, body_core, lang)
            ok, err = self._deliver(email, subject, body, phone)
            self._log_notification(
                household_id=household_id,
                recipient_email=email,
                recipient_name=name,
                notification_type="weekly_summary",
                triggered_by="weekly_usage_summary",
                trigger_data=trigger_data,
                subject=subject,
                body_text=body,
                language=lang,
                status="sent" if ok else "failed",
                error_message=err,
            )
            if ok:
                sent_to.append(email)
        return sent_to

    def send_optimization_tips(self, household_id: str, primary_email: str,
                               tip_data: Dict[str, Any], language: str = "en") -> List[str]:
        required = ["appliance_name", "current_usage", "avg_usage", "increase_pct"]
        if not all(field in tip_data for field in required):
            return []
        current = _round2(tip_data["current_usage"])
        avg = _round2(tip_data["avg_usage"])
        increase = round(float(tip_data["increase_pct"]), 1)
        when = tip_data.get("when") or datetime.now().isoformat()
        appliance = tip_data["appliance_name"]
        trigger_data = {
            "type": "optimization_tip",
            "appliance": appliance,
            "current_usage_kw": current,
            "avg_usage_kw": avg,
            "increase_pct": increase,
            "when": when,
            "triggered_at": datetime.now().isoformat(),
        }
        recipients = self.recipients_for(
            household_id, primary_email, "notification_optimization_tips", language
        )
        sent_to = []
        for email, name, lang, phone in recipients:
            appliance_lbl = localized_appliance_label(appliance, lang)
            subject = t_lang("email_tip_subject", lang, appliance=appliance_lbl)
            body_core = t_lang(
                "email_tip_body", lang,
                appliance=appliance_lbl, current=current, avg=avg,
                increase_pct=increase, when=when,
            )
            body = self._compose(name, body_core, lang)
            ok, err = self._deliver(email, subject, body, phone)
            self._log_notification(
                household_id=household_id,
                recipient_email=email,
                recipient_name=name,
                notification_type="optimization_tip",
                triggered_by="usage_anomaly",
                trigger_data=trigger_data,
                subject=subject,
                body_text=body,
                language=lang,
                status="sent" if ok else "failed",
                error_message=err,
            )
            if ok:
                sent_to.append(email)
        return sent_to

    def send_test_notification(self, household_id: str, recipient_email: str,
                               language: str = "en",
                               predicted_cost: Optional[float] = None,
                               predicted_kwh: Optional[float] = None,
                               month: Optional[str] = None) -> Tuple[bool, str]:
        """
        Send a test email whose body contains real computed prediction figures.
        If those figures are not provided, they are loaded from stored forecast data.

        Every attempt is written to the notification log, including the ones
        that fail before a message is even built.
        """
        subject = t_lang("email_test_subject", language)
        try:
            if predicted_cost is None or month is None:
                stats = load_prediction_stats(tariff_rate=8.0)
                predicted_cost = stats["predicted_cost"]
                predicted_kwh = stats["predicted_kwh"]
                month = stats["month"]
            predicted_cost = _round2(predicted_cost)
            predicted_kwh = _round2(predicted_kwh or 0)
            when = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            month_lbl = format_localized_month(month, language)
            subject = t_lang("email_test_subject", language)
            body_core = t_lang(
                "email_test_body", language,
                when=when, month=month_lbl, cost=predicted_cost, kwh=predicted_kwh,
            )
            body = self._compose("EnergyPulse user", body_core, language)
            ok, err = self._deliver(recipient_email, subject, body, None)
            trigger_data = {
                "type": "test",
                "predicted_cost": predicted_cost,
                "predicted_kwh": predicted_kwh,
                "month": month,
                "triggered_at": datetime.now().isoformat(),
            }
            self._log_notification(
                household_id=household_id,
                recipient_email=recipient_email,
                recipient_name="Test recipient",
                notification_type="test",
                triggered_by="manual_test",
                trigger_data=trigger_data,
                subject=subject,
                body_text=body,
                language=language,
                status="sent" if ok else "failed",
                error_message=err,
            )
            if ok:
                return (True, "sent")
            return (False, err or "email_failed")
        except Exception as e:
            # A failure here (bad month string, unreadable forecast, ...) must
            # still leave a row in the history, otherwise the user clicks
            # "Send test", sees an error and has nothing to match it against.
            err = self._clean_error(e)
            self._log_notification(
                household_id=household_id,
                recipient_email=recipient_email,
                recipient_name="Test recipient",
                notification_type="test",
                triggered_by="manual_test",
                trigger_data={
                    "type": "test",
                    "predicted_cost": predicted_cost,
                    "predicted_kwh": predicted_kwh,
                    "month": month,
                    "triggered_at": datetime.now().isoformat(),
                },
                subject=subject,
                body_text="",
                language=language,
                status="failed",
                error_message=err,
            )
            return (False, err)

    def _compose(self, name: str, body_core: str, language: str) -> str:
        hello = t_lang("email_hello", language, name=name)
        footer = t_lang("email_footer", language)
        return f"{hello}\n\n{body_core}\n\n{footer}"

    def _deliver(self, to_email: str, subject: str, body: str,
                 phone: Optional[str]) -> Tuple[bool, Optional[str]]:
        try:
            email_ok, send_error = self._send_email(to_email, subject, body)
            if phone and self.sms_backend:
                try:
                    self.send_sms(phone, subject[:150])
                except Exception:
                    pass
            if email_ok:
                self.email_verified = True
                self.last_email_error = None
                return (True, None)
            if not self.email_backend:
                missing = ", ".join(self.email_missing()) or "SENDGRID_API_KEY"
                error = (
                    "No email backend configured. Set these in .env (or "
                    f".streamlit/secrets.toml) and restart the app: {missing}"
                )
                self.last_email_error = error
                return (False, error)
            # Surface the provider's own message, otherwise a rejected app
            # password and an unreachable server look identical to the user.
            self.last_email_error = send_error or "Email send failed"
            return (False, self.last_email_error)
        except Exception as e:
            self.last_email_error = self._clean_error(e)
            return (False, self.last_email_error)

    def _send_email(self, to_email: str, subject: str,
                    body_text: str) -> Tuple[bool, Optional[str]]:
        try:
            if self.email_backend == "sendgrid":
                return self._send_via_sendgrid(to_email, subject, body_text)
            if self.email_backend == "smtp":
                return self._send_via_smtp(to_email, subject, body_text)
            return (False, "No email backend selected")
        except Exception as e:
            error = self._clean_error(e)
            log.error("Email send to %s failed: %s", to_email, error)
            return (False, error)

    def _send_via_sendgrid(self, to_email: str, subject: str,
                           body_text: str) -> Tuple[bool, Optional[str]]:
        try:
            from sendgrid.helpers.mail import Mail, Email, To, Content
            message = Mail(
                from_email=Email(self.SMTP_FROM),
                to_emails=To(to_email),
                subject=subject,
                plain_text_content=Content("text/plain", body_text),
            )
            response = self.sg_client.send(message)
            if 200 <= response.status_code < 300:
                return (True, None)
            body = ""
            try:
                body = (response.body or "")[:400]
            except Exception:
                body = ""
            return (False, f"HTTP {response.status_code} from SendGrid: {body}".strip())
        except Exception as e:
            error = self._clean_error(e)
            log.error("SendGrid send to %s failed: %s", to_email, error)
            return (False, error)

    def _send_via_smtp(self, to_email: str, subject: str,
                       body_text: str) -> Tuple[bool, Optional[str]]:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.SMTP_FROM
        msg["To"] = to_email
        msg.attach(MIMEText(body_text, "plain"))
        try:
            if self.SMTP_USE_SSL:
                server = smtplib.SMTP_SSL(self.SMTP_HOST, self.SMTP_PORT, timeout=20)
            else:
                server = smtplib.SMTP(self.SMTP_HOST, self.SMTP_PORT, timeout=20)
            with server:
                # Port 465 negotiates TLS at connect time, so there is no
                # STARTTLS step to perform in that case.
                if not self.SMTP_USE_SSL and self.SMTP_STARTTLS:
                    server.starttls()
                if self._real(self.SMTP_USERNAME) or self.SMTP_PASSWORD:
                    server.login(self.SMTP_USERNAME, self.SMTP_PASSWORD)
                server.sendmail(self.SMTP_FROM, [to_email], msg.as_string())
            return (True, None)
        except Exception as e:
            error = self._clean_error(e)
            # to_email only — never the password or the API key.
            log.error("SMTP send to %s via %s:%s failed: %s",
                      to_email, self.SMTP_HOST, self.SMTP_PORT, error)
            return (False, error)

    def send_sms(self, to_phone: str, message: str) -> bool:
        if self.sms_backend != "twilio":
            return False
        try:
            self.twilio_client.messages.create(
                body=message,
                from_=self.TWILIO_FROM_NUMBER,
                to=to_phone,
            )
            return True
        except Exception as e:
            error = self._clean_error(e)
            log.error("SMS send to %s failed: %s", to_phone, error)
            return False


def load_prediction_stats(tariff_rate: float = 8.0) -> Dict[str, Any]:
    """Load predicted monthly cost from the stored forecast CSV (same as the dashboard).

    The forecast is stored for the month after the raw dataset ends (Dec 2010),
    so it is shifted onto the current dates exactly like the dashboard does
    before the monthly cost is computed — otherwise "next month" would read as
    "December 2010".
    """
    import os
    import pandas as pd
    from cost import next_month_cost
    from model import load_data
    from data import remap_to_current_dates, shift_forecast_to_current_dates
    path = os.path.join("data", "next_month_forecast.csv")
    if os.path.exists(path):
        forecast_df = pd.read_csv(path, parse_dates=["datetime"])
        raw = load_data()
        if raw is not None and not raw.empty and "datetime" in raw.columns:
            current = remap_to_current_dates(raw, last_n_days=90)
            if current is not None and not current.empty:
                forecast_df = shift_forecast_to_current_dates(forecast_df, raw, current)
        info = next_month_cost(forecast_df, tariff_rate)
        return {
            "predicted_cost": _round2(info.get("total_cost") or 0),
            "predicted_kwh": _round2(info.get("total_kwh") or 0),
            "month": info.get("month") or "N/A",
        }
    return {"predicted_cost": 0.0, "predicted_kwh": 0.0, "month": "N/A"}


def dispatch_household_alerts(household_id: str, primary_email: str,
                              predicted_cost: float, predicted_kwh: float,
                              month: str, tariff_rate: float,
                              history_df, language: str = "en") -> Dict[str, Any]:
    """
    Trigger bill/weekly/optimization alerts from real dashboard numbers.
    Failures never raise. Returns a status dict for a non-blocking UI message.
    """
    status = {"sent": [], "failed": False, "errors": []}
    try:
        db = get_db()
        service = get_notification_service()
        settings = db.get_household_settings(household_id)
        threshold = float(settings.get("bill_threshold") or 1500.0)
        now = datetime.now()

        if predicted_cost >= threshold:
            last = settings.get("last_bill_alert_at")
            due = True
            if last:
                try:
                    due = datetime.fromisoformat(str(last)) < now - timedelta(hours=20)
                except Exception:
                    due = True
            if due:
                sent = service.send_bill_alert(
                    household_id, primary_email, month,
                    predicted_cost, threshold, language, predicted_kwh,
                )
                status["sent"].append(("bill_alert", sent))
                db.update_household_settings(
                    household_id, {"last_bill_alert_at": now.isoformat()}
                )

        last_week = settings.get("last_weekly_summary_at")
        weekly_due = True
        if last_week:
            try:
                weekly_due = datetime.fromisoformat(str(last_week)) < now - timedelta(days=7)
            except Exception:
                weekly_due = True
        if weekly_due and history_df is not None and not history_df.empty:
            from cost import weekly_cost
            latest = history_df["datetime"].dt.date.iloc[-1]
            week_start = latest - timedelta(days=6)
            week_info = weekly_cost(history_df, str(week_start), tariff_rate)
            sent = service.send_weekly_summary(
                household_id, primary_email,
                week_kwh=week_info.get("total_kwh") or 0,
                week_cost=week_info.get("total_cost") or 0,
                pct_change=week_info.get("pct_change") or 0,
                language=language,
            )
            status["sent"].append(("weekly_summary", sent))
            db.update_household_settings(
                household_id, {"last_weekly_summary_at": now.isoformat()}
            )

        if history_df is not None and not history_df.empty and len(history_df) > 100:
            from optimize import detect_anomalies
            anomalies = detect_anomalies(history_df)
            if anomalies is not None and not anomalies.empty:
                row = anomalies.iloc[-1]
                key = f"{row['datetime']}|{row['top_appliance']}"
                if key != settings.get("last_optimization_key"):
                    usage = float(row["Global_active_power"])
                    mean = float(row["rolling_mean"])
                    pct = ((usage - mean) / mean * 100.0) if mean else 0.0
                    sent = service.send_optimization_tips(
                        household_id, primary_email,
                        {
                            "appliance_name": row["top_appliance"],
                            "current_usage": usage,
                            "avg_usage": mean,
                            "increase_pct": pct,
                            "when": str(row["datetime"]),
                        },
                        language=language,
                    )
                    status["sent"].append(("optimization_tip", sent))
                    db.update_household_settings(
                        household_id, {"last_optimization_key": key}
                    )
    except Exception as e:
        status["failed"] = True
        status["errors"].append(str(e))
    return status


_notification_service = None


def get_notification_service() -> NotificationService:
    global _notification_service
    if _notification_service is None:
        _notification_service = NotificationService()
    return _notification_service
