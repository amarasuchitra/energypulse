"""
Regression tests for notification backend detection, status honesty and log types.
"""
import os
import unittest
from unittest import mock

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import notifications
from notifications import NotificationService


def build(env):
    """Construct a service with a controlled environment, no sockets opened."""
    saved = {k: os.environ.get(k) for k in env}
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        return NotificationService()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestSmtpPortParsing(unittest.TestCase):
    def test_bad_port_does_not_raise(self):
        """int() on a bad port used to crash every tab, not just notifications."""
        svc = build({
            "SMTP_HOST": "smtp.gmail.com",
            "SMTP_USERNAME": "user@gmail.com",
            "SMTP_PASSWORD": "real-password",
            "SMTP_PORT": "smtp.gmail.com",
        })
        self.assertEqual(svc.SMTP_PORT, 587)
        self.assertIsNotNone(svc.SMTP_PORT_ERROR)
        self.assertIn("587", svc.SMTP_PORT_ERROR)

    def test_out_of_range_port_falls_back(self):
        svc = build({
            "SMTP_HOST": "smtp.gmail.com",
            "SMTP_USERNAME": "user@gmail.com",
            "SMTP_PASSWORD": "real-password",
            "SMTP_PORT": "99999",
        })
        self.assertEqual(svc.SMTP_PORT, 587)
        self.assertIsNotNone(svc.SMTP_PORT_ERROR)

    def test_valid_port_is_kept(self):
        svc = build({
            "SMTP_HOST": "smtp.gmail.com",
            "SMTP_USERNAME": "user@gmail.com",
            "SMTP_PASSWORD": "real-password",
            "SMTP_PORT": "465",
        })
        self.assertEqual(svc.SMTP_PORT, 465)
        self.assertIsNone(svc.SMTP_PORT_ERROR)

    def test_blank_port_uses_default_without_warning(self):
        svc = build({"SMTP_PORT": ""})
        self.assertEqual(svc.SMTP_PORT, 587)
        self.assertIsNone(svc.SMTP_PORT_ERROR)


class TestStatusHonesty(unittest.TestCase):
    def test_no_credentials_is_not_configured(self):
        svc = build({
            "SENDGRID_API_KEY": None, "SMTP_HOST": None,
            "SMTP_USERNAME": None, "SMTP_PASSWORD": None,
        })
        self.assertEqual(svc.email_status(), "not_configured")

    def test_credentials_alone_are_not_reported_as_connected(self):
        """Reporting "Connected" from env vars alone was a claim we never tested."""
        svc = build({
            "SENDGRID_API_KEY": None,
            "SMTP_HOST": "smtp.gmail.com",
            "SMTP_USERNAME": "user@gmail.com",
            "SMTP_PASSWORD": "real-password",
        })
        self.assertEqual(svc.email_status(), "configured")
        self.assertFalse(svc.email_verified)

    def test_successful_send_marks_verified(self):
        svc = build({
            "SENDGRID_API_KEY": None,
            "SMTP_HOST": "smtp.gmail.com",
            "SMTP_USERNAME": "user@gmail.com",
            "SMTP_PASSWORD": "real-password",
        })
        svc._send_email = lambda *a, **k: (True, None)
        ok, err = svc._deliver("to@example.com", "s", "b", None)
        self.assertTrue(ok)
        self.assertEqual(svc.email_status(), "verified")

    def test_failed_send_is_reported(self):
        svc = build({
            "SENDGRID_API_KEY": None,
            "SMTP_HOST": "smtp.gmail.com",
            "SMTP_USERNAME": "user@gmail.com",
            "SMTP_PASSWORD": "real-password",
        })
        svc._send_email = lambda *a, **k: (False, "SMTPAuthenticationError: bad credentials")
        ok, err = svc._deliver("to@example.com", "s", "b", None)
        self.assertFalse(ok)
        self.assertEqual(svc.email_status(), "failed")
        self.assertIsNotNone(svc.last_email_error)
        # The provider's own text must reach the user, not a generic message.
        self.assertIn("SMTPAuthenticationError", svc.last_email_error)

    def test_placeholder_credentials_are_not_configured(self):
        svc = build({
            "SENDGRID_API_KEY": None,
            "SMTP_HOST": "smtp.gmail.com",
            "SMTP_USERNAME": "your_email@gmail.com",
            "SMTP_PASSWORD": "your_app_password",
        })
        self.assertEqual(svc.email_status(), "not_configured")


class TestSmtpUserVariableName(unittest.TestCase):
    """SMTP_USER is the documented name; SMTP_USERNAME stays as an alias."""

    def test_smtp_user_is_read(self):
        svc = build({
            "SENDGRID_API_KEY": None, "SMTP_USER": "user@gmail.com",
            "SMTP_USERNAME": None, "SMTP_PASSWORD": "real-password",
        })
        self.assertEqual(svc.email_backend, "smtp")
        self.assertEqual(svc.SMTP_USERNAME, "user@gmail.com")

    def test_smtp_username_alias_still_works(self):
        svc = build({
            "SENDGRID_API_KEY": None, "SMTP_USER": None,
            "SMTP_USERNAME": "user@gmail.com", "SMTP_PASSWORD": "real-password",
        })
        self.assertEqual(svc.email_backend, "smtp")

    def test_smtp_user_wins_over_alias(self):
        svc = build({
            "SENDGRID_API_KEY": None, "SMTP_USER": "new@gmail.com",
            "SMTP_USERNAME": "old@gmail.com", "SMTP_PASSWORD": "real-password",
        })
        self.assertEqual(svc.SMTP_USERNAME, "new@gmail.com")

    def test_gmail_defaults_apply(self):
        """Host, port and STARTTLS must be usable without being spelled out."""
        svc = build({
            "SENDGRID_API_KEY": None, "SMTP_HOST": None, "SMTP_PORT": None,
            "SMTP_USER": "user@gmail.com", "SMTP_PASSWORD": "real-password",
        })
        self.assertEqual(svc.SMTP_HOST, "smtp.gmail.com")
        self.assertEqual(svc.SMTP_PORT, 587)
        self.assertTrue(svc.SMTP_STARTTLS)
        self.assertFalse(svc.SMTP_USE_SSL)
        self.assertEqual(svc.SMTP_FROM, "user@gmail.com")


class TestMissingVariables(unittest.TestCase):
    """The UI lists what is missing instead of only saying 'not configured'."""

    def test_missing_lists_the_user_and_password(self):
        svc = build({
            "SENDGRID_API_KEY": None, "SMTP_USER": None,
            "SMTP_USERNAME": None, "SMTP_PASSWORD": None,
        })
        self.assertEqual(svc.email_status(), "not_configured")
        missing = svc.email_missing()
        self.assertIn("SMTP_USER", missing)
        self.assertIn("SMTP_PASSWORD", missing)
        # Defaults mean these are never "missing".
        self.assertNotIn("SMTP_HOST", missing)

    def test_missing_is_empty_once_configured(self):
        svc = build({
            "SENDGRID_API_KEY": None, "SMTP_USER": "user@gmail.com",
            "SMTP_PASSWORD": "real-password",
        })
        self.assertTrue(svc.email_configured())
        self.assertEqual(svc.email_missing(), [])

    def test_sms_is_optional_and_lists_its_variables(self):
        svc = build({
            "TWILIO_ACCOUNT_SID": None, "TWILIO_AUTH_TOKEN": None,
            "TWILIO_FROM_NUMBER": None,
        })
        self.assertFalse(svc.sms_configured())
        self.assertEqual(svc.sms_missing(), [
            "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER",
        ])


class TestCredentialsAreNeverLeaked(unittest.TestCase):
    def test_error_text_is_redacted(self):
        svc = build({
            "SENDGRID_API_KEY": None, "SMTP_USER": "user@gmail.com",
            "SMTP_PASSWORD": "sup3rs3cret",
        })
        error = svc._clean_error(RuntimeError("login failed for sup3rs3cret"))
        self.assertNotIn("sup3rs3cret", error)
        self.assertIn("****", error)

    def test_no_email_backend_error_names_missing_vars(self):
        svc = build({
            "SENDGRID_API_KEY": None, "SMTP_USER": None,
            "SMTP_USERNAME": None, "SMTP_PASSWORD": None,
        })
        ok, err = svc._deliver("to@example.com", "s", "b", None)
        self.assertFalse(ok)
        self.assertIn("SMTP_USER", err)
        self.assertIn(".env", err)


class TestTestNotificationAlwaysLogs(unittest.TestCase):
    """Every attempt reaches the history table, even a crashing one."""

    def test_crash_is_logged_as_failed(self):
        svc = build({
            "SENDGRID_API_KEY": None, "SMTP_USER": "user@gmail.com",
            "SMTP_PASSWORD": "real-password",
        })
        logged = []
        svc.db.log_notification = lambda **kw: logged.append(kw)
        svc._deliver = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))

        ok, msg = svc.send_test_notification("hh-1", "to@example.com")

        self.assertFalse(ok)
        self.assertIn("boom", msg)
        self.assertEqual(len(logged), 1)
        self.assertEqual(logged[0]["status"], "failed")
        self.assertEqual(logged[0]["notification_type"], "test")
        self.assertIn("boom", logged[0]["error_message"])

    def test_success_is_logged_as_sent(self):
        svc = build({
            "SENDGRID_API_KEY": None, "SMTP_USER": "user@gmail.com",
            "SMTP_PASSWORD": "real-password",
        })
        logged = []
        svc.db.log_notification = lambda **kw: logged.append(kw)
        svc._send_email = lambda *a, **k: (True, None)

        ok, _msg = svc.send_test_notification(
            "hh-1", "to@example.com", predicted_cost=120.0,
            predicted_kwh=15.0, month="January 2026",
        )

        self.assertTrue(ok)
        self.assertEqual(logged[0]["status"], "sent")
        self.assertEqual(logged[0]["recipient_email"], "to@example.com")


class _GroupedSecrets(dict):
    """Mimics a secrets.toml with a [notifications] section."""

    def __getitem__(self, key):
        if key == "notifications":
            return {"SMTP_USER": "grp@gmail.com", "SMTP_PASSWORD": "grp-app-password"}
        raise KeyError(key)


class _FakeStreamlit:
    secrets = _GroupedSecrets()


class TestSecretsFallback(unittest.TestCase):
    """Credentials may come from st.secrets instead of the environment."""

    def test_missing_secrets_file_is_not_an_error(self):
        self.assertIsNone(notifications._read_setting("SMTP_USER"))

    def test_grouped_secrets_are_read(self):
        with mock.patch.dict("sys.modules", {"streamlit": _FakeStreamlit}):
            svc = build({
                "SENDGRID_API_KEY": None, "SMTP_USER": None,
                "SMTP_USERNAME": None, "SMTP_PASSWORD": None,
            })
        self.assertEqual(svc.SMTP_USERNAME, "grp@gmail.com")
        self.assertEqual(svc.email_backend, "smtp")

    def test_environment_wins_over_secrets(self):
        """A stale secrets.toml must not shadow a freshly edited .env."""
        env = {"SMTP_USER": "env@gmail.com", "SMTP_PASSWORD": "env-app-password"}
        with mock.patch.object(notifications, "_secret_lookup",
                               side_effect=lambda n: env.get(n)):
            svc = build({
                "SENDGRID_API_KEY": None, "SMTP_USER": "env@gmail.com",
                "SMTP_USERNAME": None, "SMTP_PASSWORD": "env-app-password",
            })
        self.assertEqual(svc.SMTP_USERNAME, "env@gmail.com")


class TestLogTypes(unittest.TestCase):
    """Every stored row's type must match the type in its own payload."""

    def test_weekly_summary_is_not_logged_as_bill_alert(self):
        import inspect
        src = inspect.getsource(NotificationService.send_weekly_summary)
        self.assertIn('notification_type="weekly_summary"', src)
        self.assertNotIn('notification_type="bill_alert"', src)

    def test_payload_and_row_types_agree(self):
        import inspect
        for method, expected in [
            ("send_bill_alert", "bill_alert"),
            ("send_weekly_summary", "weekly_summary"),
            ("send_optimization_tips", "optimization_tip"),
            ("send_test_notification", "test"),
        ]:
            src = inspect.getsource(getattr(NotificationService, method))
            self.assertIn(f'notification_type="{expected}"', src, method)
            self.assertIn(f'"type": "{expected}"', src, method)

    def test_every_logged_type_has_a_localized_label(self):
        from i18n import TRANSLATIONS
        import chatbot
        mapping = chatbot.NOTIFICATION_TYPE_KEY
        for type_name in ("bill_alert", "weekly_summary", "optimization_tip", "test"):
            key = mapping[type_name]
            for lang, catalog in TRANSLATIONS.items():
                self.assertIn(key, catalog, f"{type_name} missing in {lang}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

