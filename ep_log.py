"""
Logging for EnergyPulse.

Technical details (data flow, model runs, alerts, failures) go to the server
log, never to the dashboard.  Level from ENERGYPULSE_LOG_LEVEL (default INFO).

    from ep_log import get_logger
    log = get_logger(__name__)
    log.info("...")
"""
import logging
import os

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        root = logging.getLogger("energypulse")
        if not root.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            root.addHandler(handler)
        root.setLevel(os.environ.get("ENERGYPULSE_LOG_LEVEL", "INFO").upper())
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(f"energypulse.{name}")
