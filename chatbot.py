"""
Grounded Q&A Chatbot for EnergyPulse
=====================================
A question-answering chatbot that ONLY answers based on real household data.

Core principle:
  - Every numeric claim must come from a tool that queries real data
  - If data isn't available, the chatbot says so honestly
  - All responses are validated against actual tool output before display
  - Supports all four languages (English, Hindi, Kannada, Telugu) — questions
    are understood and answered in the selected language, never defaulting back
    to English mid-conversation.

Questions the chatbot can answer:
  - "Why was my bill higher last week?"
  - "Which appliance uses the most power?"
  - "How can I reduce my AC usage cost?"
  - "What's my average daily consumption?"
  - "What notifications were sent to my family?"
"""

import json
import re
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta

from db import get_db
from llm_guard import LLMHallucinationGuard
from i18n import t_lang, format_localized_month, NOTIFICATION_TYPE_KEY


# ──────────────────────────────────────────────────────────────────────────
# Multilingual question understanding
#   The selected UI language is always known at ask-time, so intent matching
#   is done against keywords in that language (English fallback).
# ──────────────────────────────────────────────────────────────────────────
_QUESTION_TERMS = {
    "why_compare": {
        "en": ["why", "higher", "lower", "compare", "compared", "versus", "change",
               "went up", "went down", "increased", "decreased", "difference"],
        "hi": ["क्यों", "अधिक", "तुलना", "बढ़", "घट", "बदल", "अंतर"],
        "kn": ["ಏಕೆ", "ಹೆಚ್ಚು", "ಕಡಿಮೆ", "ಹೋಲಿಕೆ", "ಏರಿತು", "ಇಳಿಯಿತು", "ವ್ಯತ್ಯಾಸ", "ಬದಲಾವಣೆ"],
        "te": ["ఎందుకు", "ఎక్కువ", "తక్కువ", "పోలిక", "పెరిగాయి", "తగ్గాయి", "మార్పు", "తేడా"],
    },
    "usage_history": {
        "en": ["history", "last", "how much", "consumption", "used", "usage", "trend",
               "average daily", "week"],
        "hi": ["इतिहास", "पिछले", "कितना", "खपत", "उपयोग", "औसत", "सप्ताह", "दिन"],
        "kn": ["ಇತಿಹಾಸ", "ಕಳೆದ", "ಎಷ್ಟು", "ಬಳಕೆ", "ಸರಾಸರಿ", "ವಾರ", "ದಿನ"],
        "te": ["చరిత్ర", "గత", "ఎంత", "వినియోగం", "సగటు", "వారం", "రోజు"],
    },
    "appliance": {
    "en": ["appliance", "most", "device", "consume"],
    "hi": ["उपकरण", "सबसे", "यंत्र"],
    "kn": ["ಉಪಕರಣ", "ಸಾಧನ", "ಹೆಚ್ಚು ಬಳಸುವ"],
    "te": ["ఉపకరణం", "పరికరం", "ఎక్కువ ఉపయోగించే"],
    },
    "prediction": {
        "en": ["predict", "forecast", "next month", "monthly", "cost", "expensive",
               "rupee", "rs ", "projection", "expected", "spend", "bill"],
        "hi": ["अनुमान", "पूर्वानुमान", "अगले", "मासिक", "लागत", "महंगा", "रुपये",
               "प्रोजेक्शन", "बिल", "खर्च", "बिजली"],
        "kn": ["ಅಂದಾಜು", "ಮುನ್ಸೂಚನೆ", "ಮುಂದಿನ", "ಮಾಸಿಕ", "ವೆಚ್ಚ", "ದುಬಾರಿ", "ರೂ",
               "ಬಿಲ್", "ವಿದ್ಯುತ್"],
        "te": ["అంచనా", "ఫోరెకాస్ట్", "తదుపరి", "మాసిక", "ఖర్చు", "ఖరీదు", "రూ",
               "బిల్", "విద్యుత్"],
    },
    "optimize": {
        "en": ["reduce", "save", "lower", "optimize", "tip", "efficient", "how can i",
               "cut", "switch off", "less"],
        "hi": ["बचाएँ", "बचाएं", "बचाऊँ", "बचा", "घटाएँ", "घटाऊँ", "घटाने", "सुधार",
               "सुझाव", "कुशल", "बंद", "कम कर"],
        "kn": ["ಉಳಿಸಿ", "ಉಳಿಸು", "ಸುಧಾರಿಸಿ", "ಸಲಹೆ", "ದಕ್ಷತೆ", "ಆಫ್",
               "ಕಡಿಮೆ ಮಾಡ", "ಕಡಿಮೆ ಮಾಡಿ"],
        "te": ["తగ్గించు", "ఆదా", "మెరుగు", "సూచన", "దక్షత", "తగ్గించండి",
               "తగ్గించాలి"],
    },
    "notifications": {
        "en": ["notification", "email", "alert", "log", "sent", "message received",
               "test notification"],
        "hi": ["सूचना", "ईमेल", "अलर्ट", "लॉग", "भेजा"],
        "kn": ["ಅಧಿಸೂಚನೆ", "ಇಮೇಲ್", "ಎಚ್ಚರಿಕೆ", "ದಾಖಲೆ", "ಕಳುಹಿಸಿದ"],
        "te": ["నోటిఫికేషన్", "ఇమెయిల్", "హెచ్చరిక", "లాగ్", "పంపిన"],
    },
}

# Appliance names, asked about directly far more often than "which appliance
# uses the most". Without these, "how much does the geyser use" matched nothing
# in particular and fell through to a generic usage answer.
_APPLIANCE_TERMS = {
    "en": ["ac", "air conditioner", "aircon", "fridge", "refrigerator", "washing machine",
           "washer", "water heater", "geyser", "oven", "microwave", "tubelight",
           "tube light", "light", "lights", "fan", "tv", "television", "laptop",
           "computer", "dishwasher", "heater", "iron", "mixer"],
    "hi": ["एसी", "एयर कंडीशनर", "फ्रिज", "वाशिंग मशीन", "वॉशिंग मशीन", "वॉशर मशीन",
           "वॉशिंग मैचिन", "गीजर", "गेजर", "वॉटर हीटर", "ओवन", "माइक्रोवेव",
           "ट्यूबलाइट", "ट्यूब लाइट", "पंखा", "टीवी", "लैपटॉप", "कंप्यूटर",
           "डिशवॉशर", "इस्तरी"],
    "kn": ["ಎಸಿ", "ಏರ್ ಕಂಡಿಷನರ್", "ಫ್ರಿಡ್ಜ್", "ಒಲೆ", "ವಾಷಣ ಯಂತ್ರ", "ವಾಷಣ ಮೈಸಿನ್",
           "ಗೇಸರ್", "ನೀರಿನ ಹೆಚ್ಚುವರ", "ಮೈಕ್ರೋವೇವ್", "ಟ್ಯೂಬ್‌ಲೈಟ್", "ಫ್ಯಾನ್", "ಟಿವಿ",
           "ಲ್ಯಾಪ್‌ಟಾಪ್", "ಕಂಪ್ಯೂಟರ್", "ಡಿಶ್‌ವಾಶರ್", "ಇಸ್ತ್ರಿ"],
    "te": ["ఏసీ", "ఎయిర్ కండిషనర్", "ఫ్రిజ్", "వాషింగ్ మెషిన్", "గేసర్", "వాటర్ హీటర్",
           "ఓవెన్", "మైక్రోవేవ్", "ట్యూబ్‌లైట్", "ఫ్యాన్", "టీవీ", "ల్యాప్‌టాప్",
           "కంప్యూటర్", "డిశ్‌వాషర్", "ఇస్త్రీ"],
}

# Keyword fragments shorter than this are unusable. Scripts without spaces fuse
# morphemes, so "ఏ" (which) sits inside a large share of Telugu vocabulary and
# "कम" (less) sits inside "कम्प्यूटर" (computer); both produced false hits.
_MIN_KEYWORD_CHARS = 3

_compiled_cache: Dict[str, Any] = {}


def _keyword_pattern(keyword: str):
    """
    Compile a keyword matcher.

    Latin-script keywords are matched on word boundaries so "log" no longer
    fires on "login" and "less" no longer fires on "useless". Indic keywords are
    matched as substrings, because those scripts are written without spaces and
    a word-boundary match would reject every inflected form.
    """
    cached = _compiled_cache.get(keyword)
    if cached is not None:
        return cached
    if keyword.isascii():
        pattern = re.compile(r"(?<![a-z0-9])" + re.escape(keyword.lower())
                             + r"(?![a-z0-9])")
    else:
        pattern = re.compile(re.escape(keyword))
    _compiled_cache[keyword] = pattern
    return pattern


def _matches_topic(lower: str, keywords) -> bool:
    for keyword in keywords:
        if len(keyword.strip()) < _MIN_KEYWORD_CHARS and not keyword.isascii():
            continue
        if _keyword_pattern(keyword).search(lower):
            return True
    return False


_OUT_OF_SCOPE = {
    "en": ["weather", "stock", "share market", "movie", "film", "news", "sport", "cricket",
           "recipe", "cooking", "travel", "covid", "traffic", "politics", "history of "],
    "hi": ["मौसम", "शेयर", "फिल्म", "समाचार", "खेल", "क्रिकेट", "पकवान", "यात्रा", "कोविड",
           "यातायात", "राजनीति"],
    "kn": ["ಹವಾಮಾನ", "ಷೇರು", "ಚಲನಚಿತ್ರ", "ಸಿನಿಮಾ", "ಸುದ್ದಿ", "ಕ್ರೀಡೆ", "ಕ್ರಿಕೆಟ್", "ಅಡುಗೆ", "ಪ್ರಯಾಣ",
            "ಕೋವಿಡ್", "ಸಂಚಾರ", "ರಾಜಕೀಯ"],
    "te": ["వాతావరణం", "షేరు", "సినిమా", "వార్తలు", "క్రీడ", "క్రికెట్", "వంట", "ప్రయాణం",
           "కోవిడ్", "ట్రాఫిక్", "రాజకీయాలు"],
}

_APPLIANCE_LABEL_KEY = {
    "Kitchen": "app_kitchen",
    "Laundry Room": "app_laundry",
    "Water Heater & AC": "app_wh_ac",
    "Other": "app_other",
}

_NOTIFICATION_TYPE_KEY = NOTIFICATION_TYPE_KEY


def _prepare_household_data(household_data: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Return the household's real stored usage data.
    When the caller (UI) passes its own computed dataframe, use it so the
    chatbot answers match what the dashboard shows. Otherwise fall back to
    the stored dataset remapped to current dates.
    """
    if household_data is not None and isinstance(household_data, pd.DataFrame) and not household_data.empty:
        return household_data.copy()
    from model import load_data
    from data import remap_to_current_dates
    raw = load_data()
    return remap_to_current_dates(raw, last_n_days=90)


class ChatbotTool:
    """Base class for chatbot tools that query real household data."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def execute(self, household_id: str, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError


class GetUsageHistoryTool(ChatbotTool):
    """Query hourly usage history for a date range (real measured data)."""

    def __init__(self):
        super().__init__(
            name="get_usage_history",
            description="Get measured hourly electricity usage for a date range",
        )

    def execute(self, household_id: str, days_back: int = 7,
                household_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        try:
            df = _prepare_household_data(household_data)
            if df.empty or "Global_active_power" not in df.columns:
                return {"status": "no_data", "message": "no usage data stored"}

            recent = df.tail(int(days_back) * 24).copy()
            recent["datetime"] = pd.to_datetime(recent["datetime"])
            if recent.empty:
                return {"status": "no_data", "message": "No usage data in this range"}

            # Every row is a 1-hour reading in kW → kWh = kW * 1h.
            total_kwh = float(recent["Global_active_power"].sum())
            days_covered = int(recent["datetime"].dt.date.nunique())
            daily_avg = float(total_kwh / days_covered) if days_covered else 0.0
            hourly_max = float(recent["Global_active_power"].max())
            hourly_min = float(recent["Global_active_power"].min())

            return {
                "status": "success",
                "period_days": days_covered,
                "total_kwh": round(total_kwh, 2),
                "daily_avg_kwh": round(daily_avg, 2),
                "hourly_max_kw": round(hourly_max, 2),
                "hourly_min_kw": round(hourly_min, 2),
                "data_points": int(len(recent)),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}


class GetApplianceBreakdownTool(ChatbotTool):
    """Get usage breakdown by appliance category from real sub-meter data."""

    def __init__(self):
        super().__init__(
            name="get_appliance_breakdown",
            description="Get measured electricity usage split by appliance/category",
        )

    def execute(self, household_id: str, days_back: int = 7,
                household_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        try:
            df = _prepare_household_data(household_data)
            if df.empty or "Global_active_power" not in df.columns:
                return {"status": "no_data", "message": "no usage data stored"}

            recent = df.tail(int(days_back) * 24).copy()
            if recent.empty:
                return {"status": "no_data", "message": "No usage data in this range"}

            sub_cols = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
            present = [c for c in sub_cols if c in recent.columns]
            total_kwh = float(recent["Global_active_power"].sum())

            def _kwh(col):
                if col in recent.columns:
                    return float(recent[col].sum() / 1000.0)
                return 0.0

            kitchen = _kwh("Sub_metering_1")
            laundry = _kwh("Sub_metering_2")
            wh_ac = _kwh("Sub_metering_3")
            known = kitchen + laundry + wh_ac
            other = max(0.0, total_kwh - known)

            breakdown = {
                "Kitchen": round(kitchen, 2),
                "Laundry Room": round(laundry, 2),
                "Water Heater & AC": round(wh_ac, 2),
                "Other": round(other, 2),
            }
            total_kwh = round(total_kwh, 2)
            top = max(breakdown, key=breakdown.get)
            top_pct = (breakdown[top] / total_kwh * 100.0) if total_kwh > 0 else 0.0

            return {
                "status": "success",
                "period_days": int(days_back),
                "breakdown_kwh": breakdown,
                "total_kwh": total_kwh,
                "top_appliance": top,
                "top_appliance_kwh": breakdown[top],
                "top_appliance_pct": round(top_pct, 1),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}


class GetCurrentPredictionTool(ChatbotTool):
    """Get the model's predicted monthly cost from the stored forecast."""

    def __init__(self):
        super().__init__(
            name="get_current_prediction",
            description="Get the model-predicted monthly electricity cost",
        )

    def execute(self, household_id: str, tariff_rate: float = 8.0,
                scaling_factor: float = 1.0) -> Dict[str, Any]:
        try:
            from notifications import load_prediction_stats
            stats = load_prediction_stats(tariff_rate=float(tariff_rate))
            predicted_cost = round(float(stats["predicted_cost"]) * scaling_factor, 2)
            predicted_kwh = round(float(stats["predicted_kwh"]) * scaling_factor, 2)
            return {
                "status": "success",
                "predicted_monthly_cost_rs": predicted_cost,
                "predicted_monthly_kwh": predicted_kwh,
                "tariff_rate": float(tariff_rate),
                "month": stats["month"],
                "basis": "computed from the stored model forecast",
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}


class GetWeekComparisonTool(ChatbotTool):
    """Compare the most recent 7 days against the prior 7 days (real data)."""

    def __init__(self):
        super().__init__(
            name="get_week_comparison",
            description="Compare this week's measured usage/cost with the previous week",
        )

    def execute(self, household_id: str, tariff_rate: float = 8.0,
                household_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        try:
            from cost import weekly_cost
            df = _prepare_household_data(household_data)
            if df.empty or len(df) < 7 * 24:
                return {"status": "no_data", "message": "not enough data for a weekly comparison"}
            latest = pd.to_datetime(df["datetime"]).max()
            this_start = latest - timedelta(days=6)
            info = weekly_cost(df, str(this_start.date()), float(tariff_rate))
            # Only claim a comparison if the PRIOR week actually has substantial
            # measured coverage (a single partial day is not a fair baseline and
            # would produce a misleading percentage).
            from datetime import timedelta as _td
            pws = this_start.date() - _td(days=7)
            pwe = this_start.date() - _td(days=1)
            dt_col = pd.to_datetime(df["datetime"])
            prev_zone = df.loc[(dt_col.dt.date >= pws) & (dt_col.dt.date <= pwe)]
            prev_days_covered = int(prev_zone["datetime"].dt.date.nunique()) if not prev_zone.empty else 0
            prev_available = float(info.get("prev_total_kwh") or 0) > 0 and prev_days_covered >= 6
            return {
                "status": "success",
                "this_cost_rs": float(info.get("total_cost") or 0),
                "this_kwh": float(info.get("total_kwh") or 0),
                "prev_cost_rs": float(info.get("prev_total_cost") or 0),
                "prev_kwh": float(info.get("prev_total_kwh") or 0),
                "pct_change": float(info.get("pct_change") or 0),
                "prev_available": prev_available,
                "week_start": info.get("week_start"),
                "week_end": info.get("week_end"),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}


class GetOptimizationTipsTool(ChatbotTool):
    """Get tips referencing a specific real usage spike detected in the data."""

    def __init__(self):
        super().__init__(
            name="get_optimization_tips",
            description="Get tips tied to detected spikes in measured appliance usage",
        )

    def execute(self, household_id: str,
                household_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        try:
            from optimize import detect_anomalies
            df = _prepare_household_data(household_data)
            if df.empty or len(df) < 7 * 24:
                return {"status": "no_data", "message": "not enough data to detect patterns"}

            anomalies = detect_anomalies(df, window_days=7)
            tips = []
            if isinstance(anomalies, pd.DataFrame) and not anomalies.empty:
                for _, row in anomalies.tail(3).iterrows():
                    usage = float(row["Global_active_power"])
                    mean = float(row["rolling_mean"])
                    pct = ((usage - mean) / mean * 100.0) if mean else 0.0
                    tips.append({
                        "appliance_name": row["top_appliance"],
                        "current_usage": round(usage, 2),
                        "avg_usage": round(mean, 2),
                        "increase_pct": round(pct, 1),
                        "when": str(row["datetime"]),
                    })
            return {
                "status": "success",
                "anomalies_detected": int(len(anomalies)) if isinstance(anomalies, pd.DataFrame) else 0,
                "tips": tips,
                "basis": "detected from real usage data",
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}


class GetFamilyNotificationLogTool(ChatbotTool):
    """Query the household's notification audit log (who was sent what)."""

    def __init__(self):
        super().__init__(
            name="get_family_notification_log",
            description="Get the audit log of notifications sent to this household",
        )

    def execute(self, household_id: str, limit: int = 10) -> Dict[str, Any]:
        try:
            log = get_db().get_notification_log(household_id, limit=int(limit))
            entries = []
            for row in log:
                entries.append({
                    "type": row.get("notification_type") or "unknown",
                    "recipient": row.get("recipient_email") or "",
                    "status": row.get("status") or "unknown",
                    "time": row.get("sent_at") or "",
                    "subject": row.get("subject") or "",
                })
            return {
                "status": "success",
                "count": int(len(entries)),
                "entries": entries,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}


class EnergyPulseChatbot:
    """
    Grounded Q&A chatbot that answers questions about household energy.
    All answers are validated against real data before display.
    """

    def __init__(self):
        self.db = get_db()
        self.guard = LLMHallucinationGuard()
        self.tools = {
            "get_usage_history": GetUsageHistoryTool(),
            "get_appliance_breakdown": GetApplianceBreakdownTool(),
            "get_current_prediction": GetCurrentPredictionTool(),
            "get_week_comparison": GetWeekComparisonTool(),
            "get_optimization_tips": GetOptimizationTipsTool(),
            "get_family_notification_log": GetFamilyNotificationLogTool(),
        }

    def answer_question(self, household_id: str, email: str, question: str,
                        language: str = "en", tariff_rate: float = 8.0,
                        household_data: Optional[pd.DataFrame] = None,
                        scaling_factor: float = 1.0) -> Tuple[str, bool, Dict[str, Any]]:
        """
        Answer a user question about their own household's energy usage.

        Returns (answer_text, is_valid_grounded, metadata_dict).
        """
        self.guard = LLMHallucinationGuard()
        language = language or "en"

        intent = self._plan_tools(question, language)
        if intent == ["out_of_scope"]:
            answer = t_lang("chat_out_of_scope", language)
            self.db.save_conversation(
                household_id=household_id, email=email, question=question, answer=answer,
                language=language, tool_calls=["out_of_scope"], grounding_data={}, is_valid=True,
            )
            return (answer, True, {"tool_calls": ["out_of_scope"], "tool_results": {},
                                   "validation_valid": True, "validation_issues": []})

        tool_results = {}
        for tool_name in intent:
            tool = self.tools.get(tool_name)
            if tool is None:
                continue
            try:
                if tool_name == "get_current_prediction":
                    result = tool.execute(household_id, tariff_rate=tariff_rate,
                                          scaling_factor=scaling_factor)
                elif tool_name == "get_usage_history":
                    result = tool.execute(household_id, days_back=7, household_data=household_data)
                elif tool_name == "get_appliance_breakdown":
                    result = tool.execute(household_id, days_back=7, household_data=household_data)
                elif tool_name == "get_week_comparison":
                    result = tool.execute(household_id, tariff_rate=tariff_rate,
                                          household_data=household_data)
                elif tool_name == "get_optimization_tips":
                    result = tool.execute(household_id, household_data=household_data)
                else:
                    result = tool.execute(household_id)
                tool_results[tool_name] = result
                self._register_tool_result_with_guard(tool_name, result)
            except Exception as e:  # never let a tool break the conversation
                # The exception text can contain SQL, file paths or column
                # names, none of which belong in a chat bubble.
                tool_results[tool_name] = {
                    "status": "error",
                    "message": t_lang("chat_tool_failed", language),
                    "error_type": type(e).__name__,
                }

        answer = self._generate_answer(question, tool_results, language)

        is_valid, validation_issues, corrected = self.guard.validate_response(answer, language=language)
        if not is_valid and corrected != answer:
            answer = corrected

        self.db.save_conversation(
            household_id=household_id, email=email, question=question, answer=answer,
            language=language, tool_calls=intent, grounding_data=tool_results,
            is_valid=is_valid,
        )
        metadata = {
            "tool_calls": intent,
            "tool_results": tool_results,
            "validation_valid": is_valid,
            "validation_issues": validation_issues,
        }
        return (answer, is_valid, metadata)

    # ── Guard wiring ───────────────────────────────────────

    def _register_tool_result_with_guard(self, tool_name: str, result: Dict[str, Any]) -> None:
        """Register every real number (plus date/time components) with the guard."""
        if not isinstance(result, dict) or result.get("status") != "success":
            return
        # Mark that grounded tool data exists so the guard's "missing data"
        # check does not flag answers built from real tool output.
        self.guard.tool_results[tool_name] = True
        for key, val in result.items():
            if isinstance(val, (int, float)):
                self.guard.register_computed_stat(f"{tool_name}_{key}", float(val))
            elif isinstance(val, str):
                # Register numeric components of ISO date/time strings (e.g. "2025-09-10 14:00")
                # so the validation step recognises them as grounded data, not invented claims.
                for num in re.findall(r"\d+", val):
                    self.guard.register_computed_stat(f"{tool_name}_{key}", float(num))
            elif isinstance(val, dict):
                for sub_key, sub_val in val.items():
                    if isinstance(sub_val, (int, float)):
                        self.guard.register_computed_stat(f"{tool_name}_{key}_{sub_key}", float(sub_val))
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        for sub_key, sub_val in item.items():
                            if isinstance(sub_val, (int, float)):
                                self.guard.register_computed_stat(f"{tool_name}_{key}_{sub_key}", float(sub_val))
                            elif isinstance(sub_val, str):
                                for num in re.findall(r"\d+", sub_val):
                                    self.guard.register_computed_stat(f"{tool_name}_{key}_{sub_key}", float(num))

    # ── Intent planning ────────────────────────────────────

    def _plan_tools(self, question: str, language: str) -> List[str]:
        """
        Decide which real-data tools to call, driven by the question language.

        A question that matches nothing in scope is reported as out of scope.
        The previous version fell back to three tools, so "what is the capital
        of France" was answered with household electricity figures.
        """
        lower = (question or "").strip().lower()
        if not lower:
            return ["out_of_scope"]

        out_kws = _OUT_OF_SCOPE.get(language, _OUT_OF_SCOPE["en"])
        if _matches_topic(lower, out_kws):
            return ["out_of_scope"]

        matched = set()
        for topic, langs in _QUESTION_TERMS.items():
            kws = langs.get(language, langs["en"])
            if _matches_topic(lower, kws):
                matched.add(topic)

        appliance_terms = _APPLIANCE_TERMS.get(language, _APPLIANCE_TERMS["en"])
        if _matches_topic(lower, appliance_terms):
            matched.add("appliance")
            # "how much does the geyser cost" is an appliance question, not a
            # forecast question, so the cheaper appliance answer is the useful one.
            matched.discard("prediction")

        if not matched:
            return ["out_of_scope"]

        # "Why is my bill higher / compare last week" style questions.
        # When the person explicitly asks about an appliance/category, treat it
        # as a what-type question instead, so the appliance answer is returned.
        if "why_compare" in matched and "appliance" not in matched:
            tools = ["get_week_comparison"]
            for topic, tool in [
                ("usage_history", "get_usage_history"),
                ("appliance", "get_appliance_breakdown"),
                ("optimize", "get_optimization_tips"),
                ("notifications", "get_family_notification_log"),
            ]:
                if topic in matched and tool not in tools:
                    tools.append(tool)
            return tools

        tools: List[str] = []
        for topic, tool in [
            ("prediction", "get_current_prediction"),
            ("usage_history", "get_usage_history"),
            ("appliance", "get_appliance_breakdown"),
            ("optimize", "get_optimization_tips"),
            ("notifications", "get_family_notification_log"),
        ]:
            if topic in matched and tool not in tools:
                tools.append(tool)
        return tools

    # ── Answer generation ──────────────────────────────────

    def _generate_answer(self, question: str, tool_results: Dict[str, Any],
                         language: str) -> str:
        if not tool_results:
            return t_lang("chat_no_data", language)

        success_results = {k: v for k, v in tool_results.items() if v.get("status") == "success"}
        if not success_results:
            return t_lang("chat_no_data", language)

        lines: List[str] = []
        for tool_name in [
            "get_week_comparison", "get_usage_history", "get_appliance_breakdown",
            "get_current_prediction", "get_optimization_tips", "get_family_notification_log",
        ]:
            result = success_results.get(tool_name)
            if result is None:
                continue
            line = self._render_tool_line(tool_name, result, language)
            if line:
                lines.append(line)

        if not lines:
            return t_lang("chat_no_data", language)
        return "\n".join(lines)

    def _render_tool_line(self, tool_name: str, result: Dict[str, Any], language: str) -> str:
        if tool_name == "get_usage_history":
            return t_lang("chat_usage", language,
                          days=result.get("period_days", 7),
                          total_kwh=result.get("total_kwh", 0),
                          daily_avg=result.get("daily_avg_kwh", 0),
                          hourly_max=result.get("hourly_max_kw", 0))

        if tool_name == "get_week_comparison":
            if not result.get("prev_available"):
                return t_lang("chat_no_prior_week", language)
            return t_lang("chat_week_compare", language,
                          this_cost=result.get("this_cost_rs", 0),
                          this_kwh=result.get("this_kwh", 0),
                          prev_cost=result.get("prev_cost_rs", 0),
                          prev_kwh=result.get("prev_kwh", 0),
                          pct=result.get("pct_change", 0))

        if tool_name == "get_appliance_breakdown":
            top_label = self._localize_appliance(result.get("top_appliance", "Other"), language)
            return t_lang("chat_top_appliance", language,
                          appliance=top_label,
                          kwh=result.get("top_appliance_kwh", 0),
                          pct=result.get("top_appliance_pct", 0),
                          total=result.get("total_kwh", 0),
                          days=result.get("period_days", 7))

        if tool_name == "get_current_prediction":
            return t_lang("chat_prediction", language,
                          month=format_localized_month(result.get("month", "N/A"), language),
                          cost=result.get("predicted_monthly_cost_rs", 0),
                          kwh=result.get("predicted_monthly_kwh", 0),
                          rate=result.get("tariff_rate", 0))

        if tool_name == "get_optimization_tips":
            tips = result.get("tips") or []
            if not tips:
                return t_lang("chat_no_tips", language)
            tip_lines = []
            for tip in tips:
                app_label = self._localize_appliance(tip.get("appliance_name", "Other"), language)
                tip_lines.append(t_lang("chat_tip_line", language,
                                        appliance=app_label,
                                        usage=tip.get("current_usage", 0),
                                        when=tip.get("when", ""),
                                        mean=tip.get("avg_usage", 0),
                                        increase_pct=tip.get("increase_pct", 0)))
            return "\n".join(tip_lines)

        if tool_name == "get_family_notification_log":
            entries = result.get("entries") or []
            if not entries:
                return t_lang("chat_notif_empty", language)
            latest = entries[0]
            summary = (
                f"{self._localize_notification_type(latest.get('type', ''), language)} → "
                f"{latest.get('recipient', '')} ({self._localize_status(latest.get('status', ''), language)}, "
                f"{latest.get('time', '')})"
            )
            return t_lang("chat_notif_log", language, n=result.get("count", len(entries)), summary=summary)

        return ""

    def _localize_appliance(self, name: str, language: str) -> str:
        key = _APPLIANCE_LABEL_KEY.get(name)
        if key:
            return t_lang(key, language)
        return name

    def _localize_notification_type(self, type_name: str, language: str) -> str:
        key = _NOTIFICATION_TYPE_KEY.get(type_name)
        if key:
            return t_lang(key, language)
        return type_name

    def _localize_status(self, status: str, language: str) -> str:
        key = "notif_status_sent" if status == "sent" else "notif_status_failed"
        if status not in ("sent", "failed"):
            return status
        return t_lang(key, language)

    # ── History helpers ────────────────────────────────────

    def get_conversation_history(self, household_id: str, email: str,
                                 limit: int = 20) -> List[Dict[str, Any]]:
        return self.db.get_conversation_history(household_id, email, limit=limit)

    def clear_conversation_history(self, household_id: str, email: str) -> int:
        """Permanently remove this user's conversation history. Returns rows deleted."""
        return self.db.clear_conversations(household_id, email)


# Singleton instance
_chatbot = None


def get_chatbot() -> EnergyPulseChatbot:
    """Get or create the chatbot singleton."""
    global _chatbot
    if _chatbot is None:
        _chatbot = EnergyPulseChatbot()
    return _chatbot