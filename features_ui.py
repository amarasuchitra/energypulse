"""
EnergyPulse — Feature Tab UI
============================
Renderer functions for the three integrated feature tabs:

  1. Family tab     — add / edit / remove household members + preferences
  2. Notifications  — backend status, threshold, live trigger, test, audit log
  3. Energy Chat    — grounded multilingual Q&A over the household's real data

These tabs are wired into app.py and receive the SAME scaled dataframe
(full_data) and scaling factor the rest of the dashboard uses, so chatbot
and notification numbers always match the visible dashboard figures.
"""

import hashlib
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any

import streamlit as st
from motion import notify

from db import get_db
from i18n import T, t_lang, LANGS, RELATIONSHIP_CODES, NOTIFICATION_TYPE_KEY, localized_relationship, relationship_code_from_label, format_localized_month
from notifications import get_notification_service, dispatch_household_alerts
from chatbot import get_chatbot
from cost import next_month_cost


# ──────────────────────────────────────────────────────────────────────
# Small visual helpers (match app.py's styles which are already injected
# into the page via the main app's CSS).
# ──────────────────────────────────────────────────────────────────────
def _section(title: str, icon: str = ""):
    icon_html = f'<span style="font-size:1rem;">{icon}</span>' if icon else ""
    st.markdown(
        f'<div class="section-header"><h2>{icon_html} {title}</h2>'
        f'<div class="section-line"></div></div>',
        unsafe_allow_html=True,
    )


def _lang_codes() -> list:
    return list(LANGS.keys())


def _lang_names() -> list:
    return [LANGS[c] for c in LANGS]


def _code_from_name(name: str) -> str:
    for code, label in LANGS.items():
        if label == name:
            return code
    return "en"


# ──────────────────────────────────────────────────────────────────────
# Household resolution
# ──────────────────────────────────────────────────────────────────────
def resolve_household_id(email: str) -> str:
    """The logged-in account may exist only in-memory; sync it to the DB so the
    persistent family/notification/chat tables have a stable household key."""
    email = (email or "").strip().lower()
    user = get_db().get_user(email)
    if user and user.get("household_id"):
        return user["household_id"]
    return hashlib.md5(email.encode()).hexdigest()


def ensure_login(user_email: str, name: str = None, is_guest: bool = False) -> str:
    """Persist the current login and return its household_id."""
    lang = "en"
    try:
        lang = st.session_state.get("lang", "en")
    except Exception:
        pass
    info = get_db().ensure_user_login(user_email, name=name, is_guest=is_guest,
                                      preferred_language=lang)
    return info["household_id"]


# ──────────────────────────────────────────────────────────────────────
# 1. FAMILY TAB
# ──────────────────────────────────────────────────────────────────────
def render_family_tab(db, household_id: str) -> None:
    _section(T("fam_title"))
    st.markdown(T("fam_intro"))

    members = db.get_family_members(household_id)

    # ── Edit / remove existing members ─────────────────────────────
    _section(T("fam_existing"))
    if not members:
        st.info(T("fam_none"))
    for member in members:
        m_id = int(member["id"])
        rel_code = relationship_code_from_label(member.get("relationship"))
        with st.expander(
            f"{member['name']} ({localized_relationship(rel_code)}, {member['email']})",
            key=f"fam_exp_{m_id}",
        ):
            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input(T("fam_name"), value=member["name"], key=f"fam_name_{m_id}")
            with c2:
                rel_options = [localized_relationship(code) for code in RELATIONSHIP_CODES]
                rel_code_by_label = {
                    localized_relationship(code): code for code in RELATIONSHIP_CODES
                }
                rel_index = RELATIONSHIP_CODES.index(rel_code)
                relationship = st.selectbox(T("fam_relationship"), rel_options,
                                            index=rel_index, key=f"fam_rel_{m_id}")
            c3, c4 = st.columns(2)
            with c3:
                email = st.text_input(T("fam_email"), value=member["email"], key=f"fam_email_{m_id}")
            with c4:
                phone = st.text_input(T("fam_phone"), value=member.get("phone") or "",
                                      key=f"fam_phone_{m_id}")
            default_lang = member.get("preferred_language") or "en"
            lang_name = LANGS.get(default_lang, LANGS["en"])
            lang_sel = st.selectbox(T("fam_language"), _lang_names(),
                                    index=_lang_names().index(lang_name), key=f"fam_lang_{m_id}")
            c5, c6 = st.columns(2)
            with c5:
                notify_bills = st.checkbox(
                    T("fam_notify_bills"),
                    value=bool(member.get("notification_bill_alerts")),
                    key=f"fam_bills_{m_id}",
                )
            with c6:
                notify_tips = st.checkbox(
                    T("fam_notify_tips"),
                    value=bool(member.get("notification_optimization_tips")),
                    key=f"fam_tips_{m_id}",
                )
            b1, b2 = st.columns([1, 1])
            with b1:
                if st.button(T("fam_save"), key=f"fam_save_{m_id}", type="primary"):
                    ok, msg = db.update_family_member(m_id, household_id, {
                        "name": name,
                        "relationship": relationship_code_from_label(relationship),
                        "email": email,
                        "phone": phone,
                        "preferred_language": _code_from_name(lang_sel),
                        "notification_bill_alerts": notify_bills,
                        "notification_optimization_tips": notify_tips,
                    })
                    if ok:
                        st.success(T("fam_updated", name=name))
                        notify("success", T("fam_updated", name=name))
                        st.rerun()
                    else:
                        st.error(_family_error(msg))
            with b2:
                if st.button(T("fam_remove"), key=f"fam_remove_{m_id}"):
                    if db.remove_family_member(m_id, household_id):
                        st.success(T("fam_removed", name=member["name"]))
                        notify("info", T("fam_removed", name=member["name"]))
                        st.rerun()

    # ── Add a new member ────────────────────────────────────────────
    _section(T("fam_add"))
    with st.form("fam_add_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            new_name = st.text_input(T("fam_name"), key="fam_new_name")
        with c2:
            rel_code_by_label = {
                localized_relationship(code): code for code in RELATIONSHIP_CODES
            }
            new_rel = st.selectbox(T("fam_relationship"),
                                   [localized_relationship(code) for code in RELATIONSHIP_CODES],
                                   key="fam_new_rel")
        c3, c4 = st.columns(2)
        with c3:
            new_email = st.text_input(T("fam_email"), key="fam_new_email")
        with c4:
            new_phone = st.text_input(T("fam_phone"), key="fam_new_phone")
        new_lang = st.selectbox(T("fam_language"), _lang_names(), key="fam_new_lang")
        c5, c6 = st.columns(2)
        with c5:
            new_bills = st.checkbox(T("fam_notify_bills"), value=True, key="fam_new_bills")
        with c6:
            new_tips = st.checkbox(T("fam_notify_tips"), value=True, key="fam_new_tips")
        submitted = st.form_submit_button(T("fam_add_btn"), type="primary", width="stretch")
        if submitted:
            ok, msg = db.add_family_member(
                household_id=household_id,
                name=new_name,
                relationship=rel_code_by_label.get(new_rel, "Other"),
                email=new_email,
                phone=new_phone,
                preferred_language=_code_from_name(new_lang),
                notify_bills=new_bills,
                notify_tips=new_tips,
            )
            if ok:
                st.success(T("fam_added", name=new_name))
                notify("success", T("fam_added", name=new_name))
                st.rerun()
            else:
                st.error(_family_error(msg))


def _family_error(msg: str) -> str:
    mapping = {
        "invalid_email": T("fam_err_invalid_email"),
        "duplicate_email": T("fam_err_duplicate_email"),
        "duplicate_member": T("fam_err_duplicate_member"),
        "missing_name": T("fam_err_name"),
        "not_found": T("fam_err_failed"),
    }
    return mapping.get(msg, T("fam_err_failed"))


# ──────────────────────────────────────────────────────────────────────
# 2. NOTIFICATIONS TAB
# ──────────────────────────────────────────────────────────────────────
def render_notifications_tab(db, household_id: str, primary_email: str,
                             language: str, tariff_rate: float,
                             history: Optional[pd.DataFrame],
                             forecast_df: Optional[pd.DataFrame]) -> None:
    service = get_notification_service()
    # The daily forecast is the notification people come here for, so it leads the page.
    latest = next((e for e in db.get_notification_log(household_id, limit=200)
                   if e.get("notification_type") == "daily_forecast"), None)
    if latest:
        _section("Today's forecast notification")
        with st.container(border=True):
            st.markdown(f"**{latest.get('subject') or ''}**")
            st.text(latest.get("body_text") or "")
            from toolkit_ui import whatsapp_link
            st.link_button("Share on WhatsApp", whatsapp_link(latest.get("subject") or "", latest.get("body_text") or ""))
            st.caption("Sent once a day to every household member"
                       + (" by email and in the app." if latest.get("status") == "sent"
                          else ". Shown in the app; set up email below to receive it by email too."))

    _section(T("notif_title"))
    st.markdown(T("notif_intro"))

    # Backend status. Credentials being present is not a connection, so the
    # metric only says "Connected" once a real delivery has succeeded.
    c1, c2 = st.columns(2)
    email_backend = service.email_backend
    email_state = service.email_status()
    _EMAIL_STATE_KEY = {
        "not_configured": "notif_not_configured",
        "configured": "notif_configured",
        "verified": "notif_connected",
        "failed": "notif_send_failed",
    }
    with c1:
        email_status = T(_EMAIL_STATE_KEY[email_state])
        tick = ""
        st.metric(T("notif_email_backend"), f"{tick}{email_status}")
        if service.SMTP_PORT_ERROR:
            st.warning(service.SMTP_PORT_ERROR)
        if email_state == "failed" and service.last_email_error:
            st.caption(t_lang("notif_last_error", language,
                              error=service.last_email_error))
        if email_backend == "smtp":
            st.caption(t_lang("notif_email_caption_smtp", language,
                              user=service.SMTP_USERNAME))
        elif email_backend == "sendgrid":
            st.caption(T("notif_email_caption_sendgrid"))
        if email_state == "configured":
            # Honest: the credentials were found, nothing has been sent yet.
            st.caption(T("notif_configured_unverified"))
        if not email_backend:
            missing = service.email_missing()
            if missing:
                st.caption(t_lang("notif_missing", language, vars=", ".join(missing)))
    with c2:
        if service.sms_configured():
            st.metric(T("notif_sms_backend"), T('notif_connected'))
        else:
            # SMS is optional, so an absent Twilio key is a normal state and is
            # not reported as a problem.
            st.metric(T("notif_sms_backend"), T("notif_sms_not_configured"))
        st.caption(T("notif_sms_optional"))
        missing_sms = service.sms_missing()
        if missing_sms:
            st.caption(t_lang("notif_missing", language, vars=", ".join(missing_sms)))

    if not email_backend:
        with st.expander(T("notif_howto_title"), expanded=True):
            st.markdown(T("notif_howto_intro"))
            st.markdown(T("notif_howto_step1"))
            st.markdown(T("notif_howto_step2"))
            st.markdown(T("notif_howto_step3"))
            st.markdown(T("notif_howto_step4"))
            st.code(T("notif_howto_env"), language="properties")
            st.markdown(T("notif_howto_step5"))
            st.markdown(T("notif_howto_secrets"))
            st.markdown(f"**{T('notif_howto_alt_title')}**")
            st.markdown(T("notif_howto_alt"))

    # Bill alert threshold
    settings = db.get_household_settings(household_id)
    threshold = float(settings.get("bill_threshold") or 1500.0)
    _section(T("notif_threshold"))
    new_threshold = st.number_input(
        T("notif_threshold"), min_value=0.0, value=threshold, step=50.0,
        help=T("notif_threshold_help"), key="notif_threshold_input",
    )
    if float(new_threshold) != threshold:
        db.update_household_settings(household_id, {"bill_threshold": float(new_threshold)})

    # Run scheduled checks now (grounded in the dashboard's own numbers)
    nm_info = next_month_cost(forecast_df, tariff_rate) if forecast_df is not None and not forecast_df.empty \
        else {"total_cost": 0.0, "total_kwh": 0.0, "month": "N/A"}
    _section(T("notif_test"))
    cA, cB = st.columns([1, 1])
    with cA:
        recipients = list(dict.fromkeys(
            [primary_email] + [m["email"] for m in db.get_family_members(household_id)]
        ))
        recipient = st.selectbox(T("notif_test_email"), recipients, key="notif_test_recipient")
        if st.button(T("notif_test_btn"), key="notif_send_test", type="primary",
                     disabled=not recipients):
            with st.spinner("..."):
                ok, msg = service.send_test_notification(
                    household_id, recipient, language=language,
                    predicted_cost=float(nm_info.get("total_cost") or 0),
                    predicted_kwh=float(nm_info.get("total_kwh") or 0),
                    month=nm_info.get("month", "N/A"),
                )
            # msg carries the provider's own words, so an authentication
            # refusal and an unreachable host no longer look the same.
            if ok:
                st.success(t_lang("notif_test_ok", language, email=recipient))
                notify("success", t_lang("notif_test_ok", language, email=recipient))
            else:
                st.error(t_lang("notif_test_failed", language,
                                email=recipient,
                                error=msg or t_lang("notif_status_fail", language)))
    with cB:
        if st.button(T("notif_run_now"), key="notif_run_checks"):
            with st.spinner("..."):
                status = dispatch_household_alerts(
                    household_id, primary_email,
                    predicted_cost=float(nm_info.get("total_cost") or 0),
                    predicted_kwh=float(nm_info.get("total_kwh") or 0),
                    month=nm_info.get("month", "N/A"),
                    tariff_rate=tariff_rate,
                    history_df=history,
                    language=language,
                )
            if status.get("failed"):
                st.warning(T("notif_status_fail"))
            elif status.get("sent"):
                st.success(T("notif_status_ok"))
            else:
                st.info(T("notif_none"))

    # Audit log
    _section(T("notif_history"))
    log = db.get_notification_log(household_id, limit=50)
    if not log:
        st.info(T("notif_none"))
        return

    rows = []
    for entry in log:
        rows.append({
            "time": entry.get("sent_at") or "",
            "type": _localize_notif_type(entry.get("notification_type") or "", language),
            "to": entry.get("recipient_email") or "",
            "status": _localize_status(entry.get("status") or "", language),
            "subject": entry.get("subject") or "",
            "error": entry.get("error_message") or "",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _localize_notif_type(type_name: str, language: str) -> str:
    key = NOTIFICATION_TYPE_KEY.get(type_name)
    return t_lang(key, language) if key else type_name


def _localize_status(status: str, language: str) -> str:
    if status == "sent":
        return t_lang("notif_status_sent", language)
    if status == "failed":
        return t_lang("notif_status_failed", language)
    return status


# ──────────────────────────────────────────────────────────────────────
# 3. ENERGY CHAT TAB
# ──────────────────────────────────────────────────────────────────────
def render_chat_tab(db, household_id: str, email: str, language: str,
                    full_data: Optional[pd.DataFrame], scaling_factor: float,
                    tariff_rate: float) -> None:
    _section(T("chat_title"))
    st.markdown(T("chat_intro"))

    chatbot = get_chatbot()

    # New conversation button
    top = st.columns([5, 1])[1]
    with top:
        if st.button(T("chat_clear"), key="chat_clear_btn"):
            chatbot.clear_conversation_history(household_id, email)
            st.success(T("chat_cleared"))
            notify("info", T("chat_cleared"))
            st.rerun()

    history = chatbot.get_conversation_history(household_id, email, limit=20)
    if not history:
        st.info(T("chat_empty"))
    for turn in history:
        question = turn.get("question") or ""
        answer = turn.get("answer") or ""
        if not question and not answer:
            continue
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            st.markdown(answer)
            if not turn.get("is_valid_grounded", True) and turn.get("message_type") == "question":
                st.caption(T("chat_guard_note"))

    st.markdown("---")
    with st.form("chat_form", clear_on_submit=True):
        question = st.text_input(T("chat_title"), placeholder=T("chat_placeholder"),
                                 key="chat_input", label_visibility="collapsed")
        ask_col = st.columns([3, 1])
        with ask_col[1]:
            submitted = st.form_submit_button(T("chat_ask"), type="primary", width="stretch")
        if submitted and question and question.strip():
            with ask_col[0]:
                with st.spinner("..."):
                    answer, is_valid, _metadata = chatbot.answer_question(
                        household_id=household_id,
                        email=email,
                        question=question.strip(),
                        language=language,
                        tariff_rate=tariff_rate,
                        household_data=full_data,
                        scaling_factor=scaling_factor,
                    )
            st.rerun()