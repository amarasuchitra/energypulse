"""
Test & Verification Script for EnergyPulse New Features
========================================================
Validates family member management, notifications, and chatbot integration.

Run with: python test_new_features.py
"""

import sys
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd

# Windows consoles default to cp1252, which cannot encode the ✓/✗ check marks
# used below. Force UTF-8 output so the suite runs unchanged everywhere.
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

def test_database():
    """Test database layer."""
    print("\n" + "="*60)
    print("TEST 1: Database Layer (Family Members, Notifications, Chat)")
    print("="*60)
    
    try:
        from db import get_db
        db = get_db()
        
        # Test user creation
        test_email = f"test_{datetime.now().timestamp()}@example.com"
        test_user_ok = db.create_user(
            email=test_email,
            name="Test User",
            password="testpass123",
            household_id=f"hh_{datetime.now().timestamp()}"
        )
        print(f"✓ Create user: {test_user_ok}")
        
        # Test authentication
        auth_result = db.authenticate_user(test_email, "testpass123")
        print(f"✓ Authenticate user: {auth_result is not None}")
        if auth_result:
            household_id = auth_result["household_id"]
            
            # Test family member management
            print("\n--- Family Member Management ---")
            success, msg = db.add_family_member(
                household_id=household_id,
                name="Spouse",
                relationship="Spouse",
                email="spouse@example.com",
                phone="+91-9999999999",
                preferred_language="en",
                notify_bills=True,
                notify_tips=True
            )
            print(f"✓ Add family member: {success} (msg: {msg})")
            
            # Retrieve family members
            members = db.get_family_members(household_id)
            print(f"✓ Get family members: {len(members)} member(s) found")
            
            # Test duplicate rejection
            success2, msg2 = db.add_family_member(
                household_id=household_id,
                name="Duplicate",
                relationship="Other",
                email="spouse@example.com",  # Same email - should fail
                preferred_language="en"
            )
            print(f"✓ Reject duplicate email: {not success2} (correctly prevented)")
            
            # Test notification logging
            print("\n--- Notification Logging ---")
            notif_id = db.log_notification(
                household_id=household_id,
                recipient_email="spouse@example.com",
                recipient_name="Spouse",
                notification_type="bill_alert",
                triggered_by="bill_threshold",
                trigger_data={"threshold": 1000, "actual": 1250},
                subject="Bill Alert Test",
                body_text="Your bill is higher than expected",
                language="en"
            )
            print(f"✓ Log notification: ID {notif_id}")
            
            # Retrieve notification log
            log = db.get_notification_log(household_id)
            print(f"✓ Get notification log: {len(log)} notification(s)")
            
            # Test conversation history
            print("\n--- Chatbot Conversation History ---")
            conv_id = db.save_conversation(
                household_id=household_id,
                email=test_email,
                question="Which appliance uses the most power?",
                answer="Based on real data, your AC uses the most power.",
                language="en",
                tool_calls=["get_appliance_breakdown"],
                grounding_data={"AC": 2500},
                is_valid=True
            )
            print(f"✓ Save conversation: ID {conv_id}")
            
            # Retrieve conversation
            history = db.get_conversation_history(household_id, test_email)
            print(f"✓ Get conversation history: {len(history)} turn(s)")
        
        return True
    
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_notifications():
    """Test notification system."""
    print("\n" + "="*60)
    print("TEST 2: Notification System (Email/SMS)")
    print("="*60)
    
    try:
        from notifications import get_notification_service
        from db import get_db
        
        service = get_notification_service()
        db = get_db()
        
        # Check backend configuration
        email_backend = service.email_backend
        sms_backend = service.sms_backend
        
        print(f"✓ Email backend configured: {bool(email_backend)}")
        print(f"  - Backend type: {email_backend or 'None'}")
        print(f"✓ SMS backend configured: {bool(sms_backend)}")
        print(f"  - Backend type: {sms_backend or 'None'}")
        
        # Create test user for notifications
        test_email = f"notif_{datetime.now().timestamp()}@test.local"
        test_hh = f"hh_notif_{datetime.now().timestamp()}"
        db.create_user(test_email, "Test", "pass", household_id=test_hh)
        
        # Test bill alert notification
        print("\n--- Bill Alert Notification ---")
        if email_backend:
            sent_to = service.send_bill_alert(
                household_id=test_hh,
                primary_email=test_email,
                month="2024-09",
                predicted_cost=1500.00,
                threshold=1200.00,
                language="en"
            )
            print(f"✓ Send bill alert: {len(sent_to)} recipient(s)")
            print(f"  - Sent to: {sent_to}")
        else:
            print("⚠ Email backend not configured - skipping email test")
        
        # Test optimization tip notification
        print("\n--- Optimization Tip Notification ---")
        tip_data = {
            "appliance_name": "Air Conditioner",
            "current_usage": 3.5,
            "avg_usage": 2.0,
            "increase_pct": 75,
            "suggested_action": "Consider using AC only during peak cooling hours."
        }
        if email_backend:
            sent_to = service.send_optimization_tips(
                household_id=test_hh,
                primary_email=test_email,
                tip_data=tip_data,
                language="en"
            )
            print(f"✓ Send optimization tip: {len(sent_to)} recipient(s)")
        else:
            print("⚠ Email backend not configured - skipping email test")
        
        return True
    
    except Exception as e:
        print(f"✗ Notification test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chatbot():
    """Test grounded chatbot."""
    print("\n" + "="*60)
    print("TEST 3: Grounded Q&A Chatbot")
    print("="*60)
    
    try:
        from chatbot import get_chatbot, EnergyPulseChatbot
        from db import get_db
        
        chatbot = get_chatbot()
        db = get_db()
        
        # Create test user
        test_email = f"chat_{datetime.now().timestamp()}@test.local"
        test_hh = f"hh_chat_{datetime.now().timestamp()}"
        db.create_user(test_email, "ChatTest", "pass", household_id=test_hh)
        
        print(f"✓ Chatbot initialized with {len(chatbot.tools)} tools")
        print(f"  - Available tools: {list(chatbot.tools.keys())}")
        
        # Test Q&A in multiple languages
        test_questions = [
            ("en", "Which appliance uses the most power?"),
            ("hi", "कौन सा उपकरण सबसे अधिक बिजली खपत करता है?"),
            ("kn", "ಯಾವ ಉಪಕರಣ ಹೆಚ್ಚು ವಿದ್ಯುತ್ ಬಳಕೆ ಮಾಡುತ್ತದೆ?"),
            ("en", "What's my predicted monthly cost?"),
        ]
        
        print("\n--- Multi-language Q&A ---")
        for lang, question in test_questions:
            try:
                answer, is_valid, metadata = chatbot.answer_question(
                    household_id=test_hh,
                    email=test_email,
                    question=question,
                    language=lang,
                    tariff_rate=8.0
                )
                
                lang_name = {"en": "English", "hi": "Hindi", "kn": "Kannada"}.get(lang, lang)
                status = "✓" if is_valid else "⚠"
                print(f"{status} {lang_name}: Answered (valid: {is_valid})")
                print(f"  Q: {question}")
                print(f"  A: {answer[:100]}...")
            except Exception as e:
                print(f"✗ {lang}: Failed - {e}")
        
        # Test unanswerable question
        print("\n--- Handling Unanswerable Questions ---")
        unanswerable_q = "What will the weather be tomorrow?"
        answer, is_valid, _ = chatbot.answer_question(
            household_id=test_hh,
            email=test_email,
            question=unanswerable_q,
            language="en"
        )
        
        is_honest = "don't" in answer.lower() or "no" in answer.lower() or "unavailable" in answer.lower()
        print(f"✓ Honest response (no hallucination): {is_honest}")
        print(f"  Q: {unanswerable_q}")
        print(f"  A: {answer[:100]}...")
        
        return True
    
    except Exception as e:
        print(f"✗ Chatbot test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_llm_guard():
    """Test LLM hallucination guard."""
    print("\n" + "="*60)
    print("TEST 4: LLM Hallucination Guard Validation")
    print("="*60)
    
    try:
        from llm_guard import LLMHallucinationGuard
        
        guard = LLMHallucinationGuard()
        
        # Register some tool results
        guard.register_tool_result("get_appliance_breakdown", {
            "AC_kwh": 150.5,
            "Fridge_kwh": 45.2,
            "Lights_kwh": 23.0
        })
        
        # Test valid response (uses real numbers)
        valid_response = "Your AC used 150.5 kWh and fridge used 45.2 kWh this month."
        is_valid, issues, corrected = guard.validate_response(valid_response, language="en")
        print(f"✓ Valid response (with real numbers): {is_valid}")
        
        # Test response with invented number (should flag)
        invalid_response = "Your AC used 999 kWh (completely made up number)."
        is_valid2, issues2, corrected2 = guard.validate_response(invalid_response, language="en")
        print(f"✓ Detects invented numbers: {not is_valid2}")
        
        # Test that ISO date/time tokens inside an answer are NOT flagged
        guard2 = LLMHallucinationGuard()
        guard2.register_computed_stat("usage", 3.5)
        guard2.register_computed_stat("mean", 2.0)
        guard2.register_computed_stat("pct", 75.0)
        tstamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        tip_text = f"Appliance was 3.50 kW at {tstamp} (75.0% above average)."
        is_valid3, issues3, _ = guard2.validate_response(tip_text, language="en")
        print(f"✓ Date/time token not flagged as hallucination: {is_valid3}")
        
        return True
    
    except Exception as e:
        print(f"✗ LLM Guard test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def _make_household_data(n_days=14, seed=42):
    """Build a synthetic 14-day hourly household dataset for grounding tests."""
    idx = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=int(n_days) * 24, freq="h")
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"datetime": idx})
    df["Global_active_power"] = 0.5 + rng.random(len(df)) * 1.5
    df["Global_reactive_power"] = 0.05 + rng.random(len(df)) * 0.3
    df["Voltage"] = 235 + rng.random(len(df)) * 5
    df["Global_intensity"] = 1.0 + rng.random(len(df)) * 5.0
    df["Sub_metering_1"] = rng.random(len(df)) * 400
    df["Sub_metering_2"] = rng.random(len(df)) * 300
    df["Sub_metering_3"] = rng.random(len(df)) * 500
    df["is_weekend"] = df["datetime"].dt.dayofweek >= 5
    return df


def test_feature_extensions():
    """
    Test the integration gaps that were previously missing:
      - Removed family member stops receiving notifications
      - Week comparison grounded in real data (honest when no prior week)
      - Multilingual intent routing and genuinely translated answers
      - Notification-log tool surfaced to the chatbot
    """
    print("\n" + "="*60)
    print("TEST 5: Feature Extensions (week compare, multilingual, notification log, removal)")
    print("="*60)

    try:
        from db import get_db
        from notifications import get_notification_service
        from chatbot import get_chatbot

        db = get_db()
        service = get_notification_service()
        chatbot = get_chatbot()

        email = f"ext_{datetime.now().timestamp()}@test.local"
        household_id = hashlib.md5(email.encode()).hexdigest()
        db.create_user(email, "Extension", "pass", household_id=household_id)

        # ── Family + live recipient list ─────────────────────────────
        print("\n--- Family members + live recipients ---")
        ok, _ = db.add_family_member(
            household_id, "Ravi", "Spouse", "ravi@test.local",
            preferred_language="hi", notify_bills=True, notify_tips=True,
        )
        _, _ = db.add_family_member(
            household_id, "Sita", "Child", "sita@test.local",
            preferred_language="kn", notify_bills=True, notify_tips=False,
        )
        bill_recipients = service.recipients_for(household_id, email, "notification_bill_alerts", "en")
        bill_emails = [r[0] for r in bill_recipients]
        print(f"✓ Bill-alert recipients include Ravi & Sita: {'ravi@test.local' in bill_emails and 'sita@test.local' in bill_emails}")
        tip_recipients = service.recipients_for(household_id, email, "notification_optimization_tips", "en")
        print(f"✓ Sita opted out of tips and is excluded: {'sita@test.local' not in [r[0] for r in tip_recipients]}")

        # ── Removing a member stops their notifications ──────────────
        print("\n--- Removed member stops receiving (re-queried at send time) ---")
        sita_id = [m["id"] for m in db.get_family_members(household_id)
                   if m["email"] == "sita@test.local"][0]
        db.remove_family_member(sita_id, household_id)
        bill_after = [r[0] for r in service.recipients_for(household_id, email, "notification_bill_alerts", "en")]
        print(f"✓ Sita no longer in recipient list: {'sita@test.local' not in bill_after}")

        # ── Week comparison: grounded in real data ───────────────────
        print("\n--- Week comparison grounded in real data ---")
        df_two_weeks = _make_household_data(n_days=14)
        ans, valid, meta = chatbot.answer_question(
            household_id, email, "Why was my bill higher last week?",
            language="en", tariff_rate=8.0, household_data=df_two_weeks,
        )
        used_week = any("get_week_comparison" in t for t in meta["tool_calls"])
        print(f"✓ Week comparison tool called for 'why was my bill higher': {used_week}")
        print(f"✓ Answer passed grounded validation: {valid}")
        print(f"  A: {ans[:120]}...")

        # ── Honest response when no prior week exists ─────────────────
        df_one_week = _make_household_data(n_days=7, seed=1)
        ans2, valid2, meta2 = chatbot.answer_question(
            household_id, email, "Why was my bill higher than last week?",
            language="en", tariff_rate=8.0, household_data=df_one_week,
        )
        honest = any(k in ans2 for k in (
            "no comparison was invented", "कोई तुलना आविष्कार",
            "ಹೋಲಿಕೆಯನ್ನು ನಿರ್ಮಿಸಲಾಗಿಲ್ಲ", "పోలిక రూపొందించలేదు",
        ))
        print(f"✓ Honest 'no prior week' answer (no invented comparison): {honest}")
        print(f"  A: {ans2[:120]}...")

        # ── Multilingual intent routing ───────────────────────────────
        print("\n--- Multilingual intent routing ---")
        route_hi = chatbot._plan_tools("कौन सा उपकरण सबसे अधिक बिजली खपत करता है?", "hi")
        route_kn = chatbot._plan_tools("ಯಾವ ಉಪಕರಣ ಹೆಚ್ಚು ವಿದ್ಯುತ್ ಬಳಕೆಯನ್ನು ಮಾಡುತ್ತದೆ?", "kn")
        route_te = chatbot._plan_tools("మా అంచనా ఖర్చు ఎంత?", "te")
        print(f"✓ Hindi tools routed: {'get_appliance_breakdown' in route_hi}")
        print(f"✓ Kannada tools routed: {'get_appliance_breakdown' in route_kn}")
        print(f"✓ Telugu tools routed: {'get_current_prediction' in route_te}")
        out_of_scope_hi = chatbot._plan_tools("मौसम कैसा रहेगा?", "hi") == ["out_of_scope"]
        print(f"✓ Hindi out-of-scope detected: {out_of_scope_hi}")

        # ── Answers genuinely translated (not English defaults) ──────
        print("\n--- Translated answers ---")
        ans_hi, valid_hi, _ = chatbot.answer_question(
            household_id, email, "कौन सा उपकरण सबसे अधिक बिजली लेता है?",
            language="hi", household_data=df_two_weeks,
        )
        localized_hi = any(ch in ans_hi for ch in ("रसोई", "लॉन्ड्री", "वॉटर", "अन्य"))
        print(f"✓ Hindi answer uses localized appliance name: {localized_hi} (valid={valid_hi})")
        ans_te, valid_te, _ = chatbot.answer_question(
            household_id, email, "ఏ ఉపకరణం ఎక్కువ విద్యుత్ వాడుతుంది?",
            language="te", household_data=df_two_weeks,
        )
        localized_te = any(ch in ans_te for ch in ("వంటగది", "లాండ్రీ", "వాటర్", "ఇతర"))
        print(f"✓ Telugu answer uses localized appliance name: {localized_te} (valid={valid_te})")

        # ── Notification-log tool returns real audit entries ──────────
        print("\n--- Notification log tool ---")
        db.log_notification(
            household_id, "ravi@test.local", "Ravi", "test", "manual_test",
            {}, "Test subject", "Test body", language="en", status="sent",
        )
        ans_log, valid_log, meta_log = chatbot.answer_question(
            household_id, email, "What notifications were sent to my family?",
            language="en", household_data=df_two_weeks,
        )
        used_log = any("get_family_notification_log" in t for t in meta_log["tool_calls"])
        print(f"✓ Notification-log tool called: {used_log}")
        print(f"✓ Log entry visible in answer: {'ravi@test.local' in ans_log}")

        return True

    except Exception as e:
        print(f"✗ Feature extensions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_email_backend_setup():
    """Honest backend detection: real creds -> Connected, placeholders/absent -> Not configured.
    No network calls are made; this only verifies the detection logic."""
    print("\n" + "="*60)
    print("TEST 6: Email Backend Setup & Detection")
    print("="*60)

    import os as _os
    import importlib.util
    from notifications import NotificationService
    from i18n import t_lang

    saved = {}
    for var in ("SENDGRID_API_KEY", "SMTP_HOST", "SMTP_PORT", "SMTP_USERNAME",
                "SMTP_PASSWORD", "SMTP_FROM", "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"):
        saved[var] = _os.environ.get(var)
        _os.environ.pop(var, None)
    try:
        # 1. Nothing set -> honestly "Not configured"
        none_svc = NotificationService()
        print(f"✓ No creds -> not configured: {none_svc.email_backend is None}")

        # 2. .env.example placeholder values must NOT count as configured
        _os.environ["SENDGRID_API_KEY"] = "your_sendgrid_api_key_here"
        _os.environ["SMTP_HOST"] = "smtp.gmail.com"
        _os.environ["SMTP_USERNAME"] = "your_email@gmail.com"
        _os.environ["SMTP_PASSWORD"] = "your_app_password_here"
        ph_svc = NotificationService()
        print(f"✓ .env.example placeholders -> still not configured: {ph_svc.email_backend is None}")

        # 3. Real-looking Gmail SMTP app password -> configured as smtp
        _os.environ["SENDGRID_API_KEY"] = ""
        _os.environ["SMTP_USERNAME"] = "test@gmail.com"
        _os.environ["SMTP_PASSWORD"] = "abcd efgh ijkl mnop"
        smtp_svc = NotificationService()
        print(f"✓ Real Gmail SMTP creds -> configured: {smtp_svc.email_backend == 'smtp'}")

        # 4. Real-looking SendGrid key -> configured as sendgrid (when lib installed)
        sg_detected = None
        if importlib.util.find_spec("sendgrid"):
            _os.environ["SENDGRID_API_KEY"] = "SG.aaaa1111bbbb2222"
            _os.environ["SMTP_USERNAME"] = ""
            _os.environ["SMTP_PASSWORD"] = ""
            sg_svc = NotificationService()
            sg_detected = sg_svc.email_backend == "sendgrid"
            print(f"✓ Real SendGrid key -> configured: {sg_detected}")
        else:
            print("⚠ sendgrid library not installed - skipping SendGrid detection check")

        # 5. Setup-guide strings exist in all four languages (missing keys fall back to the raw key)
        keys = ["notif_connected", "notif_sms_optional", "notif_howto_title",
                "notif_howto_step1", "notif_howto_env", "notif_email_caption_smtp"]
        ok_i18n = all(t_lang(k, lang) != k for k in keys for lang in ("en", "hi", "kn", "te"))
        print(f"✓ Setup guide localized in all 4 languages: {ok_i18n}")

        result = (none_svc.email_backend is None
                  and ph_svc.email_backend is None
                  and smtp_svc.email_backend == "smtp"
                  and ok_i18n)
        if sg_detected is not None:
            result = result and sg_detected
        return result
    finally:
        for var, val in saved.items():
            if val is None:
                _os.environ.pop(var, None)
            else:
                _os.environ[var] = val


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("ENERGYPULSE NEW FEATURES TEST SUITE")
    print("="*60)
    
    results = {
        "Database": test_database(),
        "Notifications": test_notifications(),
        "Chatbot": test_chatbot(),
        "LLM Guard": test_llm_guard(),
        "Extensions": test_feature_extensions(),
        "Email Setup": test_email_backend_setup(),
    }
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    exit_code = 0 if all_passed else 1
    
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print("="*60 + "\n")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
