"""
Regression tests for chatbot intent routing and the hallucination guard.

Routing bugs are silent: the wrong tool still returns real numbers, so the
answer looks plausible while answering a question nobody asked. These cases pin
the specific misroutes that were found.
"""
import unittest

from chatbot import EnergyPulseChatbot
from llm_guard import LLMHallucinationGuard, admits_no_data


def plan(question, language="en"):
    return EnergyPulseChatbot._plan_tools(EnergyPulseChatbot, question, language)


class TestOutOfScope(unittest.TestCase):
    def test_unrelated_question_is_not_answered_with_energy_data(self):
        """This used to fall through to three tools and answer with figures."""
        for question in [
            "what is the capital of France",
            "who won the football match yesterday",
            "tell me a joke",
            "how do I boil an egg",
            "translate this to Spanish",
        ]:
            self.assertEqual(plan(question), ["out_of_scope"], question)

    def test_empty_question(self):
        self.assertEqual(plan(""), ["out_of_scope"])
        self.assertEqual(plan("   "), ["out_of_scope"])

    def test_explicit_out_of_scope_topics(self):
        self.assertEqual(plan("what is the weather today"), ["out_of_scope"])
        self.assertEqual(plan("tell me about the cricket match"), ["out_of_scope"])

    def test_greeting_does_not_invent_readings(self):
        self.assertEqual(plan("hello"), ["out_of_scope"])
        self.assertEqual(plan("hi there"), ["out_of_scope"])


class TestRouting(unittest.TestCase):
    def test_appliance_question(self):
        self.assertIn("get_appliance_breakdown",
                      plan("which appliance uses the most power"))

    def test_appliance_named_directly(self):
        for question in ["how much does the geyser use",
                         "is the washing machine expensive to run",
                         "tell me about the refrigerator",
                         "how much power does the AC use"]:
            self.assertIn("get_appliance_breakdown", plan(question), question)

    def test_forecast_question(self):
        self.assertIn("get_current_prediction",
                      plan("what will my bill be next month"))

    def test_usage_history(self):
        self.assertIn("get_usage_history", plan("how much did I use last week"))

    def test_optimization(self):
        self.assertIn("get_optimization_tips",
                      plan("how can I reduce my electricity bill"))

    def test_notification_log(self):
        self.assertIn("get_family_notification_log",
                      plan("show the notification log"))

    def test_comparison_uses_week_comparison(self):
        self.assertIn("get_week_comparison",
                      plan("why is my bill higher than last week"))


class TestFalsePositives(unittest.TestCase):
    def test_login_does_not_match_the_notification_log(self):
        """"log" is a substring of "login"; it must not pull in the log tool."""
        self.assertNotIn("get_family_notification_log", plan("help me login"))

    def test_useless_does_not_match_optimize(self):
        self.assertNotIn("get_optimization_tips", plan("this tool is useless"))

    def test_computer_is_an_appliance_but_the_word_is_not_the_reason(self):
        tools = plan("how much does my computer use")
        self.assertIn("get_appliance_breakdown", tools)

    def test_unused_does_not_match_usage_history(self):
        self.assertNotIn("get_usage_history", plan("the unused feature list"))

    def test_covid_is_out_of_scope(self):
        self.assertEqual(plan("covid vaccine side effects"), ["out_of_scope"])


class TestHindiRouting(unittest.TestCase):
    def test_hindi_reduces_bill(self):
        self.assertIn("get_optimization_tips",
                      plan("मैं बिजली का बिल कैसे घटाऊँ", "hi"))

    def test_hindi_appliance(self):
        self.assertIn("get_appliance_breakdown",
                      plan("वॉशिंग मशीन कितनी बिजली लेती है", "hi"))

    def test_hindi_unrelated(self):
        self.assertEqual(plan("फिल्म कौन सा है", "hi"), ["out_of_scope"])


class TestKannadaTeluguRouting(unittest.TestCase):
    def test_kannada_savings(self):
        self.assertIn("get_optimization_tips",
                      plan("ವಿದ್ಯುತ್ ಬಿಲ್ ಹೇಗೆ ಕಡಿಮೆ ಮಾಡುವುದು", "kn"))

    def test_telugu_savings(self):
        self.assertIn("get_optimization_tips",
                      plan("విద్యుత్ బిల్ ఎలా తగ్గించుకోవాలి", "te"))

    def test_telugu_unrelated(self):
        self.assertEqual(plan("సినిమా ఏది బాగుంది", "te"), ["out_of_scope"])


class TestGuardNumberExtraction(unittest.TestCase):
    def setUp(self):
        self.guard = LLMHallucinationGuard()

    def test_grounded_numbers_pass(self):
        self.guard.register_computed_stat("cost", 1234.56)
        valid, issues, _ = self.guard.validate_response(
            "That comes to Rs. 1234.56", language="en")
        self.assertTrue(valid, issues)

    def test_ungrounded_number_is_caught(self):
        self.guard.register_computed_stat("cost", 1234.56)
        valid, issues, _ = self.guard.validate_response(
            "That comes to Rs. 9999.99", language="en")
        self.assertFalse(valid)
        self.assertTrue(any("match" in i.lower() for i in issues))

    def test_rounded_grounded_number_passes(self):
        self.guard.register_computed_stat("cost", 1234.56)
        valid, issues, _ = self.guard.validate_response(
            "About Rs. 1235", language="en")
        self.assertTrue(valid, issues)

    def test_two_claims_sharing_a_window_are_both_checked(self):
        """Keys built on truncated context used to collapse into one entry."""
        self.guard.register_computed_stat("cost", 100.0)
        numbers = self.guard.extract_numbers_from_text(
            "Rs. 100 and Rs. 777 and Rs. 100")
        self.assertEqual(len(numbers), 3, numbers)

    def test_negative_claim_does_not_validate_against_positive(self):
        self.guard.register_computed_stat("drop", 12.0)
        valid, _, _ = self.guard.validate_response(
            "It went down 12 kWh", language="en")
        self.assertFalse(valid, "a decrease must not match an increase")

    def test_small_fabricated_amount_is_caught(self):
        """Rs. 5 was skipped because the check re-filtered on value >= 10."""
        self.guard.register_computed_stat("cost", 1234.56)
        valid, _, _ = self.guard.validate_response(
            "The extra fee is Rs. 5", language="en")
        self.assertFalse(valid)

    def test_dates_are_not_treated_as_claims(self):
        self.guard.register_computed_stat("cost", 1234.56)
        valid, issues, _ = self.guard.validate_response(
            "On 2025-09-10 the total was Rs. 1234.56", language="en")
        self.assertTrue(valid, issues)


class TestGuardNoData(unittest.TestCase):
    def setUp(self):
        self.guard = LLMHallucinationGuard()

    def test_english_no_data_reply_is_accepted(self):
        valid, issues, _ = self.guard.validate_response(
            "I don't have that information available.", language="en")
        self.assertTrue(valid, issues)

    def test_hindi_no_data_reply_is_accepted(self):
        """An English-only phrase list flagged every correct Hindi reply."""
        valid, issues, _ = self.guard.validate_response(
            "मेरे पास यह जानकारी उपलब्ध नहीं है।", language="hi")
        self.assertTrue(valid, issues)

    def test_kannada_no_data_reply_is_accepted(self):
        valid, issues, _ = self.guard.validate_response(
            "ನನ್ನಲ್ಲಿ ಈ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ.", language="kn")
        self.assertTrue(valid, issues)

    def test_telugu_no_data_reply_is_accepted(self):
        valid, issues, _ = self.guard.validate_response(
            "నా వద్ద ఈ సమాచారం లేదు.", language="te")
        self.assertTrue(valid, issues)

    def test_confident_claim_without_grounding_is_flagged(self):
        valid, issues, _ = self.guard.validate_response(
            "You used 480 kWh last week.", language="en")
        self.assertFalse(valid)
        self.assertTrue(issues)

    def test_marker_detection(self):
        self.assertTrue(admits_no_data("no data for that", "en"))
        self.assertFalse(admits_no_data("you used 480 kWh", "en"))
        self.assertTrue(admits_no_data("ಡೇಟಾ ಇಲ್ಲ", "kn"))
        self.assertFalse(admits_no_data("480 kWh ಬಳಸಿದ್ದಾರೆ", "kn"))


class TestEndToEndAnswering(unittest.TestCase):
    """The out-of-scope guard only works if answer_question checks the plan."""

    @classmethod
    def setUpClass(cls):
        import os
        import tempfile

        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._cwd = os.getcwd()
        os.chdir(cls._tmpdir.name)
        import db as db_module
        db_module.DB_PATH = os.path.join(cls._tmpdir.name, "chat_test.db")
        import importlib
        import chatbot as chatbot_module
        importlib.reload(db_module)
        importlib.reload(chatbot_module)
        cls.bot = chatbot_module.EnergyPulseChatbot()

    @classmethod
    def tearDownClass(cls):
        import os
        os.chdir(cls._cwd)
        cls._tmpdir.cleanup()

    def test_out_of_scope_question_is_refused_in_every_language(self):
        cases = {
            "en": "what is the capital of France",
            "hi": "फिल्म कौन सा है",
            "kn": "ಸಿನಿಮಾ ಯಾವುದು ಚೆನ್ನದು",
            "te": "సినిమా ఏది బాగుంది",
        }
        for language, question in cases.items():
            answer, is_grounded, meta = self.bot.answer_question(
                "test-house", "test@example.com", question, language=language)
            self.assertEqual(meta["tool_calls"], ["out_of_scope"],
                             f"{language}: {answer}")
            self.assertEqual(meta["tool_results"], {}, language)
            self.assertTrue(is_grounded, language)
            # Each refusal must come from that language's catalogue, not the
            # English string, which used to be the fallback for all of them.
            if language != "en":
                self.assertNotIn("i can only answer", answer.lower(), language)
            self.assertIn(language, ("en", "hi", "kn", "te"))

    def test_out_of_scope_refusal_does_not_leak_numbers(self):
        answer, _, _ = self.bot.answer_question(
            "test-house", "test@example.com",
            "what is the capital of France", language="en")
        self.assertNotIn("kWh", answer)

    def test_empty_question_is_refused(self):
        answer, _, meta = self.bot.answer_question(
            "test-house", "test@example.com", "", language="en")
        self.assertEqual(meta["tool_calls"], ["out_of_scope"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
