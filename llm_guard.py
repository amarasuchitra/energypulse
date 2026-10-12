"""
LLM Hallucination Guard & Validation System
=============================================
Prevents AI-generated content from stating unverified facts, numbers, or claims.
Ensures all claims are grounded in actual tool results and real computed data.

Core principles:
  1. All factual claims must come from tool results or computed statistics
  2. If data is unavailable, the LLM must explicitly state so (not guess)
  3. Every numeric claim is validated against actual tool outputs
  4. Fallback behavior with made-up defaults is strictly prohibited
  5. Full language-aware error reporting for all four supported languages

Usage:
    guard = LLMHallucinationGuard()
    
    # Register real data from tool calls
    guard.register_tool_result("check_slot_risk", {"slot_id": "A1", "risk_percent": 15})
    guard.register_tool_result("calculate_incentive", {"incentive": 250})
    
    # Validate LLM response
    llm_response = "Risk is 15% and incentive is Rs. 250"
    is_valid, issues, corrected = guard.validate_response(llm_response, language="en")
"""

import re
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# Simplified translation for guard messages (in production, use i18n.T())
GUARD_MESSAGES = {
    "en": {
        "validation_error": "Response contains unverified claims",
        "missing_data_instruction": "For missing data, explicitly state: 'I don't have this information available.'",
        "hallucination_detected": "Detected {n} ungrounded numeric claims",
        "numbers_not_matching": "Numbers don't match tool results: {details}",
        "unknown_data_claim": "Claims knowledge of {field} not available in tools",
        "regenerate_response": "Response will be regenerated with stricter grounding",
    },
    "hi": {
        "validation_error": "प्रतिक्रिया में अयोग्य दावे हैं",
        "missing_data_instruction": "लापता डेटा के लिए स्पष्ट रूप से कहें: 'मेरे पास यह जानकारी उपलब्ध नहीं है।'",
        "hallucination_detected": "{n} अप्रमाणित संख्यात्मक दावों का पता चला",
        "numbers_not_matching": "संख्याएं उपकरण परिणामों से मेल नहीं खाती: {details}",
        "unknown_data_claim": "{field} का ज्ञान उपलब्ध नहीं दावा करता है",
        "regenerate_response": "प्रतिक्रिया को सख्त आधार के साथ पुनः उत्पन्न किया जाएगा",
    },
    "kn": {
        "validation_error": "ಪ್ರತಿಕ್ರಿಯೆ ಅಯೋಗ್ಯ ಹೇಳಿಕೆಗಳನ್ನು ಹೊಂದಿದೆ",
        "missing_data_instruction": "ಲಾಪ್ತ ಡೇಟಾಗೆ, ಸ್ಪಷ್ಟವಾಗಿ ಹೇಳಿ: 'ನನ್ನಲ್ಲಿ ಈ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ.'",
        "hallucination_detected": "{n} ಆಧಾರಿತವಲ್ಲದ ಸಂಖ್ಯೆಯ ದಾವೆ ಕಂಡುಬಂದಿದೆ",
        "numbers_not_matching": "ಸಂಖ್ಯೆಗಳು ಸರಂಜಾಮದ ಫಲಿತಾಂಶಗಳೊಂದಿಗೆ ಹೊಂದಿಕೆ ಹೋಗುತ್ತವೆ ಅಲ್ಲ: {details}",
        "unknown_data_claim": "{field} ಸಾಧನಗಳಲ್ಲಿ ಲಭ್ಯವಿಲ್ಲ ಎಂಬ ಜ್ಞಾನವನ್ನು ದಾವೆ ಮಾಡುತ್ತದೆ",
        "regenerate_response": "ಪ್ರತಿಕ್ರಿಯೆಯನ್ನು ಕಠಿಣ ನೆಲೆಯೊಂದಿಗೆ ಪುನಃ ಉತ್ಪಾದಿಸಲಾಗುವುದು",
    },
    "te": {
        "validation_error": "ప్రతిక్రియ అసంపూర్ణ దావాలను కలిగి ఉంది",
        "missing_data_instruction": "లేని డేటా కోసం, స్పష్టంగా పేర్కొనండి: 'నా వద్ద ఈ సమాచారం లేదు.'",
        "hallucination_detected": "{n} ఆధారహీన సంఖ్యా దావాలు కనుగొనబడ్డాయి",
        "numbers_not_matching": "సంఖ్యలు సాధన ఫలితాలకు సరిపోవు: {details}",
        "unknown_data_claim": "సాధనాలలో లేని {field} యొక్క జ్ఞానాన్ని దావా చేస్తుంది",
        "regenerate_response": "ఎక్కువ కఠోర ఆధారంతో ప్రతిక్రియ పునర్నిర్మాణం చేయబడుతుంది",
    },
}

# Short markers that mean "this data is not available", per language. Kept here
# rather than in i18n.py because they are guard configuration, not UI copy, and
# the guard is imported by tools that should not depend on the Streamlit app.
_NO_DATA_PHRASES = {
    "en": ["don't have", "do not have", "not available", "no data", "unable to",
           "cannot determine", "can't determine", "insufficient", "missing",
           "not enough", "could not", "cannot be"],
    "hi": ["उपलब्ध नहीं", "मेरे पास नहीं", "कोई डेटा नहीं", "पर्याप्त नहीं",
           "नहीं पता", "ज्ञात नहीं", "संभव नहीं"],
    "kn": ["ಲಭ್ಯವಿಲ್ಲ", "ನನ್ನಲ್ಲಿ ಇಲ್ಲ", "ಡೇಟಾ ಇಲ್ಲ", "ಸಾಕಷ್ಟು ಇಲ್ಲ",
           "ಗೊತ್ತಿಲ್ಲ", "ತಿಳಿಯುವುದಿಲ್ಲ"],
    "te": ["సమాచారం లేదు", "డేటా లేదు", "సరಿಪೋಲಿಲ್ಲ", "తెలಯದು",
           "అందుబాటులో లేదు", "లేదు"],
}
# Telugu "లేదు" is a single common negator, so it is matched as its own token
# rather than as a bare substring, which would fire on any word containing it.
_TE_NO_DATA_TOKEN = re.compile(r"^[^\w]*" + re.escape("లేదు") + r"[^\w]*$")


def no_data_phrases(language: str = "en") -> List[str]:
    """Markers of an honest "no data" reply in the response's own language."""
    return _NO_DATA_PHRASES.get(language, _NO_DATA_PHRASES["en"])


def admits_no_data(response: str, language: str = "en") -> bool:
    """True when the response says, in its own language, that data is missing."""
    if not response:
        return False
    lower = response.lower()
    for phrase in no_data_phrases(language):
        if language == "te" and phrase == "లేదు":
            if any(_TE_NO_DATA_TOKEN.match(token) for token in lower.split()):
                return True
            continue
        if phrase in lower:
            return True
    return False


class LLMHallucinationGuard:
    """
    Validates LLM responses to prevent hallucinations and ensure all claims
    are grounded in real tool results and computed data.
    """
    
    def __init__(self):
        """Initialize the guard system."""
        self.tool_results: Dict[str, Any] = {}
        self.extracted_numbers: Dict[str, float] = {}
        self.validation_log: List[Dict] = []
    
    def register_tool_result(self, tool_name: str, result: Any) -> None:
        """
        Register a tool result (e.g., from check_slot_risk, calculate_incentive).
        
        Parameters
        ----------
        tool_name : str
            Name of the tool (e.g. 'check_slot_risk', 'calculate_incentive')
        result : Any
            The actual result from the tool (dict, scalar, etc.)
        """
        self.tool_results[tool_name] = result
        
        # Extract all numbers from the result for validation
        if isinstance(result, dict):
            for key, val in result.items():
                if isinstance(val, (int, float)):
                    self.extracted_numbers[f"{tool_name}_{key}"] = float(val)
        elif isinstance(result, (int, float)):
            self.extracted_numbers[tool_name] = float(result)
    
    def register_computed_stat(self, stat_name: str, value: float) -> None:
        """
        Register a computed statistic (e.g., daily_cost, monthly_savings).
        
        Parameters
        ----------
        stat_name : str
            Name of the statistic
        value : float
            The actual computed value
        """
        self.extracted_numbers[stat_name] = float(value)
    
    def extract_numbers_from_text(self, text: str) -> Dict[str, float]:
        """
        Extract all numbers from text (with context).
        Returns a dict mapping approximate context to extracted number.

        Keys are keyed on the match offset, which is unique. Keying on the
        surrounding text instead silently dropped numbers whenever two claims
        shared a 30-character window, letting an ungrounded value through.
        
        Parameters
        ----------
        text : str
            Text to extract numbers from
        
        Returns
        -------
        Dict[str, float]
            Mapping of {description: number_value}
        """
        numbers = {}
        if not text:
            return numbers
        
        # Skip numbers that are parts of ISO date/time tokens (e.g. "2025-09-10 14:00"
        # or "09/10/2025 02:30 PM"). These come verbatim from tool data and are never
        # invented numeric claims, so they must not be flagged as hallucinations.
        datetime_token_re = re.compile(
            r"\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)?"
            r"|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}(?:[\sT]\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)?"
        )
        cleaned_text = datetime_token_re.sub(" ", text)

        # A leading minus, or a nearby "decrease" cue, makes a claim negative.
        # Ignoring the sign let "down 12 kWh" validate against a grounded 12.
        negative_cue = re.compile(
            r"(?:decreas\w*|drop\w*|fall\w*|fell|down|lower\w*|reduc\w*|sav\w*"
            r"|less|lesser|below|under|घट|कम|ಕಡಿಮೆ|తగ్గ)\s*"
            r"(?:by|of|about|around|approximately)?\s*:?\s*$",
            re.IGNORECASE,
        )
        
        def sign_at(index: int) -> float:
            prefix = cleaned_text[max(0, index - 40):index]
            if prefix.rstrip().endswith("-"):
                return -1.0
            if negative_cue.search(prefix):
                return -1.0
            return 1.0

        def record(prefix: str, start: int, end: int, raw: str) -> None:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                return
            context = cleaned_text[max(0, start - 20):end + 20]
            numbers[f"{prefix}@{start}_{context[:30]}"] = sign_at(start) * value

        # Pattern for "Rs. XXX" or "₹ XXX" or "Rs XXX"
        rs_pattern = r'(?:Rs\.?|₹)\s*([0-9,]+(?:\.\d{1,2})?)'
        for match in re.finditer(rs_pattern, cleaned_text, re.IGNORECASE):
            num_str = match.group(1).replace(',', '')
            if not num_str:
                continue
            context = cleaned_text[max(0, match.start() - 20):match.end() + 20]
            numbers[f"rs@{match.start()}_{context[:30]}"] = (
                sign_at(match.start()) * float(num_str))
        # Consume those figures so the plain-number pass does not also pick them
        # up: "Rs. 100" used to be extracted twice and checked twice. The span is
        # blanked with equal-length padding so later match offsets, and therefore
        # the sign lookups, still line up with the original text.
        def blank(_match):
            return " " * len(_match.group(0))

        consumed = re.sub(rs_pattern, blank, cleaned_text, flags=re.IGNORECASE)

        # Pattern for "XX %" or "XX%"
        pct_pattern = r'(\d+(?:\.\d{1,2})?)\s*%'
        for match in re.finditer(pct_pattern, consumed):
            record("pct", match.start(), match.end(), match.group(1))
        consumed = re.sub(pct_pattern, blank, consumed)
        
        # Pattern for plain numbers >= 10 (assume significant claims).
        # The lookbehind/lookahead stop the matcher from splitting off the
        # fractional part of a decimal (e.g. "3.50 kW" must not yield "50").
        num_pattern = r'(?<![.\d])(\d{2,}(?:\.\d+)?)(?![\d.])\s*(?:kWh|units?|days?|hours?|months?|years?|people|persons?)?'
        for match in re.finditer(num_pattern, consumed):
            if float(match.group(1)) >= 10:
                record("num", match.start(), match.end(), match.group(1))
        
        return numbers
    
    def validate_response(self, response: str, language: str = "en") -> Tuple[bool, List[str], str]:
        """
        Validate an LLM response against registered tool results and stats.
        
        Parameters
        ----------
        response : str
            The LLM's response to validate
        language : str
            Language code ('en', 'hi', 'kn', 'te')
        
        Returns
        -------
        Tuple[is_valid, issues, potentially_corrected_response]
            - is_valid: True if response contains only grounded claims
            - issues: List of validation issues found
            - corrected: Suggested correction or original if valid
        """
        issues = []
        msgs = GUARD_MESSAGES.get(language, GUARD_MESSAGES["en"])
        
        # Extract numbers from response
        response_numbers = self.extract_numbers_from_text(response)
        
        # Check if response admits missing data appropriately.
        # Grounding evidence = any registered tool result or computed stat.
        # If nothing was grounded yet, the response must say data is unavailable.
        # The phrases have to match the response's language: this used to test an
        # English-only list against Hindi, Kannada and Telugu answers, so every
        # correct "I don't have that data" reply in those languages was flagged.
        if self.extracted_numbers == {} and self.tool_results == {} and not admits_no_data(
            response, language
        ):
            issues.append(msgs["missing_data_instruction"])
        
        # Validate each extracted number against known values
        unmatched_numbers = []
        for desc, response_val in response_numbers.items():
            found_match = False
            
            # Check against known statistics (allow small tolerance for rounding)
            for stat_name, stat_val in self.extracted_numbers.items():
                if abs(response_val - stat_val) < 0.01 * max(abs(response_val), abs(stat_val), 1):
                    found_match = True
                    break
            
            # Also accept a value that is the rounded form of a grounded one,
            # which is how a kW figure legitimately becomes a whole number.
            # The sign is kept here on purpose: comparing magnitudes would let
            # "went down 12 kWh" validate against a grounded increase of 12.
            if not found_match:
                for stat_name, stat_val in self.extracted_numbers.items():
                    if abs(round(stat_val) - response_val) < 0.51:
                        found_match = True
                        break
            
            # The extractor already decided which numbers are significant, so
            # every one of them is checked. Re-filtering on ">= 10" here let a
            # fabricated "Rs. 5" through.
            if not found_match:
                unmatched_numbers.append((desc, response_val))
        
        if unmatched_numbers:
            details = "; ".join([f"{d}={v}" for d, v in unmatched_numbers[:3]])
            issues.append(msgs["numbers_not_matching"].format(details=details))
        
        # Log validation result
        self.validation_log.append({
            "timestamp": datetime.now().isoformat(),
            "response": response,
            "is_valid": len(issues) == 0,
            "issues": issues,
            "language": language,
        })
        
        # Generate corrected version if issues found
        corrected = response
        if issues:
            note = msgs["validation_error"]
            if any("missing_data_instruction" in issue for issue in issues):
                note += " " + msgs["missing_data_instruction"]
            corrected = f"{response}\n\n⚠️ Guard Note: {note}"
        
        return len(issues) == 0, issues, corrected
    
    def generate_system_prompt_for_language(self, language: str = "en") -> str:
        """
        Generate a system prompt that instructs the LLM to avoid hallucinations.
        Should be prepended to every LLM call in this language.
        
        Parameters
        ----------
        language : str
            Language code ('en', 'hi', 'kn', 'te')
        
        Returns
        -------
        str
            System prompt in the requested language
        """
        prompts = {
            "en": (
                "You are a helpful energy advisor. CRITICAL RULE: Only state facts, numbers, "
                "or recommendations that come DIRECTLY from the tool results provided to you. "
                "NEVER estimate, guess, or infer a plausible-sounding number. "
                "If information is not available from a tool result, you MUST clearly state: "
                "'I don't have this information available.' "
                "Before stating any number tied to a tool's purpose (delivery time, risk %, incentive, cost, etc.), "
                "you MUST have called that tool in this conversation. Do not make assumptions."
            ),
            "hi": (
                "आप एक सहायक ऊर्जा सलाहकार हैं। गंभीर नियम: केवल वे तथ्य, संख्याएँ, "
                "या सिफारिशें बताएँ जो आपको प्रदान की गई उपकरण परिणामों से सीधे आती हैं। "
                "कभी भी अनुमान, अनुमान, या संभावित-लगने वाली संख्या का अनुमान न लगाएँ। "
                "यदि कोई जानकारी उपकरण परिणाम से उपलब्ध नहीं है, तो आपको स्पष्ट रूप से कहना चाहिए: "
                "'मेरे पास यह जानकारी उपलब्ध नहीं है।' "
                "किसी भी संख्या को बताने से पहले जो किसी उपकरण के उद्देश्य से संबंधित है (समय, जोखिम %, प्रोत्साहन, लागत, आदि), "
                "आपने इस बातचीत में वह उपकरण कॉल किया होना चाहिए। धारणाएँ न बनाएँ।"
            ),
            "kn": (
                "ನೀವು ಸಹಾಯಕ ಶಕ್ತಿ ಸಲಹೆದಾತ ಗಳು. ಗಂಭೀರ ನಿಯಮ: ಕೇವಲ ಅವ್ಯವಹಾರಿಕ, ಸಂಖ್ಯೆಗಳು, "
                "ಅಥವಾ ನೀಡಿದ ಸಾಧನ ಫಲಿತಾಂಶಗಳಿಂದ ನೇರವಾಗಿ ಬರುವ ಶಿಫಾರಸುಗಳನ್ನು ಹೇಳಿ. "
                "ಎಂದಿಗೂ ಅಂದಾಜುಗಾರಿ, ಊಹೆ ಅಥವಾ ಸಂಭವನೀಯ-ಸುಂದರವಾದ ಸಂಖ್ಯೆಯನ್ನು ಅನುಮಾನಿಸಬೇಡಿ. "
                "ಸಾಧನ ಫಲಿತಾಂಶದಿಂದ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲದಿದ್ದರೆ, ನೀವು ಸ್ಪಷ್ಟವಾಗಿ ಹೇಳಬೇಕು: "
                "'ನನ್ನಲ್ಲಿ ಈ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ.' "
                "ಸಾಧನದ ಉದ್ದೇಶ್ಯಕ್ಕೆ ಸಂಬಂಧಿಸಿದ ಯಾವುದೇ ಸಂಖ್ಯೆಯನ್ನು ಹೇಳುವ ಮೊದಲು (ಸಮಯ, ಝೋಖಿಮೆ %, ಪ್ರೋತ್ಸಾಹನ, ವೆಚ್ಚ, ಇತ್ಯಾದಿ), "
                "ನೀವು ಈ ಸಂಭಾಷಣೆಯಲ್ಲಿ ಆ ಸಾಧನವನ್ನು ಕರೆದಿರಬೇಕು. ಭಾವನೆಗಳನ್ನು ಮಾಡಬೇಡಿ."
            ),
            "te": (
                "మీరు ఉపయోగకరమైన శక్తి సలహాదారు. విమర్శనీయ నియమం: మీకు అందించిన సాధన ఫలితాల నుండి నేరుగా వచ్చే "
                "సంగతులు, సంఖ్యలు, లేదా సిఫారసులను మాత్రమే చెప్పండి. "
                "ఎప్పుడూ అంచనా, ఊహ, లేదా సంభావ్య-సౌందర్య సంఖ్యను ఊహించవద్దు. "
                "సాధన ఫలితం నుండి సమాచారం లభ్యం కాకపోతే, మీరు స్పష్టంగా చెప్పాలి: "
                "'నా వద్ద ఈ సమాచారం లేదు.' "
                "సాధన యొక్క ఉద్దేశ్యానికి సంబంధించిన ఏదైనా సంఖ్యను చెప్పే ముందు (సమయం, ఝోఖిమ %, ప్రేరణ, ఖర్చు, మొదలైనవి), "
                "మీరు ఈ సంభాషణలో ఆ సాధనను పిలిచి ఉండాలి. ఊహలను చేయవద్దు."
            ),
        }
        
        return prompts.get(language, prompts["en"])
    
    def get_validation_report(self) -> Dict[str, Any]:
        """
        Get a summary report of all validations performed.
        
        Returns
        -------
        Dict with validation statistics and issues
        """
        valid_count = sum(1 for log in self.validation_log if log["is_valid"])
        total_count = len(self.validation_log)
        
        return {
            "total_responses_validated": total_count,
            "valid_responses": valid_count,
            "invalid_responses": total_count - valid_count,
            "validation_rate": valid_count / total_count if total_count > 0 else 0,
            "registered_stats": len(self.extracted_numbers),
            "recent_issues": [log["issues"] for log in self.validation_log[-5:] if log["issues"]],
        }
