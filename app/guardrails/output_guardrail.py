"""Output guardrail for LLM response verification and leak prevention."""

import re
from typing import Tuple, List, Optional
from app.guardrails.pii_scrubber import PIIScrubber


class OutputGuardrail:
    """
    Verifies upstream LLM responses before caching and delivering to the client.
    Prevents leakage of internal system instructions, developer prompts, and accidental secret disclosure.
    """

    SYSTEM_LEAK_PATTERNS = [
        r"(?i)\b(?:my|the)\s+(?:system\s+(?:prompt|instructions?|directive|message)|initial\s+instructions?)\s+(?:is|are|was|were|states?):",
        r"(?i)\bhere\s+(?:is|are)\s+the\s+(?:system\s+(?:prompt|instructions?|message)|hidden\s+developer\s+prompt):?",
        r"(?i)<\|im_start\|>system",
        r"(?i)INTERNAL_GATEWAY_GUARDRAIL",
    ]

    SECRET_PATTERNS = [
        r"\b(?:sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,}|AIza[0-9A-Za-z-_]{35})\b",
    ]

    def __init__(self, enable_leak_prevention: bool = True, pii_scrubber: Optional[PIIScrubber] = None):
        self.enable_leak_prevention = enable_leak_prevention
        self.pii_scrubber = pii_scrubber or PIIScrubber()
        self._leak_regexes = [re.compile(p) for p in self.SYSTEM_LEAK_PATTERNS]
        self._secret_regexes = [re.compile(p) for p in self.SECRET_PATTERNS]

    def verify_and_clean(self, output_text: str) -> Tuple[str, bool, List[str]]:
        if not output_text:
            return output_text, True, []

        violations: List[str] = []
        cleaned_text = output_text

        if self.enable_leak_prevention:
            for regex in self._leak_regexes:
                if regex.search(cleaned_text):
                    violations.append("System prompt disclosure pattern detected in response")
                    cleaned_text = regex.sub("[REDACTED_SYSTEM_DIRECTIVE]", cleaned_text)

        for regex in self._secret_regexes:
            if regex.search(cleaned_text):
                violations.append("Exposed API key / token detected in response")
                cleaned_text = regex.sub("[REDACTED_SECRET_KEY]", cleaned_text)

        is_safe = len(violations) == 0
        return cleaned_text, is_safe, violations
