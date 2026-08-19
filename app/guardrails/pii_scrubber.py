"""PII Scrubbing and anonymization engine."""

import re
from typing import Dict, List, Tuple, Any, Optional


class PIIScrubber:
    """
    Scrubs sensitive Personally Identifiable Information (PII) and secrets
    from input payloads prior to forwarding to upstream LLM providers.
    """

    PATTERNS: Dict[str, Tuple[str, str]] = {
        "EMAIL": (
            r"(?i)\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            "Email Address",
        ),
        "PHONE": (
            r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            "Phone Number",
        ),
        "SSN": (
            r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b",
            "Social Security Number (SSN)",
        ),
        "CREDIT_CARD": (
            r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|6(?:011|5[0-9][0-9])[0-9]{12}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|(?:2131|1800|35\d{3})\d{11}|(?:[0-9]{4}[-\s]){3}[0-9]{4})\b",
            "Credit Card Number",
        ),
        "IPV4": (
            r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
            "IPv4 Address",
        ),
        "API_KEY": (
            r"\b(?:sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,}|AIza[0-9A-Za-z-_]{35}|eyJ[a-zA-Z0-9_\-]{20,}\.[a-zA-Z0-9_\-]{20,}\.[a-zA-Z0-9_\-]+)\b",
            "API Key / Secret Token / JWT",
        ),
    }

    def __init__(self, mask_style: str = "tokenized"):
        self.mask_style = mask_style
        self._compiled_patterns = {
            entity_type: (re.compile(pattern), desc)
            for entity_type, (pattern, desc) in self.PATTERNS.items()
        }

    def scrub(self, text: str, mask_style: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]], Dict[str, str]]:
        if not text:
            return text, [], {}

        style = mask_style or self.mask_style
        detected_entities: List[Dict[str, Any]] = []
        token_mapping: Dict[str, str] = {}
        type_counters: Dict[str, int] = {}
        sanitized_text = text

        for entity_type, (regex, desc) in self._compiled_patterns.items():
            matches = list(regex.finditer(sanitized_text))
            if not matches:
                continue

            for match in reversed(matches):
                original_val = match.group(0)
                start, end = match.span()

                if entity_type == "CREDIT_CARD":
                    digits = re.sub(r"\D", "", original_val)
                    if len(digits) < 13 or len(digits) > 19:
                        continue

                type_counters[entity_type] = type_counters.get(entity_type, 0) + 1
                counter = type_counters[entity_type]

                if style == "tokenized":
                    token = f"<{entity_type}_{counter}>"
                else:
                    token = f"[REDACTED_{entity_type}]"

                token_mapping[token] = original_val
                detected_entities.append({
                    "type": entity_type,
                    "description": desc,
                    "original": original_val,
                    "token": token,
                    "start": start,
                    "end": end,
                })

                sanitized_text = sanitized_text[:start] + token + sanitized_text[end:]

        detected_entities.reverse()
        return sanitized_text, detected_entities, token_mapping

    def restore(self, sanitized_text: str, token_mapping: Dict[str, str]) -> str:
        restored = sanitized_text
        for token, original in token_mapping.items():
            restored = restored.replace(token, original)
        return restored
