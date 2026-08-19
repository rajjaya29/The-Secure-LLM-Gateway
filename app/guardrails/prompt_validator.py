"""Custom Prompt Validation Middleware / Component for Security Filtering."""

import re
from typing import Optional, List, Tuple
from pydantic import BaseModel, Field


class PromptValidationResult(BaseModel):
    is_valid: bool = Field(default=True, description="True if prompt passed validation")
    blocked: bool = Field(default=False, description="True if prompt should be rejected with HTTP 400")
    reason: Optional[str] = Field(default=None, description="Reason for rejection if blocked")
    threat_type: Optional[str] = Field(default=None, description="Category of threat detected")
    score: float = Field(default=0.0, description="Confidence score of injection heuristic")


class PromptValidator:
    """
    Prompt Validator detecting malicious prompt injections, jailbreaks,
    system prompt exfiltrations, and delimiter spoofing attempts.
    """

    PATTERNS: List[Tuple[str, str, str, float]] = [
        (
            r"(?i)\b(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+|any\s+|your\s+|the\s+|prior\s+|previous\s+|above\s+|former\s+)*(?:instructions?|directives?|rules?|constraints?|prompts?|guidelines?|filters?)\b",
            "Direct instruction override attempt",
            "instruction_override",
            0.95,
        ),
        (
            r"(?i)\b(?:you\s+are\s+now|enter|enable|activate)\s+(?:in\s+)?(?:DAN|developer|unfiltered|jailbreak|evil|god|root)\s+mode\b",
            "Jailbreak persona activation (DAN/Developer mode)",
            "jailbreak",
            0.90,
        ),
        (
            r"(?i)\b(?:do\s+anything\s+now|freedom\s+mode|unshackled\s+ai|jailbroken)\b",
            "Jailbreak slogan / persona exploit",
            "jailbreak",
            0.85,
        ),
        (
            r"(?i)(?:<\|im_start\|>system|<\|system\|>|<<SYS>>|\[INST\]|###\s*System:|<system_prompt>)",
            "System prompt delimiter spoofing",
            "delimiter_spoofing",
            0.92,
        ),
        (
            r"(?i)\b(?:print|display|reveal|dump|output|repeat|show)\s+(?:the\s+)?(?:system\s+prompt|initial\s+instructions?|system\s+message|secret\s+instructions?)\b",
            "System prompt exfiltration attempt",
            "prompt_exfiltration",
            0.85,
        ),
        (
            r"(?i)\b(?:verbatim|word-for-word|repeat\s+(?:the\s+)?(?:words?|text|everything)\s+(?:above|verbatim))\b",
            "Context window verbatim extraction attempt",
            "context_exfiltration",
            0.80,
        ),
        (
            r"(?i)\b(?:act\s+as|pretend\s+to\s+be)\s+(?:an?\s+)?(?:unaligned|unrestricted|unfiltered|evil|god\s+mode)\s+(?:AI|assistant|bot)\b",
            "Ethical constraint removal instruction",
            "jailbreak",
            0.88,
        ),
    ]

    def __init__(self, threshold: float = 0.70, enabled: bool = True):
        self.threshold = threshold
        self.enabled = enabled
        self._compiled = [(re.compile(p), desc, t_type, w) for p, desc, t_type, w in self.PATTERNS]

    def validate(self, text: str) -> PromptValidationResult:
        if not self.enabled or not text:
            return PromptValidationResult(is_valid=True, blocked=False, score=0.0)

        # Normalize whitespace
        cleaned = " ".join(text.strip().split())

        for pattern, desc, t_type, weight in self._compiled:
            if pattern.search(cleaned):
                if weight >= self.threshold:
                    return PromptValidationResult(
                        is_valid=False,
                        blocked=True,
                        reason=desc,
                        threat_type=t_type,
                        score=weight,
                    )

        return PromptValidationResult(is_valid=True, blocked=False, score=0.0)
