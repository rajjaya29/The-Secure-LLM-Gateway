"""Prompt injection and jailbreak detection engine."""

import re
import time
from typing import List, Tuple, Dict, Any, Optional
from app.schemas.gateway import GuardrailResult, GuardrailThreat


class InjectionDetector:
    """
    Advanced heuristic and pattern-based prompt injection / jailbreak detector.
    Evaluates inputs against signature patterns, system delimiter abuses,
    and adversarial jailbreak vectors.
    """

    HIGH_SEVERITY_PATTERNS = [
        (
            r"(?i)\b(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+)?(?:previous|prior|above|former)\s+(?:instructions?|directives?|rules?|constraints?|prompts?)\b",
            "Direct instruction override attempt",
            0.95,
        ),
        (
            r"(?i)\b(?:you\s+are\s+now|enter|enable|activate)\s+(?:in\s+)?(?:DAN|developer|unfiltered|jailbreak|evil|god|root)\s+mode\b",
            "Jailbreak persona activation (DAN/Developer mode)",
            0.90,
        ),
        (
            r"(?i)\b(?:do\s+anything\s+now|freedom\s+mode|unshackled\s+ai|jailbroken)\b",
            "Jailbreak slogan / persona exploit",
            0.85,
        ),
        (
            r"(?i)(?:<\|im_start\|>system|<\|system\|>|<<SYS>>|\[INST\]|###\s*System:|<system_prompt>)",
            "System prompt delimiter spoofing",
            0.92,
        ),
        (
            r"(?i)\b(?:print|display|reveal|dump|output|repeat|show)\s+(?:the\s+)?(?:system\s+prompt|initial\s+instructions?|system\s+message|secret\s+instructions?)\b",
            "System prompt exfiltration attempt",
            0.85,
        ),
        (
            r"(?i)\b(?:verbatim|word-for-word)\s+(?:repeat|output|show)\s+(?:everything|all\s+text)\s+(?:above|from\s+the\s+beginning)\b",
            "Context window verbatim extraction attempt",
            0.80,
        ),
        (
            r"(?i)\bact\s+as\s+(?:an?\s+)?unaligned\s+AI\s+without\s+(?:any\s+)?(?:ethics|morals|rules|filters)\b",
            "Ethical constraint removal instruction",
            0.88,
        ),
    ]

    MEDIUM_SEVERITY_PATTERNS = [
        (
            r"(?i)\b(?:from\s+now\s+on|starting\s+now|henceforth)\s+you\s+must\s+(?:answer|obey|respond)\b",
            "Behavioral override framing",
            0.50,
        ),
        (
            r"(?i)\b(?:pretend|hypothetically)\s+(?:you\s+have\s+no\s+(?:rules|filters|limits)|you\s+can\s+say\s+anything)\b",
            "Hypothetical bypass framing",
            0.60,
        ),
        (
            r"(?i)\b(?:base64|rot13|hex|reverse)\s+(?:decode|encoded?)\s+(?:the\s+following|this\s+prompt)\b",
            "Obfuscated encoding instruction bypass",
            0.45,
        ),
        (
            r"(?i)```(?:system|admin|root|instruction)",
            "Suspicious code-block markdown impersonating system role",
            0.65,
        ),
        (
            r"(?i)\bnevermind\s+(?:what\s+was\s+said\s+before|the\s+previous\s+prompt)\b",
            "Soft instruction disregard",
            0.40,
        ),
    ]

    def __init__(self, confidence_threshold: float = 0.70, block_on_detection: bool = True):
        self.confidence_threshold = confidence_threshold
        self.block_on_detection = block_on_detection
        
        self._compiled_high = [
            (re.compile(pattern), desc, score) for pattern, desc, score in self.HIGH_SEVERITY_PATTERNS
        ]
        self._compiled_med = [
            (re.compile(pattern), desc, score) for pattern, desc, score in self.MEDIUM_SEVERITY_PATTERNS
        ]

    def detect(self, text: str) -> GuardrailResult:
        start_time = time.perf_counter()
        threats: List[GuardrailThreat] = []
        max_score = 0.0

        if not text or not text.strip():
            return GuardrailResult(
                is_safe=True,
                blocked=False,
                action_taken="allow",
                injection_score=0.0,
                threats=[],
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        for regex, desc, weight in self._compiled_high:
            match = regex.search(text)
            if match:
                threats.append(
                    GuardrailThreat(
                        category="prompt_injection_high",
                        confidence=weight,
                        matched_pattern=match.group(0),
                        description=desc,
                    )
                )
                if weight > max_score:
                    max_score = weight

        for regex, desc, weight in self._compiled_med:
            match = regex.search(text)
            if match:
                threats.append(
                    GuardrailThreat(
                        category="adversarial_cue_medium",
                        confidence=weight,
                        matched_pattern=match.group(0),
                        description=desc,
                    )
                )
                max_score = max_score + weight * (1.0 - max_score)

        system_role_markers = len(re.findall(r"(?i)\b(?:system|assistant|user)\s*:", text))
        if system_role_markers >= 3 and max_score > 0.3:
            max_score = min(1.0, max_score + 0.20)
            threats.append(
                GuardrailThreat(
                    category="structural_role_spoofing",
                    confidence=0.75,
                    matched_pattern=f"Found {system_role_markers} role markers",
                    description="Multiple artificial role delimiters detected in user message",
                )
            )

        is_safe = max_score < self.confidence_threshold
        blocked = (not is_safe) and self.block_on_detection
        action = "block" if blocked else ("flag" if not is_safe else "allow")

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return GuardrailResult(
            is_safe=is_safe,
            blocked=blocked,
            action_taken=action,
            injection_score=round(max_score, 4),
            threats=threats,
            processing_time_ms=round(elapsed_ms, 3),
        )
