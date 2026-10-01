from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass
from app.semantic_detector import semantic_check

# Zero-width and invisible Unicode characters that can be inserted between
# letters to break regex patterns without changing the visible text.
_INVISIBLE_CHARS = frozenset([
    "​",  # ZERO WIDTH SPACE
    "‌",  # ZERO WIDTH NON-JOINER
    "‍",  # ZERO WIDTH JOINER
    "‎",  # LEFT-TO-RIGHT MARK
    "‏",  # RIGHT-TO-LEFT MARK
    "﻿",  # ZERO WIDTH NO-BREAK SPACE (BOM)
    "­",  # SOFT HYPHEN
    "⁠",  # WORD JOINER
    "᠎",  # MONGOLIAN VOWEL SEPARATOR
])

# Cyrillic and Greek characters that are visually indistinguishable from
# their Latin equivalents, used to craft homoglyph-based evasion payloads.
_CONFUSABLE_MAP = str.maketrans({
    # Cyrillic lowercase
    "а": "a",  # а → a
    "е": "e",  # е → e
    "о": "o",  # о → o
    "р": "p",  # р → p
    "с": "c",  # с → c
    "х": "x",  # х → x
    "у": "y",  # у → y
    "і": "i",  # і → i
    # Cyrillic uppercase
    "А": "A",  # А → A
    "Е": "E",  # Е → E
    "О": "O",  # О → O
    "Р": "P",  # Р → P
    "С": "C",  # С → C
    "Х": "X",  # Х → X
    "У": "Y",  # У → Y
    "І": "I",  # І → I
    # Greek lookalikes
    "α": "a",  # α → a
    "ε": "e",  # ε → e
    "ο": "o",  # ο → o
    "ρ": "p",  # ρ → p
    "ν": "v",  # ν → v
})


def _strip_invisible(text: str) -> str:
    """Remove zero-width and invisible Unicode characters from text."""
    return "".join(c for c in text if c not in _INVISIBLE_CHARS)


def _normalize_confusables(text: str) -> str:
    """Return text with Unicode confusables mapped to ASCII equivalents.

    Applies NFKC normalization first (handles fullwidth chars and ligatures),
    then substitutes Cyrillic/Greek homoglyphs.  Used to detect evasion
    attempts that replace ASCII letters with visually identical Unicode chars.
    """
    return unicodedata.normalize("NFKC", text).translate(_CONFUSABLE_MAP)


def _has_invisible_chars(text: str) -> bool:
    """Return True if text contains any zero-width or invisible Unicode chars."""
    return any(c in _INVISIBLE_CHARS for c in text)

@dataclass
class DetectionResult:
    is_injection: bool
    confidence: float
    matched_patterns: list[str]
    severity: str
    semantic_score: float = 0.0

# Injection patterns, ordered from high to low severity
INJECTION_PATTERNS = [
    {
        "name": "ignore_instructions",
        "pattern": r"ignore\s+(all\s+)?(previous|prior|above|preceding)\s+instructions?",
        "severity": "high",
        "weight": 1.0,
    },
    {
        "name": "system_override",
        "pattern": r"(new\s+)?system\s+prompt|override\s+(system|instructions?)|forget\s+(your\s+)?(instructions?|training)",
        "severity": "high",
        "weight": 1.0,
    },
    {
        "name": "role_hijack",
        "pattern": r"you\s+are\s+now\s+(?!an?\s+AI|an?\s+assistant)|pretend\s+(you\s+are|to\s+be)\s+(?!helpful|an?\s+AI)",
        "severity": "high",
        "weight": 0.9,
    },
    {
        "name": "jailbreak_dan",
        "pattern": r"\bDAN\b|do\s+anything\s+now|jailbreak|unrestricted\s+mode",
        "severity": "high",
        "weight": 1.0,
    },
    {
        "name": "prompt_leak",
        "pattern": r"(print|reveal|show|output|repeat|tell me)\s+(your\s+)?(system\s+prompt|instructions?|prompt)",
        "severity": "medium",
        "weight": 0.8,
    },
    {
        "name": "delimiter_injection",
        "pattern": r"(<\|.*?\|>|###\s*(system|instruction|input|output)|---\s*(system|end)\s*---)",
        "severity": "medium",
        "weight": 0.7,
    },
    {
        "name": "indirect_injection",
        "pattern": r"when\s+you\s+(see|read|process)\s+this|hidden\s+instruction|invisible\s+text",
        "severity": "medium",
        "weight": 0.7,
    },
    {
        "name": "encoding_evasion",
        "pattern": r"base64|rot13|hex\s+encoded|unicode\s+escape",
        "severity": "medium",
        "weight": 0.6,
    },
]

BLOCK_THRESHOLD = 0.7
FLAG_THRESHOLD  = 0.4

def detect(text: str) -> DetectionResult:
    matched = []
    total_weight = 0.0
    stripped = _strip_invisible(text)
    normalized = _normalize_confusables(stripped)
    found_unicode_evasion = False
    found_zero_width_evasion = False

    for p in INJECTION_PATTERNS:
        orig_hit    = bool(re.search(p["pattern"], text,       flags=re.IGNORECASE))
        stripped_hit = bool(re.search(p["pattern"], stripped,   flags=re.IGNORECASE))
        norm_hit    = bool(re.search(p["pattern"], normalized, flags=re.IGNORECASE))
        if orig_hit or norm_hit:
            matched.append(p["name"])
            total_weight += p["weight"]
            if not orig_hit:
                # Pattern only fires after some normalization step
                if stripped_hit:
                    found_zero_width_evasion = True
                else:
                    found_unicode_evasion = True

    if found_unicode_evasion:
        matched.append("unicode_evasion")
        total_weight += 0.5

    if found_zero_width_evasion:
        matched.append("zero_width_evasion")
        total_weight += 0.5

    # Semantic second layer: catches paraphrased injections that evade regex
    sem = semantic_check(text)
    if sem.available and sem.is_suspicious and "semantic_similarity" not in matched:
        matched.append("semantic_similarity")
        total_weight += 0.8

    # Normalise confidence, cap at 1.0
    confidence = min(total_weight, 1.0)

    if confidence >= BLOCK_THRESHOLD:
        severity = "high"
    elif confidence >= FLAG_THRESHOLD:
        severity = "medium"
    else:
        severity = "low"

    return DetectionResult(
        is_injection=confidence >= BLOCK_THRESHOLD,
        confidence=round(confidence, 4),
        matched_patterns=matched,
        severity=severity,
        semantic_score=sem.score,
    )
