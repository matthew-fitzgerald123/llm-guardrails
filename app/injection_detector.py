from __future__ import annotations
import re
from dataclasses import dataclass
from app.semantic_detector import semantic_check

@dataclass
class DetectionResult:
    is_injection: bool
    confidence: float
    matched_patterns: list[str]
    severity: str
    semantic_score: float = 0.0

# Injection patterns — ordered from high to low severity
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

    for p in INJECTION_PATTERNS:
        if re.search(p["pattern"], text, flags=re.IGNORECASE):
            matched.append(p["name"])
            total_weight += p["weight"]

    # Semantic second layer — catches paraphrased injections that evade regex
    sem = semantic_check(text)
    if sem.available and sem.is_suspicious and "semantic_similarity" not in matched:
        matched.append("semantic_similarity")
        total_weight += 0.8

    # Normalise confidence — cap at 1.0
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
