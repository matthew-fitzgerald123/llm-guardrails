from __future__ import annotations
import re
from dataclasses import dataclass

@dataclass
class ScrubResult:
    original: str
    redacted: str
    entities_found: list[dict]

# Patterns ordered by specificity, more specific first
PII_PATTERNS = [
    {
        "name": "credit_card",
        "pattern": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "replacement": "[CREDIT_CARD]",
        "severity": "high",
    },
    {
        "name": "ssn",
        "pattern": r"\b\d{3}-\d{2}-\d{4}\b",
        "replacement": "[SSN]",
        "severity": "high",
    },
    {
        # IBAN: 2 letter country code, 2 check digits, then 11 to 30 alphanumerics
        # (optionally grouped in fours). Starts with letters, so it never collides
        # with the digit-leading card/ssn/phone patterns above.
        "name": "iban",
        "pattern": r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Za-z0-9]{4}){2,7}(?:[ ]?[A-Za-z0-9]{1,3})?\b",
        "replacement": "[IBAN]",
        "severity": "high",
    },
    {
        "name": "email",
        "pattern": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
        "replacement": "[EMAIL]",
        "severity": "medium",
    },
    {
        "name": "phone_us",
        "pattern": r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "replacement": "[PHONE]",
        "severity": "medium",
    },
    {
        "name": "ip_address",
        "pattern": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "replacement": "[IP_ADDRESS]",
        "severity": "low",
    },
    {
        "name": "date_of_birth",
        "pattern": r"\b(?:dob|date of birth|born)[:\s]+\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b",
        "replacement": "[DOB]",
        "severity": "high",
    },
    {
        "name": "api_key",
        "pattern": r"\b(?:sk|pk|api|key)[-_][A-Za-z0-9]{20,}\b",
        "replacement": "[API_KEY]",
        "severity": "high",
    },
]

def _audit_value(match: str, severity: str) -> str:
    """Masked representation of a detected entity for the audit log.

    High-severity entities (cards, SSNs, IBANs, API keys) are fully masked so no
    raw characters ever reach the audit store; a length hint is kept for triage.
    Lower-severity entities keep a short prefix preview to help debugging."""
    if severity == "high":
        return f"***({len(match)} chars)"
    return match[:4] + "***"


def scrub(text: str) -> ScrubResult:
    redacted = text
    found = []
    for p in PII_PATTERNS:
        matches = re.findall(p["pattern"], redacted, flags=re.IGNORECASE)
        if matches:
            for m in matches:
                found.append({
                    "type":     p["name"],
                    "severity": p["severity"],
                    "value":    _audit_value(m, p["severity"]),
                })
            redacted = re.sub(
                p["pattern"],
                p["replacement"],
                redacted,
                flags=re.IGNORECASE,
            )
    return ScrubResult(
        original=text,
        redacted=redacted,
        entities_found=found,
    )
