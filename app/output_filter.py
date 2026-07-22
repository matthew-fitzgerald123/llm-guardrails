from __future__ import annotations
import re
from dataclasses import dataclass

@dataclass
class FilterResult:
    original: str
    filtered: str
    flags: list[dict]
    blocked: bool
    block_reason: str

# Patterns that should never appear in outputs
BLOCK_PATTERNS = [
    {
        "name": "system_prompt_leak",
        "pattern": r"(my\s+system\s+prompt\s+is|my\s+instructions\s+are|i\s+was\s+told\s+to)",
        "action": "block",
        "reason": "Potential system prompt leak",
    },
    {
        "name": "harmful_code",
        "pattern": r"(import\s+os\s*;\s*os\.system|subprocess\.call|eval\s*\(|exec\s*\()",
        "action": "block",
        "reason": "Potentially harmful code in output",
    },
]

# Patterns that get redacted from outputs
REDACT_PATTERNS = [
    {
        "name": "email_in_output",
        "pattern": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
        "replacement": "[EMAIL]",
    },
    {
        "name": "api_key_in_output",
        "pattern": r"\b(?:sk|pk|api|key)[-_][A-Za-z0-9]{20,}\b",
        "replacement": "[API_KEY]",
    },
    {
        "name": "credit_card_in_output",
        "pattern": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "replacement": "[CREDIT_CARD]",
    },
    {
        "name": "ssn_in_output",
        "pattern": r"\b\d{3}-\d{2}-\d{4}\b",
        "replacement": "[SSN]",
    },
    {
        "name": "phone_in_output",
        "pattern": r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "replacement": "[PHONE]",
    },
]

def filter_output(text: str) -> FilterResult:
    flags = []
    filtered = text

    # Check block patterns first
    for p in BLOCK_PATTERNS:
        if re.search(p["pattern"], filtered, flags=re.IGNORECASE):
            flags.append({"type": p["name"], "action": "block"})
            return FilterResult(
                original=text,
                filtered="[Response blocked by content filter]",
                flags=flags,
                blocked=True,
                block_reason=p["reason"],
            )

    # Apply redaction patterns
    for p in REDACT_PATTERNS:
        matches = re.findall(p["pattern"], filtered, flags=re.IGNORECASE)
        if matches:
            flags.append({"type": p["name"], "count": len(matches)})
            filtered = re.sub(
                p["pattern"],
                p["replacement"],
                filtered,
                flags=re.IGNORECASE,
            )

    return FilterResult(
        original=text,
        filtered=filtered,
        flags=flags,
        blocked=False,
        block_reason="",
    )
