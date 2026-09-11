from __future__ import annotations
import re
from dataclasses import dataclass
from app.pii_scrubber import scrub

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

    # Apply full PII scrubbing to the output (same rules as input pipeline)
    scrub_result = scrub(filtered)
    if scrub_result.entities_found:
        for entity in scrub_result.entities_found:
            flags.append({"type": f"pii_{entity['type']}_in_output", "count": 1})
        filtered = scrub_result.redacted

    return FilterResult(
        original=text,
        filtered=filtered,
        flags=flags,
        blocked=False,
        block_reason="",
    )
