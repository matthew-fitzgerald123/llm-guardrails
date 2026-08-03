from __future__ import annotations
import re
from collections import defaultdict
from dataclasses import dataclass
from app.pii_scrubber import scrub as _scrub_pii

@dataclass
class FilterResult:
    original: str
    filtered: str
    flags: list[dict]
    blocked: bool
    block_reason: str

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

    scrub_result = _scrub_pii(filtered)
    if scrub_result.entities_found:
        entity_counts: dict[str, int] = defaultdict(int)
        for entity in scrub_result.entities_found:
            entity_counts[entity["type"]] += 1
        for entity_type, count in entity_counts.items():
            flags.append({"type": entity_type, "count": count})
        filtered = scrub_result.redacted

    return FilterResult(
        original=text,
        filtered=filtered,
        flags=flags,
        blocked=False,
        block_reason="",
    )
