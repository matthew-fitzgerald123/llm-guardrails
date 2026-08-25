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
        # Covers os.system, os.popen, os.execv*/execl*; subprocess.call/run/Popen;
        # eval/exec/compile builtins; __import__ dynamic import; shell=True keyword
        "pattern": (
            r"(?:"
            r"import\s+os\s*[;,\n]?\s*os\s*\.\s*(?:system|popen|execv[pe]?|execl[pe]?)\s*\("
            r"|os\s*\.\s*(?:system|popen|execv[pe]?|execl[pe]?)\s*\("
            r"|subprocess\s*\.\s*(?:call|run|Popen|check_output|check_call)\s*\("
            r"|eval\s*\("
            r"|exec\s*\("
            r"|compile\s*\(.*\beval\b"
            r"|__import__\s*\("
            r"|shell\s*=\s*True"
            r")"
        ),
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
