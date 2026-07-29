from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Request, Header
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Any, Optional
import httpx, uuid, time, os
from dotenv import load_dotenv

from app.database import get_db, engine
from app.models import Base, AuditLog, FlaggedRequest
from app.pii_scrubber import scrub
from app.injection_detector import detect
from app.output_filter import filter_output
from app.rate_limiter import rate_limiter
from app.replay_protector import replay_protector

load_dotenv()

UPSTREAM_URL = os.getenv("UPSTREAM_URL", "http://localhost:8083")
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", 2048))

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield

app = FastAPI(title="LLM Guardrails", version="1.0.0", lifespan=lifespan)

# ── Proxy endpoint ─────────────────────────────────────────

class GuardedRequest(BaseModel):
    query: str
    client_id: str = "anonymous"
    tier: str = "free"
    nonce: Optional[str] = None
    max_steps: int = 6
    bypass_cache: bool = False

@app.post("/guard/query", tags=["guardrails"])
async def guarded_query(req: GuardedRequest, db: Session = Depends(get_db)):
    request_id = str(uuid.uuid4())[:12]
    t_start = time.time()
    flags = []

    # ── 1. Replay protection ───────────────────────────────
    if req.nonce and replay_protector.is_replay(req.nonce):
        _log_flag(db, request_id, req.client_id,
                  "replay_detected", "high", req.nonce[:12])
        _log_blocked(db, request_id, req.client_id, req.query,
                     "replay_detected", [])
        raise HTTPException(409, "Duplicate request: nonce already seen")

    # ── 2. Rate limiting ──────────────────────────────────
    rl = rate_limiter.check(req.client_id, tier=req.tier)
    if not rl.allowed:
        _log_blocked(db, request_id, req.client_id, req.query,
                     "rate_limit_exceeded", flags)
        raise HTTPException(
            status_code=429,
            detail={
                "error":        "Rate limit exceeded",
                "limit":        rl.limit,
                "reset_in_seconds": rl.reset_in_seconds,
            }
        )

    # ── 2. Input length check ──────────────────────────────
    if len(req.query.split()) > MAX_INPUT_TOKENS:
        word_count = len(req.query.split())
        _log_flag(db, request_id, req.client_id,
                  "input_too_long", "medium", f"{word_count} words")
        _log_blocked(db, request_id, req.client_id, req.query,
                     "input_too_long", flags)
        raise HTTPException(400, "Input exceeds maximum token limit")

    # ── 3. Prompt injection detection ─────────────────────
    injection = detect(req.query)
    if injection.matched_patterns:
        flags.append({
            "type":     "injection_detected",
            "patterns": injection.matched_patterns,
            "severity": injection.severity,
        })
        _log_flag(db, request_id, req.client_id,
                  "prompt_injection", injection.severity,
                  str(injection.matched_patterns))

    if injection.is_injection:
        _log_blocked(db, request_id, req.client_id, req.query,
                     "prompt_injection_blocked", flags)
        raise HTTPException(400, {
            "error":    "Request blocked: potential prompt injection detected",
            "patterns": injection.matched_patterns,
        })

    # ── 4. PII scrubbing ───────────────────────────────────
    scrub_result = scrub(req.query)
    if scrub_result.entities_found:
        flags.append({
            "type":     "pii_scrubbed",
            "entities": scrub_result.entities_found,
        })
        for entity in scrub_result.entities_found:
            _log_flag(db, request_id, req.client_id,
                      f"pii_{entity['type']}", entity["severity"],
                      entity["type"])

    clean_query = scrub_result.redacted

    # ── 5. Forward to upstream ─────────────────────────────
    upstream_response = None
    raw_output = ""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{UPSTREAM_URL}/agent/run",
                json={"query": clean_query, "max_steps": req.max_steps},
            )
            if r.status_code == 200:
                upstream_response = r.json()
                raw_output = upstream_response.get("final_answer", "")
            else:
                raw_output = f"Upstream error: {r.status_code}"
    except httpx.ConnectError:
        raw_output = "Upstream service unavailable"

    # ── 6. Output filtering ────────────────────────────────
    filter_result = filter_output(raw_output)
    if filter_result.flags:
        flags.extend([{"type": "output_" + f["type"]} for f in filter_result.flags])
    if filter_result.blocked:
        _log_flag(db, request_id, req.client_id,
                  "output_blocked", "high", filter_result.block_reason)

    # ── 7. Audit log ───────────────────────────────────────
    latency = round((time.time() - t_start) * 1000, 2)
    log = AuditLog(
        request_id=request_id,
        client_id=req.client_id,
        input_text=req.query[:500],
        output_text=filter_result.filtered[:500],
        input_redacted=clean_query[:500] if scrub_result.entities_found else None,
        blocked=filter_result.blocked,
        block_reason=filter_result.block_reason or None,
        flags=flags,
        latency_ms=latency,
    )
    db.add(log)
    db.commit()

    return {
        "request_id": request_id,
        "answer":     filter_result.filtered,
        "flags":      flags,
        "blocked":    filter_result.blocked,
        "meta": {
            "pii_scrubbed":         bool(scrub_result.entities_found),
            "injection_score":      injection.confidence,
            "rate_limit_remaining": rl.remaining,
            "rate_limit_tier":      rl.tier,
            "latency_ms":           latency,
        },
    }


@app.get("/tiers", tags=["guardrails"])
def list_tiers():
    from app.rate_limiter import _TIERS, _DEFAULT_TIER, _DEFAULT_RPM
    tiers = dict(_TIERS) if _TIERS else {"default": _DEFAULT_RPM}
    return {"tiers": tiers, "default_tier": _DEFAULT_TIER}

# ── Direct check endpoints ─────────────────────────────────

class CheckReq(BaseModel):
    text: str

@app.post("/check/injection", tags=["checks"])
def check_injection(req: CheckReq):
    result = detect(req.text)
    return {
        "is_injection":      result.is_injection,
        "confidence":        result.confidence,
        "matched_patterns":  result.matched_patterns,
        "severity":          result.severity,
        "semantic_score":    result.semantic_score,
    }

@app.post("/check/pii", tags=["checks"])
def check_pii(req: CheckReq):
    result = scrub(req.text)
    return {
        "entities_found": result.entities_found,
        "redacted":       result.redacted,
        "has_pii":        bool(result.entities_found),
    }

@app.post("/check/output", tags=["checks"])
def check_output(req: CheckReq):
    result = filter_output(req.text)
    return {
        "blocked":      result.blocked,
        "block_reason": result.block_reason,
        "flags":        result.flags,
        "filtered":     result.filtered,
    }

# ── Observability ─────────────────────────────────────────

@app.get("/audit/logs", tags=["observability"])
def audit_logs(limit: int = 20, db: Session = Depends(get_db)):
    logs = (
        db.query(AuditLog)
        .order_by(AuditLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "request_id": l.request_id,
            "client_id":  l.client_id,
            "blocked":    l.blocked,
            "flags":      l.flags,
            "latency_ms": l.latency_ms,
            "created_at": str(l.created_at),
        }
        for l in logs
    ]

@app.get("/audit/flagged", tags=["observability"])
def flagged_requests(limit: int = 20, db: Session = Depends(get_db)):
    flags = (
        db.query(FlaggedRequest)
        .order_by(FlaggedRequest.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "request_id": f.request_id,
            "client_id":  f.client_id,
            "flag_type":  f.flag_type,
            "severity":   f.severity,
            "detail":     f.detail,
            "created_at": str(f.created_at),
        }
        for f in flags
    ]

@app.get("/audit/stats", tags=["observability"])
def audit_stats(hours: Optional[int] = None, db: Session = Depends(get_db)):
    from datetime import datetime, timedelta
    from sqlalchemy import func, case

    al_q = db.query(AuditLog)
    fr_q = db.query(FlaggedRequest)
    if hours is not None:
        since = datetime.utcnow() - timedelta(hours=hours)
        al_q = al_q.filter(AuditLog.created_at >= since)
        fr_q = fr_q.filter(FlaggedRequest.created_at >= since)

    row = al_q.with_entities(
        func.count(AuditLog.id),
        func.sum(case((AuditLog.blocked == True, 1), else_=0)),  # noqa: E712
        func.avg(AuditLog.latency_ms),
    ).first()

    total = int(row[0] or 0)
    if total == 0:
        return {"message": "No requests logged yet"}

    blocked = int(row[1] or 0)
    avg_latency = round(float(row[2] or 0.0), 2)

    flag_rows = fr_q.with_entities(
        FlaggedRequest.flag_type,
        func.count(FlaggedRequest.id),
    ).group_by(FlaggedRequest.flag_type).all()
    flag_breakdown = {ft: int(cnt) for ft, cnt in flag_rows}

    flagged = int(
        fr_q.with_entities(func.count(func.distinct(FlaggedRequest.request_id))).scalar() or 0
    )

    result = {
        "total_requests": total,
        "blocked":        blocked,
        "flagged":        flagged,
        "block_rate":     round(blocked / total, 4),
        "avg_latency_ms": avg_latency,
        "flag_breakdown": flag_breakdown,
    }
    if hours is not None:
        result["window_hours"] = hours
    return result

@app.get("/audit/dashboard", tags=["observability"])
def audit_dashboard(hours: int = 24, bucket_minutes: int = 60, flagged_limit: int = 20, db: Session = Depends(get_db)):
    from datetime import datetime, timedelta
    from collections import defaultdict
    from sqlalchemy import func, case

    since = datetime.utcnow() - timedelta(hours=hours)

    # Overall stats via SQL aggregation
    row = (
        db.query(AuditLog)
        .filter(AuditLog.created_at >= since)
        .with_entities(
            func.count(AuditLog.id),
            func.sum(case((AuditLog.blocked == True, 1), else_=0)),  # noqa: E712
            func.avg(AuditLog.latency_ms),
        )
        .first()
    )
    total = int(row[0] or 0)
    blocked_count = int(row[1] or 0)
    avg_latency = round(float(row[2] or 0.0), 2)

    flag_rows = (
        db.query(FlaggedRequest)
        .filter(FlaggedRequest.created_at >= since)
        .with_entities(FlaggedRequest.flag_type, func.count(FlaggedRequest.id))
        .group_by(FlaggedRequest.flag_type)
        .all()
    )
    flag_breakdown = {ft: int(cnt) for ft, cnt in flag_rows}

    flagged_count = int(
        db.query(func.count(func.distinct(FlaggedRequest.request_id)))
        .filter(FlaggedRequest.created_at >= since)
        .scalar() or 0
    )

    stats = {
        "total_requests": total,
        "blocked":        blocked_count,
        "flagged":        flagged_count,
        "block_rate":     round(blocked_count / total, 4) if total else 0.0,
        "avg_latency_ms": avg_latency,
        "flag_breakdown": flag_breakdown,
    }

    # Timeline bucketing via in-memory pass (only loads created_at + blocked + flags)
    logs = (
        db.query(AuditLog.created_at, AuditLog.blocked, AuditLog.flags)
        .filter(AuditLog.created_at >= since)
        .all()
    )
    buckets: dict[str, dict] = defaultdict(lambda: {
        "total": 0, "blocked": 0, "flagged": 0, "flag_types": defaultdict(int)
    })
    for ts, is_blocked, flags in logs:
        bucket_key = ts.strftime("%Y-%m-%dT%H:") + f"{(ts.minute // bucket_minutes) * bucket_minutes:02d}"
        b = buckets[bucket_key]
        b["total"] += 1
        if is_blocked:
            b["blocked"] += 1
        if flags:
            b["flagged"] += 1
            for f in flags:
                b["flag_types"][f.get("type", "unknown")] += 1

    timeline = []
    for bucket_key in sorted(buckets):
        b = buckets[bucket_key]
        bt = b["total"]
        timeline.append({
            "bucket":     bucket_key,
            "total":      bt,
            "blocked":    b["blocked"],
            "flagged":    b["flagged"],
            "block_rate": round(b["blocked"] / bt, 4) if bt else 0.0,
            "flag_types": dict(b["flag_types"]),
        })

    # Recent flagged requests
    recent_flagged = [
        {
            "request_id": f.request_id,
            "client_id":  f.client_id,
            "flag_type":  f.flag_type,
            "severity":   f.severity,
            "detail":     f.detail,
            "created_at": str(f.created_at),
        }
        for f in (
            db.query(FlaggedRequest)
            .filter(FlaggedRequest.created_at >= since)
            .order_by(FlaggedRequest.created_at.desc())
            .limit(flagged_limit)
            .all()
        )
    ]

    return {
        "window_hours":   hours,
        "bucket_minutes": bucket_minutes,
        "stats":          stats,
        "recent_flagged": recent_flagged,
        "timeline":       timeline,
    }


@app.get("/health")
def health():
    return {"status": "ok", "upstream": UPSTREAM_URL}

# ── Helpers ────────────────────────────────────────────────

def _log_blocked(db, request_id, client_id, query, reason, flags):
    log = AuditLog(
        request_id=request_id,
        client_id=client_id,
        input_text=query[:500],
        blocked=True,
        block_reason=reason,
        flags=flags,
    )
    db.add(log)
    db.commit()

def _log_flag(db, request_id, client_id, flag_type, severity, detail):
    flag = FlaggedRequest(
        request_id=request_id,
        client_id=client_id,
        flag_type=flag_type,
        severity=severity,
        detail=detail,
    )
    db.add(flag)
    db.commit()
