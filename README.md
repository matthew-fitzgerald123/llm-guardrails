# LLM Guardrails

A security and safety proxy layer for LLM agent APIs. Sits in front of an upstream agent service and enforces replay protection, rate limiting, prompt injection detection (pattern-based and semantic), PII scrubbing, input length caps, and output filtering before returning a response. Every request is audit logged to Postgres.

## Stack

| Component | Library |
|---|---|
| API | FastAPI + uvicorn (port 8084) |
| Rate limiting | Redis (token bucket per client_id) |
| Upstream agent | project_05 (port 8083) via httpx |
| Persistence | PostgreSQL + SQLAlchemy |

## Setup

```bash
# Create database
createdb llm_guardrails

# Install dependencies
pip install -r requirements.txt

# Start Redis (if not already running)
brew services start redis

# Set upstream URL in .env (default: http://localhost:8083)
# UPSTREAM_URL=http://localhost:8083
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://localhost/llm_guardrails` | Postgres connection |
| `UPSTREAM_URL` | `http://localhost:8083` | Upstream agent API to forward guarded requests to |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection for rate limiting and replay protection |
| `MAX_INPUT_TOKENS` | `2048` | Max word count before a request is rejected |
| `RATE_LIMIT_RPM` | `60` | Default requests-per-minute limit |
| `RATE_LIMIT_TIERS` | _(unset)_ | Per-tier RPM overrides |
| `RATE_LIMIT_DEFAULT_TIER` | `free` | Tier applied when a request doesn't specify one |
| `NONCE_TTL_SECONDS` | `300` | Replay-protection window for request nonces |
| `SEMANTIC_INJECTION_THRESHOLD` | `0.75` | Cosine similarity threshold for semantic injection detection |
| `INJECTION_BLOCK_THRESHOLD` | `0.7` | Cumulative pattern-weight score above which a request is blocked |
| `INJECTION_FLAG_THRESHOLD` | `0.4` | Cumulative pattern-weight score above which a request is flagged |

## Running

```bash
# Start API server
make serve

# Run end-to-end demo (requires upstream agent running)
make demo

# Run tests
make test
```

## API Endpoints

### Guarded Query

| Method | Path | Description |
|---|---|---|
| POST | `/guard/query` | Run request through all guards, forward to upstream |
| GET | `/tiers` | List rate-limit tiers and their RPM limits |

### Direct Checks

| Method | Path | Description |
|---|---|---|
| POST | `/check/injection` | Test a string for prompt injection patterns (pattern + semantic score) |
| POST | `/check/pii` | Detect and redact PII entities |
| POST | `/check/output` | Apply output filter to arbitrary text |

### Observability

| Method | Path | Description |
|---|---|---|
| GET | `/audit/logs` | Recent audit log entries |
| GET | `/audit/flagged` | Flagged requests by type and severity |
| GET | `/audit/stats` | Block rate, flag breakdown, avg latency |
| GET | `/audit/dashboard` | Combined stats + recent flagged requests for a single-page view |
| GET | `/health` | Server status + upstream URL |

Interactive docs at `http://localhost:8084/docs`.

## Guard Pipeline

Requests to `/guard/query` pass through these steps in order:

1. **Replay protection** -- if a `nonce` is provided, rejects it with a 409 if already seen within `NONCE_TTL_SECONDS`
2. **Rate limiting** -- per `client_id` and `tier` via Redis; 429 on limit exceeded
3. **Input length** -- rejects if word count exceeds `MAX_INPUT_TOKENS` (default 2048)
4. **Prompt injection detection** -- pattern-based plus embedding similarity against known injection phrases; blocks high-severity matches, flags lower ones
5. **PII scrubbing** -- redacts detected entities (emails, SSNs, credit cards, IBANs, phone numbers, IP addresses, API keys, etc.) before forwarding; high-severity entities are fully masked in the audit log so no raw characters are persisted
6. **Upstream forward** -- cleaned query sent to agent API via httpx
7. **Output filtering** -- scans response for disallowed content before returning
8. **Audit log** -- full request/response record written to Postgres

## Project Structure

```
app/
  injection_detector.py   pattern-based + semantic prompt injection classifier
  semantic_detector.py    embedding similarity check against known injection phrases
  pii_scrubber.py         entity detection + redaction
  output_filter.py        response content filter
  rate_limiter.py         Redis-backed token bucket, per-tier limits
  replay_protector.py     Redis-backed nonce replay detection
  main.py                 FastAPI proxy app
  models.py               SQLAlchemy models (AuditLog, FlaggedRequest)
  database.py             engine + session
notebooks/
  demo.py                 demo: clean query, PII query, injection attempt
tests/
  test_guardrails.py      guard pipeline and endpoint tests
```
