# LLM Guardrails

A security and safety proxy layer for LLM agent APIs. Sits in front of an upstream agent service and enforces rate limiting, prompt injection detection, PII scrubbing, input length caps, and output filtering before returning a response. Every request is audit logged to Postgres.

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

### Direct Checks

| Method | Path | Description |
|---|---|---|
| POST | `/check/injection` | Test a string for prompt injection patterns |
| POST | `/check/pii` | Detect and redact PII entities |
| POST | `/check/output` | Apply output filter to arbitrary text |

### Observability

| Method | Path | Description |
|---|---|---|
| GET | `/audit/logs` | Recent audit log entries |
| GET | `/audit/flagged` | Flagged requests by type and severity |
| GET | `/audit/stats` | Block rate, flag breakdown, avg latency |
| GET | `/health` | Server status + upstream URL |

Interactive docs at `http://localhost:8084/docs`.

## Guard Pipeline

Requests to `/guard/query` pass through these steps in order:

1. **Rate limiting** -- per `client_id` via Redis; 429 on limit exceeded
2. **Input length** -- rejects if word count exceeds `MAX_INPUT_TOKENS` (default 2048)
3. **Prompt injection detection** -- pattern-based; blocks high-severity matches, flags lower ones
4. **PII scrubbing** -- redacts detected entities (names, emails, SSNs, etc.) before forwarding
5. **Upstream forward** -- cleaned query sent to agent API via httpx
6. **Output filtering** -- scans response for disallowed content before returning
7. **Audit log** -- full request/response record written to Postgres

## Project Structure

```
app/
  injection_detector.py   pattern-based prompt injection classifier
  pii_scrubber.py         entity detection + redaction
  output_filter.py        response content filter
  rate_limiter.py         Redis-backed token bucket
  main.py                 FastAPI proxy app
  models.py               SQLAlchemy models (AuditLog, FlaggedRequest)
  database.py             engine + session
notebooks/
  demo.py                 demo: clean query, PII query, injection attempt
tests/
```
