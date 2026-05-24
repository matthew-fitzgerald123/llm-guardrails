from __future__ import annotations
from sqlalchemy import Column, String, DateTime, JSON, Integer, Text, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    request_id    = Column(String, unique=True, nullable=False, index=True)
    client_id     = Column(String, nullable=False, index=True)
    input_text    = Column(Text, nullable=False)
    output_text   = Column(Text, nullable=True)
    input_redacted = Column(Text, nullable=True)
    blocked       = Column(Boolean, default=False)
    block_reason  = Column(String, nullable=True)
    flags         = Column(JSON, default=[])
    latency_ms    = Column(Float, nullable=True)
    created_at    = Column(DateTime, default=datetime.utcnow, index=True)

class FlaggedRequest(Base):
    __tablename__ = "flagged_requests"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    request_id  = Column(String, nullable=False, index=True)
    client_id   = Column(String, nullable=False, index=True)
    flag_type   = Column(String, nullable=False, index=True)
    severity    = Column(String, nullable=False)
    detail      = Column(Text, nullable=True)
    created_at  = Column(DateTime, default=datetime.utcnow, index=True)
