"""
SQLAlchemy ORM models for the transaction processing pipeline.

Three tables following the assignment's suggested data model:
  - Job          : tracks each uploaded CSV and its processing status
  - Transaction  : stores every cleaned row with anomaly flags and LLM results
  - JobSummary   : holds the LLM-generated narrative and aggregated stats
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    Text,
    DateTime,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import relationship

from app.database import Base


# ---------------------------------------------------------------------------
# Job – one row per uploaded CSV file
# ---------------------------------------------------------------------------
class Job(Base):
    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    filename = Column(String(255), nullable=False)
    status = Column(
        String(20),
        nullable=False,
        default="pending",
        index=True,
    )  # pending | processing | completed | failed
    row_count_raw = Column(Integer, nullable=True)
    row_count_clean = Column(Integer, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    completed_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)

    # Relationships
    transactions = relationship(
        "Transaction",
        back_populates="job",
        cascade="all, delete-orphan",
    )
    summary = relationship(
        "JobSummary",
        back_populates="job",
        uselist=False,           # one-to-one
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<Job id={self.id} file={self.filename!r} status={self.status!r}>"


# ---------------------------------------------------------------------------
# Transaction – one row per cleaned CSV row
# ---------------------------------------------------------------------------
class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    job_id = Column(
        String(36),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    txn_id = Column(String(50), nullable=True)          # can be blank in CSV
    date = Column(String(30), nullable=True)             # ISO 8601 string after cleaning
    merchant = Column(String(255), nullable=True)
    amount = Column(Float, nullable=True)
    currency = Column(String(10), nullable=True)         # INR | USD (uppercased)
    status = Column(String(20), nullable=True)           # SUCCESS | FAILED | PENDING
    category = Column(String(100), nullable=True)        # original from CSV or 'Uncategorised'
    account_id = Column(String(50), nullable=True)

    # Anomaly flags
    is_anomaly = Column(Boolean, nullable=False, default=False)
    anomaly_reason = Column(Text, nullable=True)         # e.g. "statistical_outlier", "currency_mismatch"

    # LLM classification results
    llm_category = Column(String(100), nullable=True)    # category assigned by LLM
    llm_raw_response = Column(Text, nullable=True)       # raw LLM output for auditability
    llm_failed = Column(Boolean, nullable=False, default=False)

    # Relationship
    job = relationship("Job", back_populates="transactions")

    def __repr__(self):
        return f"<Transaction id={self.id} txn={self.txn_id!r} amount={self.amount}>"


# ---------------------------------------------------------------------------
# JobSummary – one row per completed job, holds LLM narrative
# ---------------------------------------------------------------------------
class JobSummary(Base):
    __tablename__ = "job_summaries"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    job_id = Column(
        String(36),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    total_spend_inr = Column(Float, nullable=True)
    total_spend_usd = Column(Float, nullable=True)
    top_merchants = Column(JSON, nullable=True)          # list of top 3 merchant names
    anomaly_count = Column(Integer, nullable=True)
    narrative = Column(Text, nullable=True)              # 2-3 sentence LLM summary
    risk_level = Column(String(10), nullable=True)       # low | medium | high

    # Relationship
    job = relationship("Job", back_populates="summary")

    def __repr__(self):
        return f"<JobSummary id={self.id} job_id={self.job_id} risk={self.risk_level!r}>"
