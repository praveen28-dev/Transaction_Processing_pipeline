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


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    filename = Column(String(255), nullable=False)
    status = Column(
        String(20),
        nullable=False,
        default="pending",
        index=True,
    )
    row_count_raw = Column(Integer, nullable=True)
    row_count_clean = Column(Integer, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    completed_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)

    transactions = relationship(
        "Transaction",
        back_populates="job",
        cascade="all, delete-orphan",
    )
    summary = relationship(
        "JobSummary",
        back_populates="job",
        uselist=False,
        cascade="all, delete-orphan",
    )


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    job_id = Column(
        String(36),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    txn_id = Column(String(50), nullable=True)
    date = Column(String(30), nullable=True)
    merchant = Column(String(255), nullable=True)
    amount = Column(Float, nullable=True)
    currency = Column(String(10), nullable=True)
    status = Column(String(20), nullable=True)
    category = Column(String(100), nullable=True)
    account_id = Column(String(50), nullable=True)

    is_flagged = Column(Boolean, nullable=False, default=False)
    flag_reason = Column(Text, nullable=True)

    predicted_category = Column(String(100), nullable=True)
    provider_response = Column(Text, nullable=True)
    prediction_failed = Column(Boolean, nullable=False, default=False)

    job = relationship("Job", back_populates="transactions")


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
    top_merchants = Column(JSON, nullable=True)
    anomaly_count = Column(Integer, nullable=True)
    summary_text = Column(Text, nullable=True)
    risk_level = Column(String(10), nullable=True)

    job = relationship("Job", back_populates="summary")
