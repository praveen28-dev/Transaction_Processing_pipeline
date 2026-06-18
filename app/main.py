"""
FastAPI application with all four required endpoints.

Endpoints:
  POST /jobs/upload         – upload CSV, validate, create job, enqueue task
  GET  /jobs/{job_id}/status – poll job status
  GET  /jobs/{job_id}/results – fetch full processed results
  GET  /jobs                 – list all jobs with optional ?status= filter
"""

import io
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import Base, engine, get_db
from app.models import Job, Transaction, JobSummary

# ---------------------------------------------------------------------------
# Create tables on startup (no manual migrations needed for assignment scope)
# ---------------------------------------------------------------------------
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI-Powered Transaction Processing Pipeline",
    description="Async pipeline for financial transaction cleaning, anomaly detection, and LLM classification.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# POST /jobs/upload
# ---------------------------------------------------------------------------
@app.post("/jobs/upload")
async def upload_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Accept a CSV file upload. Validate the file format,
    create a Job record with status=pending, enqueue the
    background processing task, and return the job_id immediately.
    """

    # --- Input validation: double-check file format before processing ---
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only .csv files are accepted.",
        )

    # Read the file content into memory
    contents = await file.read()

    # Basic content validation – ensure it is not empty
    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    # Count raw rows (subtract 1 for the header)
    raw_text = contents.decode("utf-8", errors="replace")
    raw_row_count = len(raw_text.strip().splitlines()) - 1

    # Create Job record
    new_job = Job(
        filename=file.filename,
        status="pending",
        row_count_raw=raw_row_count,
    )
    db.add(new_job)
    db.commit()
    db.refresh(new_job)

    # --- Enqueue background task via Redis RQ ---
    try:
        from redis import Redis
        from rq import Queue
        import os

        redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        redis_conn = Redis.from_url(redis_url)
        task_queue = Queue(connection=redis_conn)

        # Import the worker function and enqueue with the job id + raw csv bytes
        from app.tasks import process_transaction_job

        task_queue.enqueue(
            process_transaction_job,
            new_job.id,
            contents,
            job_timeout="10m",
        )
    except Exception as enqueue_error:
        # If enqueueing fails, mark the job as failed so it's not stuck in pending
        new_job.status = "failed"
        new_job.error_message = f"Failed to enqueue task: {str(enqueue_error)}"
        db.commit()
        raise HTTPException(
            status_code=500,
            detail="Failed to enqueue processing task. Please try again.",
        )

    return {
        "job_id": new_job.id,
        "status": new_job.status,
        "filename": new_job.filename,
        "row_count_raw": new_job.row_count_raw,
        "message": "File uploaded successfully. Processing has been queued.",
    }


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}/status
# ---------------------------------------------------------------------------
@app.get("/jobs/{job_id}/status")
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """
    Return the current status of the job. If completed,
    include high-level summary stats.
    """

    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    response = {
        "job_id": job.id,
        "status": job.status,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }

    # If completed, attach summary stats
    if job.status == "completed" and job.summary:
        response["summary"] = {
            "total_spend_inr": job.summary.total_spend_inr,
            "total_spend_usd": job.summary.total_spend_usd,
            "anomaly_count": job.summary.anomaly_count,
            "risk_level": job.summary.risk_level,
        }

    # If failed, include the error message
    if job.status == "failed" and job.error_message:
        response["error_message"] = job.error_message

    return response


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}/results
# ---------------------------------------------------------------------------
@app.get("/jobs/{job_id}/results")
def get_job_results(job_id: str, db: Session = Depends(get_db)):
    """
    Return the full structured output: cleaned transactions list,
    flagged anomalies, per-category spend breakdown, and LLM narrative.
    """

    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    if job.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed yet. Current status: {job.status}",
        )

    # Build cleaned transactions list
    transactions = []
    for txn in job.transactions:
        transactions.append({
            "txn_id": txn.txn_id,
            "date": txn.date,
            "merchant": txn.merchant,
            "amount": txn.amount,
            "currency": txn.currency,
            "status": txn.status,
            "category": txn.llm_category if txn.llm_category else txn.category,
            "account_id": txn.account_id,
            "is_anomaly": txn.is_anomaly,
            "anomaly_reason": txn.anomaly_reason,
        })

    # Build anomalies list (filtered)
    anomalies = [t for t in transactions if t["is_anomaly"]]

    # Build per-category spend breakdown
    category_spend = {}
    for txn in transactions:
        cat = txn["category"] or "Uncategorised"
        category_spend[cat] = round(category_spend.get(cat, 0) + (txn["amount"] or 0), 2)

    # LLM narrative summary
    narrative_summary = None
    if job.summary:
        narrative_summary = {
            "total_spend_inr": job.summary.total_spend_inr,
            "total_spend_usd": job.summary.total_spend_usd,
            "top_merchants": job.summary.top_merchants,
            "anomaly_count": job.summary.anomaly_count,
            "narrative": job.summary.narrative,
            "risk_level": job.summary.risk_level,
        }

    return {
        "job_id": job.id,
        "filename": job.filename,
        "row_count_raw": job.row_count_raw,
        "row_count_clean": job.row_count_clean,
        "transactions": transactions,
        "anomalies": anomalies,
        "category_spend_breakdown": category_spend,
        "llm_summary": narrative_summary,
    }


# ---------------------------------------------------------------------------
# GET /jobs
# ---------------------------------------------------------------------------
@app.get("/jobs")
def list_jobs(
    status: Optional[str] = Query(None, description="Filter by job status"),
    db: Session = Depends(get_db),
):
    """
    List all jobs with their status, filename, row count, and created_at.
    Supports filtering via ?status= query parameter.
    """

    query = db.query(Job)

    # Apply status filter if provided
    if status:
        valid_statuses = {"pending", "processing", "completed", "failed"}
        if status.lower() not in valid_statuses:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status filter. Valid values: {', '.join(sorted(valid_statuses))}",
            )
        query = query.filter(Job.status == status.lower())

    # Order by most recent first
    jobs = query.order_by(Job.created_at.desc()).all()

    return {
        "total": len(jobs),
        "jobs": [
            {
                "job_id": job.id,
                "filename": job.filename,
                "status": job.status,
                "row_count_raw": job.row_count_raw,
                "row_count_clean": job.row_count_clean,
                "created_at": job.created_at.isoformat() if job.created_at else None,
            }
            for job in jobs
        ],
    }
