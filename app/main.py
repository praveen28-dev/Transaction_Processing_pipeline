import io
import os
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import Base, engine, get_db
from app.models import Job, Transaction, JobSummary

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Transaction Processing Pipeline",
    description="Pipeline for transaction cleaning, flagging, and categorizing.",
    version="1.0.0",
)


@app.post("/jobs/upload")
async def upload_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Uploads and validates a transaction CSV, then queues it for processing."""
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only .csv files are accepted.",
        )

    file_bytes = await file.read()

    # Check for empty files
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    csv_text = file_bytes.decode("utf-8", errors="replace")
    raw_row_count = len(csv_text.strip().splitlines()) - 1

    job = Job(
        filename=file.filename,
        status="pending",
        row_count_raw=raw_row_count,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    try:
        from redis import Redis
        from rq import Queue

        redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        redis_conn = Redis.from_url(redis_url)
        task_queue = Queue(connection=redis_conn)

        from app.tasks import process_transaction_job

        task_queue.enqueue(
            process_transaction_job,
            job.id,
            file_bytes,
            job_timeout="10m",
        )
    except Exception as enqueue_error:
        job.status = "failed"
        job.error_message = f"Failed to enqueue task: {str(enqueue_error)}"
        db.commit()
        raise HTTPException(
            status_code=500,
            detail="Failed to enqueue processing task. Please try again.",
        )

    return {
        "job_id": job.id,
        "status": job.status,
        "filename": job.filename,
        "row_count_raw": job.row_count_raw,
        "message": "File uploaded successfully. Processing has been queued.",
    }


@app.get("/jobs/{job_id}/status")
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """Returns the current state of a job."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    response = {
        "job_id": job.id,
        "status": job.status,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }

    if job.status == "completed" and job.summary:
        response["summary"] = {
            "total_spend_inr": job.summary.total_spend_inr,
            "total_spend_usd": job.summary.total_spend_usd,
            "anomaly_count": job.summary.anomaly_count,
            "risk_level": job.summary.risk_level,
        }

    if job.status == "failed" and job.error_message:
        response["error_message"] = job.error_message

    return response


@app.get("/jobs/{job_id}/results")
def get_job_results(job_id: str, db: Session = Depends(get_db)):
    """Returns the fully processed payload for a completed job."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    if job.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed yet. Current status: {job.status}",
        )

    transactions = []
    for txn in job.transactions:
        transactions.append({
            "txn_id": txn.txn_id,
            "date": txn.date,
            "merchant": txn.merchant,
            "amount": txn.amount,
            "currency": txn.currency,
            "status": txn.status,
            "category": txn.predicted_category if txn.predicted_category else txn.category,
            "account_id": txn.account_id,
            "is_anomaly": txn.is_flagged,
            "anomaly_reason": txn.flag_reason,
        })

    anomalies = [t for t in transactions if t["is_anomaly"]]

    category_spend = {}
    for txn in transactions:
        cat = txn["category"] or "Uncategorised"
        category_spend[cat] = round(category_spend.get(cat, 0) + (txn["amount"] or 0), 2)

    summary_stats = None
    if job.summary:
        summary_stats = {
            "total_spend_inr": job.summary.total_spend_inr,
            "total_spend_usd": job.summary.total_spend_usd,
            "top_merchants": job.summary.top_merchants,
            "anomaly_count": job.summary.anomaly_count,
            "narrative": job.summary.summary_text,
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
        "job_summary": summary_stats,
    }


@app.get("/jobs")
def list_jobs(
    status: Optional[str] = Query(None, description="Filter by job status"),
    db: Session = Depends(get_db),
):
    """Lists all jobs, with optional filtering by status."""
    query = db.query(Job)

    if status:
        valid_statuses = {"pending", "processing", "completed", "failed"}
        if status.lower() not in valid_statuses:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status filter. Valid values: {', '.join(sorted(valid_statuses))}",
            )
        query = query.filter(Job.status == status.lower())

    jobs = query.order_by(Job.created_at.desc()).all()

    return {
        "total": len(jobs),
        "jobs": [
            {
                "job_id": j.id,
                "filename": j.filename,
                "status": j.status,
                "row_count_raw": j.row_count_raw,
                "row_count_clean": j.row_count_clean,
                "created_at": j.created_at.isoformat() if j.created_at else None,
            }
            for j in jobs
        ],
    }
