"""Background worker for processing transactions."""

import io
import os
import json
import time
import logging
from datetime import datetime, timezone

import pandas as pd
from groq import Groq

from app.database import SessionLocal
from app.models import Job, Transaction, JobSummary

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_NAME = "llama-3.1-8b-instant"
BATCH_SIZE = 15
MAX_RETRIES = 3
BACKOFF_BASE = 2

DOMESTIC_MERCHANTS = {"swiggy", "ola", "irctc"}
VALID_CATEGORIES = {
    "Food", "Shopping", "Travel", "Transport",
    "Utilities", "Cash Withdrawal", "Entertainment", "Other",
}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def process_transaction_job(job_id: str, csv_bytes: bytes):
    """Processes a transaction CSV file from end-to-end."""
    db = SessionLocal()

    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found in database. Aborting.")
            return

        job.status = "processing"
        db.commit()

        df = clean_transactions(csv_bytes)
        job.row_count_clean = len(df)
        db.commit()

        df = flag_outliers(df)
        df = categorize_transactions(df)

        _save_transactions(db, job_id, df)

        summary_data = generate_summary(df)
        _save_summary(db, job_id, summary_data)

        job.status = "completed"
        job.completed_at = datetime.now(timezone.utc)
        db.commit()
        logger.info(f"Job {job_id} completed successfully.")

    except Exception as e:
        db.rollback()
        try:
            job = db.query(Job).filter(Job.id == job_id).first()
            if job:
                job.status = "failed"
                job.error_message = str(e)
                db.commit()
        except Exception:
            logger.error(f"Failed to update job {job_id} status after error.")
        logger.error(f"Job {job_id} failed: {e}")

    finally:
        db.close()


def clean_transactions(csv_bytes: bytes) -> pd.DataFrame:
    """Normalizes dates, strips currency symbols, and removes duplicates."""
    df = pd.read_csv(io.BytesIO(csv_bytes))

    df["date"] = df["date"].apply(_parse_date)
    df["amount"] = df["amount"].apply(_clean_amount)
    
    df["status"] = df["status"].astype(str).str.strip().str.upper()
    df["status"] = df["status"].replace({"NAN": None})

    df["currency"] = df["currency"].astype(str).str.strip().str.upper()
    df["currency"] = df["currency"].replace({"NAN": None})

    df["category"] = df["category"].fillna("").astype(str).str.strip()
    df["category"] = df["category"].replace({"": "Uncategorised", "nan": "Uncategorised"})

    df["merchant"] = df["merchant"].astype(str).str.strip()
    df["account_id"] = df["account_id"].astype(str).str.strip()
    df["txn_id"] = df["txn_id"].fillna("").astype(str).str.strip()
    df["notes"] = df["notes"].fillna("").astype(str).str.strip()

    row_count_before = len(df)
    df = df.drop_duplicates()
    row_count_after = len(df)
    logger.info(f"Deduplication: {row_count_before} -> {row_count_after}")

    df = df.reset_index(drop=True)
    return df


def _parse_date(value) -> str:
    if pd.isna(value):
        return None
    value = str(value).strip()
    formats = ["%d-%m-%Y", "%Y/%m/%d", "%Y-%m-%d"]
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    logger.warning(f"Could not parse date: {value!r}")
    return value


def _clean_amount(value) -> float:
    if pd.isna(value):
        return 0.0
    cleaned = str(value).strip().replace("$", "").replace(",", "")
    try:
        return round(float(cleaned), 2)
    except ValueError:
        logger.warning(f"Could not parse amount: {value!r}")
        return 0.0


def flag_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Flags anomalies like 3x median spend or currency mismatches."""
    df["is_flagged"] = False
    df["flag_reason"] = ""

    account_medians = df.groupby("account_id")["amount"].median()

    for idx, row in df.iterrows():
        reasons = []
        account_id = row["account_id"]
        amount = row["amount"]

        if account_id in account_medians.index:
            median_val = account_medians[account_id]
            if median_val > 0 and amount > (3 * median_val):
                reasons.append("statistical_outlier")

        currency = str(row.get("currency", "")).upper()
        merchant = str(row.get("merchant", "")).lower()
        if currency == "USD" and merchant in DOMESTIC_MERCHANTS:
            reasons.append("currency_mismatch")

        if reasons:
            df.at[idx, "is_flagged"] = True
            df.at[idx, "flag_reason"] = ", ".join(reasons)

    logger.info(f"Anomaly detection: flagged {df['is_flagged'].sum()} rows")
    return df


def categorize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Batches uncategorized transactions to an external API for classification."""
    df["predicted_category"] = None
    df["provider_response"] = None
    df["prediction_failed"] = False

    missing_categories = df[df["category"] == "Uncategorised"].index.tolist()

    if not missing_categories:
        logger.info("No missing categories. Skipping API call.")
        return df

    for batch_start in range(0, len(missing_categories), BATCH_SIZE):
        batch_indices = missing_categories[batch_start:batch_start + BATCH_SIZE]
        batch_df = df.loc[batch_indices]

        batch_items = []
        for idx, row in batch_df.iterrows():
            batch_items.append({
                "index": int(idx),
                "merchant": row["merchant"],
                "amount": row["amount"],
                "currency": row["currency"],
                "notes": row.get("notes", ""),
            })

        payload = _build_classification_prompt(batch_items)
        response_text = call_api_with_retry(payload)

        if response_text is None:
            for idx in batch_indices:
                df.at[idx, "prediction_failed"] = True
            continue

        _apply_classification_results(df, batch_indices, response_text)

    return df


def _build_classification_prompt(batch_items: list) -> str:
    items_json = json.dumps(batch_items, indent=2)
    return f"""Assign exactly ONE category from this list:
Food, Shopping, Travel, Transport, Utilities, Cash Withdrawal, Entertainment, Other

Respond with ONLY a valid JSON array. Each element must have "index" and "category" keys.
Do not include any explanation or markdown formatting.

Transactions:
{items_json}
"""


def _apply_classification_results(df: pd.DataFrame, batch_indices: list, response_text: str):
    try:
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        results = json.loads(cleaned)
        if not isinstance(results, list):
            raise ValueError("Response is not a JSON array")

        category_map = {}
        for item in results:
            idx = item.get("index")
            cat = item.get("category", "Other")
            if cat not in VALID_CATEGORIES:
                cat = "Other"
            category_map[idx] = cat

        for idx in batch_indices:
            df.at[idx, "predicted_category"] = category_map.get(int(idx), "Other")
            df.at[idx, "provider_response"] = response_text

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"Failed to parse classification response: {e}")
        for idx in batch_indices:
            df.at[idx, "prediction_failed"] = True
            df.at[idx, "provider_response"] = response_text


def generate_summary(df: pd.DataFrame) -> dict:
    """Generates a summary of the job run."""
    total_inr = round(float(df[df["currency"] == "INR"]["amount"].sum()), 2)
    total_usd = round(float(df[df["currency"] == "USD"]["amount"].sum()), 2)
    anomaly_count = int(df["is_flagged"].sum())

    top_merchants = (
        df.groupby("merchant")["amount"]
        .sum()
        .sort_values(ascending=False)
        .head(3)
        .index.tolist()
    )

    payload = f"""Produce a JSON summary based on these stats. ONLY valid JSON.

Stats:
- Total transactions: {len(df)}
- Total spend in INR: {total_inr}
- Total spend in USD: {total_usd}
- Top 3 merchants: {json.dumps(top_merchants)}
- Anomalies flagged: {anomaly_count}

Format:
{{
  "total_spend_inr": {total_inr},
  "total_spend_usd": {total_usd},
  "top_merchants": {json.dumps(top_merchants)},
  "anomaly_count": {anomaly_count},
  "summary_text": "A 2-3 sentence summary of spending patterns.",
  "risk_level": "low or medium or high"
}}
"""

    response_text = call_api_with_retry(payload)

    fallback = {
        "total_spend_inr": total_inr,
        "total_spend_usd": total_usd,
        "top_merchants": top_merchants,
        "anomaly_count": anomaly_count,
        "summary_text": "Summary generation failed.",
        "risk_level": _compute_fallback_risk(anomaly_count, len(df)),
    }

    if response_text is None:
        return fallback

    try:
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        summary = json.loads(cleaned)
        
        # Ensure we capture it safely
        if "narrative" in summary and "summary_text" not in summary:
             summary["summary_text"] = summary["narrative"]

        if summary.get("risk_level") not in ("low", "medium", "high"):
            summary["risk_level"] = _compute_fallback_risk(anomaly_count, len(df))

        return summary

    except (json.JSONDecodeError, ValueError):
        return fallback


def _compute_fallback_risk(anomaly_count: int, total_rows: int) -> str:
    if total_rows == 0:
        return "low"
    ratio = anomaly_count / total_rows
    if ratio > 0.2:
        return "high"
    elif ratio > 0.1:
        return "medium"
    return "low"


def call_api_with_retry(prompt: str) -> str | None:
    """Calls external API with exponential backoff."""
    client = Groq(api_key=GROQ_API_KEY)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2048,
            )
            return response.choices[0].message.content

        except Exception as e:
            wait_time = BACKOFF_BASE ** attempt
            logger.warning(f"API call failed: {e}. Retrying in {wait_time}s...")
            if attempt < MAX_RETRIES:
                time.sleep(wait_time)

    return None


def _save_transactions(db, job_id: str, df: pd.DataFrame):
    transactions = []
    for _, row in df.iterrows():
        txn = Transaction(
            job_id=job_id,
            txn_id=row.get("txn_id") or None,
            date=row.get("date"),
            merchant=row.get("merchant"),
            amount=row.get("amount"),
            currency=row.get("currency"),
            status=row.get("status"),
            category=row.get("category"),
            account_id=row.get("account_id"),
            is_flagged=bool(row.get("is_flagged", False)),
            flag_reason=row.get("flag_reason") or None,
            predicted_category=row.get("predicted_category"),
            provider_response=row.get("provider_response"),
            prediction_failed=bool(row.get("prediction_failed", False)),
        )
        transactions.append(txn)

    db.bulk_save_objects(transactions)
    db.commit()


def _save_summary(db, job_id: str, summary_data: dict):
    summary = JobSummary(
        job_id=job_id,
        total_spend_inr=summary_data.get("total_spend_inr"),
        total_spend_usd=summary_data.get("total_spend_usd"),
        top_merchants=summary_data.get("top_merchants"),
        anomaly_count=summary_data.get("anomaly_count"),
        summary_text=summary_data.get("summary_text") or summary_data.get("narrative"),
        risk_level=summary_data.get("risk_level"),
    )
    db.add(summary)
    db.commit()
