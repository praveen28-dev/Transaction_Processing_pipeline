"""
Background worker task for the transaction processing pipeline.

Executed by a Python RQ worker. When a job is dequeued, this module
runs the following steps in order:

  a) Data Cleaning       – normalise dates, strip $, uppercase status, fill blanks, dedup
  b) Anomaly Detection   – 3× account median outlier, USD + domestic merchant mismatch
  c) LLM Classification  – batch uncategorised rows in chunks of 15 to Groq API
  d) LLM Narrative       – single call to produce JSON summary
  e) Retry Logic         – 3 retries with exponential backoff on all LLM calls
"""

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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = "llama-3.1-8b-instant"
LLM_BATCH_SIZE = 15                       # sweet spot for free-tier rate limits
LLM_MAX_RETRIES = 3
LLM_BACKOFF_BASE = 2                      # seconds — doubles each retry

# Domestic-only merchants (used for currency mismatch anomaly detection)
DOMESTIC_MERCHANTS = {"swiggy", "ola", "irctc"}

# Valid LLM categories the model is allowed to assign
VALID_CATEGORIES = {
    "Food", "Shopping", "Travel", "Transport",
    "Utilities", "Cash Withdrawal", "Entertainment", "Other",
}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ===========================================================================
# MAIN ENTRY POINT — called by RQ worker
# ===========================================================================
def process_transaction_job(job_id: str, csv_bytes: bytes):
    """
    Full pipeline: clean → detect anomalies → LLM classify → LLM narrative.
    Updates the Job record in PostgreSQL at each stage.
    """
    db = SessionLocal()

    try:
        # ---- Double-check: verify job exists before starting ----
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found in database. Aborting.")
            return

        # Mark job as processing
        job.status = "processing"
        db.commit()

        # ---- Step A: Data Cleaning ----
        df = _clean_data(csv_bytes)
        job.row_count_clean = len(df)
        db.commit()

        # ---- Step B: Anomaly Detection ----
        df = _detect_anomalies(df)

        # ---- Step C: LLM Classification (batched) ----
        df = _llm_classify_batched(df)

        # ---- Persist cleaned transactions to DB ----
        _save_transactions(db, job_id, df)

        # ---- Step D: LLM Narrative Summary ----
        summary_data = _llm_narrative_summary(df)
        _save_summary(db, job_id, summary_data)

        # ---- Mark job as completed ----
        job.status = "completed"
        job.completed_at = datetime.now(timezone.utc)
        db.commit()
        logger.info(f"Job {job_id} completed successfully.")

    except Exception as e:
        # If anything goes wrong, mark the job as failed with the error
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


# ===========================================================================
# STEP A: DATA CLEANING
# ===========================================================================
def _clean_data(csv_bytes: bytes) -> pd.DataFrame:
    """
    Normalise dates to ISO 8601, strip currency symbols from amounts,
    uppercase status and currency, fill missing categories, remove duplicates.
    """
    df = pd.read_csv(io.BytesIO(csv_bytes))

    # 1. Normalise date formats → ISO 8601 (YYYY-MM-DD)
    df["date"] = df["date"].apply(_parse_date)

    # 2. Strip currency symbols from amounts and convert to float
    df["amount"] = df["amount"].apply(_clean_amount)

    # 3. Uppercase status values
    df["status"] = df["status"].astype(str).str.strip().str.upper()
    df["status"] = df["status"].replace({"NAN": None})

    # 4. Uppercase currency values
    df["currency"] = df["currency"].astype(str).str.strip().str.upper()
    df["currency"] = df["currency"].replace({"NAN": None})

    # 5. Fill missing categories with 'Uncategorised'
    df["category"] = df["category"].fillna("").astype(str).str.strip()
    df["category"] = df["category"].replace({"": "Uncategorised", "nan": "Uncategorised"})

    # 6. Clean up merchant and account_id
    df["merchant"] = df["merchant"].astype(str).str.strip()
    df["account_id"] = df["account_id"].astype(str).str.strip()

    # 7. Clean up txn_id — keep blanks as empty strings
    df["txn_id"] = df["txn_id"].fillna("").astype(str).str.strip()

    # 8. Clean up notes
    df["notes"] = df["notes"].fillna("").astype(str).str.strip()

    # 9. Remove exact duplicate rows
    row_count_before = len(df)
    df = df.drop_duplicates()
    row_count_after = len(df)
    logger.info(
        f"Deduplication: {row_count_before} → {row_count_after} "
        f"(removed {row_count_before - row_count_after} duplicates)"
    )

    # Reset index after dropping duplicates
    df = df.reset_index(drop=True)

    return df


def _parse_date(value) -> str:
    """Try multiple date formats and return ISO 8601 (YYYY-MM-DD) string."""
    if pd.isna(value):
        return None

    value = str(value).strip()
    formats = [
        "%d-%m-%Y",     # DD-MM-YYYY
        "%Y/%m/%d",     # YYYY/MM/DD
        "%Y-%m-%d",     # YYYY-MM-DD (already ISO)
    ]
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    # If no format matched, return the original value
    logger.warning(f"Could not parse date: {value!r}")
    return value


def _clean_amount(value) -> float:
    """Strip currency symbols like $ and convert to float."""
    if pd.isna(value):
        return 0.0

    cleaned = str(value).strip().replace("$", "").replace(",", "")
    try:
        return round(float(cleaned), 2)
    except ValueError:
        logger.warning(f"Could not parse amount: {value!r}")
        return 0.0


# ===========================================================================
# STEP B: ANOMALY DETECTION
# ===========================================================================
def _detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag two types of anomalies:
      1. Statistical outlier — amount > 3× the account's median
      2. Currency mismatch  — USD with a domestic-only merchant
    """
    df["is_anomaly"] = False
    df["anomaly_reason"] = ""

    # --- Outlier detection: 3× per-account median ---
    account_medians = df.groupby("account_id")["amount"].median()

    for idx, row in df.iterrows():
        reasons = []
        account_id = row["account_id"]
        amount = row["amount"]

        # Check 1: Statistical outlier
        if account_id in account_medians.index:
            median_val = account_medians[account_id]
            if median_val > 0 and amount > (3 * median_val):
                reasons.append("statistical_outlier")

        # Check 2: Currency mismatch (USD + domestic merchant)
        currency = str(row.get("currency", "")).upper()
        merchant = str(row.get("merchant", "")).lower()
        if currency == "USD" and merchant in DOMESTIC_MERCHANTS:
            reasons.append("currency_mismatch")

        if reasons:
            df.at[idx, "is_anomaly"] = True
            df.at[idx, "anomaly_reason"] = ", ".join(reasons)

    anomaly_count = df["is_anomaly"].sum()
    logger.info(f"Anomaly detection: flagged {anomaly_count} rows")

    return df


# ===========================================================================
# STEP C: LLM CLASSIFICATION (Batched, 15 per call)
# ===========================================================================
def _llm_classify_batched(df: pd.DataFrame) -> pd.DataFrame:
    """
    For rows where category is 'Uncategorised', call Groq LLM in batches
    of 15 to assign a category. Updates df in-place with llm_category.
    """
    df["llm_category"] = None
    df["llm_raw_response"] = None
    df["llm_failed"] = False

    # Filter rows that need LLM classification
    uncategorised_mask = df["category"] == "Uncategorised"
    uncategorised_indices = df[uncategorised_mask].index.tolist()

    if not uncategorised_indices:
        logger.info("No uncategorised rows — skipping LLM classification.")
        return df

    logger.info(
        f"LLM classification: {len(uncategorised_indices)} rows to classify "
        f"in batches of {LLM_BATCH_SIZE}"
    )

    # Process in batches of 15
    for batch_start in range(0, len(uncategorised_indices), LLM_BATCH_SIZE):
        batch_indices = uncategorised_indices[batch_start:batch_start + LLM_BATCH_SIZE]
        batch_df = df.loc[batch_indices]

        # Build the batch payload for the LLM
        batch_items = []
        for idx, row in batch_df.iterrows():
            batch_items.append({
                "index": int(idx),
                "merchant": row["merchant"],
                "amount": row["amount"],
                "currency": row["currency"],
                "notes": row.get("notes", ""),
            })

        prompt = _build_classification_prompt(batch_items)

        # Call LLM with retry logic
        response_text = _call_llm_with_retry(prompt)

        if response_text is None:
            # All retries failed — mark batch as llm_failed
            for idx in batch_indices:
                df.at[idx, "llm_failed"] = True
            logger.warning(f"LLM classification failed for batch starting at index {batch_start}")
            continue

        # Parse LLM response and assign categories
        _apply_classification_results(df, batch_indices, response_text)

    classified_count = df["llm_category"].notna().sum()
    failed_count = df["llm_failed"].sum()
    logger.info(f"LLM classification done: {classified_count} classified, {failed_count} failed")

    return df


def _build_classification_prompt(batch_items: list) -> str:
    """Build the prompt for batch classification."""
    items_json = json.dumps(batch_items, indent=2)

    return f"""You are a financial transaction classifier. For each transaction below,
assign exactly ONE category from this list:
Food, Shopping, Travel, Transport, Utilities, Cash Withdrawal, Entertainment, Other

Respond with ONLY a valid JSON array. Each element must have "index" and "category" keys.
Do not include any explanation or markdown formatting.

Transactions to classify:
{items_json}

Example response format:
[{{"index": 0, "category": "Food"}}, {{"index": 1, "category": "Shopping"}}]
"""


def _apply_classification_results(
    df: pd.DataFrame,
    batch_indices: list,
    response_text: str,
):
    """Parse LLM JSON response and apply categories to the dataframe."""
    try:
        # Clean the response — strip markdown code fences if present
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        results = json.loads(cleaned)

        if not isinstance(results, list):
            raise ValueError("LLM response is not a JSON array")

        # Build a lookup from index → category
        category_map = {}
        for item in results:
            idx = item.get("index")
            cat = item.get("category", "Other")
            # Validate category against allowed list
            if cat not in VALID_CATEGORIES:
                cat = "Other"
            category_map[idx] = cat

        # Apply to dataframe
        for idx in batch_indices:
            assigned_cat = category_map.get(int(idx), "Other")
            df.at[idx, "llm_category"] = assigned_cat
            df.at[idx, "llm_raw_response"] = response_text

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        # If parsing fails, mark all rows in this batch as llm_failed
        logger.warning(f"Failed to parse LLM classification response: {e}")
        for idx in batch_indices:
            df.at[idx, "llm_failed"] = True
            df.at[idx, "llm_raw_response"] = response_text


# ===========================================================================
# STEP D: LLM NARRATIVE SUMMARY (Single call)
# ===========================================================================
def _llm_narrative_summary(df: pd.DataFrame) -> dict:
    """
    Make a single LLM call to produce a JSON summary:
    total spend by currency, top 3 merchants, anomaly count,
    2-3 sentence narrative, and risk_level (low/medium/high).
    """
    # Pre-compute stats to include in the prompt for accuracy
    total_inr = round(df[df["currency"] == "INR"]["amount"].sum(), 2)
    total_usd = round(df[df["currency"] == "USD"]["amount"].sum(), 2)
    anomaly_count = int(df["is_anomaly"].sum())

    top_merchants = (
        df.groupby("merchant")["amount"]
        .sum()
        .sort_values(ascending=False)
        .head(3)
        .index.tolist()
    )

    prompt = f"""You are a financial analyst. Based on the transaction data stats below,
produce a JSON summary. Respond with ONLY valid JSON, no markdown or explanation.

Stats:
- Total transactions: {len(df)}
- Total spend in INR: {total_inr}
- Total spend in USD: {total_usd}
- Top 3 merchants by spend: {json.dumps(top_merchants)}
- Anomalies flagged: {anomaly_count}
- Unique accounts: {df["account_id"].nunique()}

Required JSON format:
{{
  "total_spend_inr": {total_inr},
  "total_spend_usd": {total_usd},
  "top_merchants": {json.dumps(top_merchants)},
  "anomaly_count": {anomaly_count},
  "narrative": "A 2-3 sentence spending summary highlighting key patterns and risks.",
  "risk_level": "low or medium or high based on anomaly ratio"
}}
"""

    response_text = _call_llm_with_retry(prompt)

    if response_text is None:
        # LLM failed after all retries — return computed stats with fallback narrative
        logger.warning("LLM narrative summary failed. Using fallback.")
        return {
            "total_spend_inr": total_inr,
            "total_spend_usd": total_usd,
            "top_merchants": top_merchants,
            "anomaly_count": anomaly_count,
            "narrative": "LLM summary generation failed after retries.",
            "risk_level": _compute_fallback_risk(anomaly_count, len(df)),
        }

    # Parse the LLM response
    try:
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        summary = json.loads(cleaned)

        # Cross-validate: ensure expected keys exist
        required_keys = {"total_spend_inr", "total_spend_usd", "top_merchants",
                         "anomaly_count", "narrative", "risk_level"}
        if not required_keys.issubset(summary.keys()):
            raise ValueError(f"Missing keys in LLM response: {required_keys - summary.keys()}")

        # Validate risk_level
        if summary.get("risk_level") not in ("low", "medium", "high"):
            summary["risk_level"] = _compute_fallback_risk(anomaly_count, len(df))

        return summary

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse LLM narrative response: {e}")
        return {
            "total_spend_inr": total_inr,
            "total_spend_usd": total_usd,
            "top_merchants": top_merchants,
            "anomaly_count": anomaly_count,
            "narrative": "LLM summary generation returned invalid format.",
            "risk_level": _compute_fallback_risk(anomaly_count, len(df)),
        }


def _compute_fallback_risk(anomaly_count: int, total_rows: int) -> str:
    """Compute a simple risk level when the LLM is unavailable."""
    if total_rows == 0:
        return "low"
    ratio = anomaly_count / total_rows
    if ratio > 0.2:
        return "high"
    elif ratio > 0.1:
        return "medium"
    return "low"


# ===========================================================================
# STEP E: LLM CALL WITH RETRY (Exponential Backoff)
# ===========================================================================
def _call_llm_with_retry(prompt: str) -> str | None:
    """
    Call Groq API with up to 3 retries and exponential backoff.
    Returns the response text on success, or None if all retries fail.
    """
    client = Groq(api_key=GROQ_API_KEY)

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,       # low temp for consistent JSON output
                max_tokens=2048,
            )
            return response.choices[0].message.content

        except Exception as e:
            wait_time = LLM_BACKOFF_BASE ** attempt   # 2s, 4s, 8s
            logger.warning(
                f"LLM call attempt {attempt}/{LLM_MAX_RETRIES} failed: {e}. "
                f"Retrying in {wait_time}s..."
            )
            if attempt < LLM_MAX_RETRIES:
                time.sleep(wait_time)

    logger.error("All LLM retries exhausted. Returning None.")
    return None


# ===========================================================================
# DATABASE PERSISTENCE HELPERS
# ===========================================================================
def _save_transactions(db, job_id: str, df: pd.DataFrame):
    """Persist all cleaned transaction rows to the database."""
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
            is_anomaly=bool(row.get("is_anomaly", False)),
            anomaly_reason=row.get("anomaly_reason") or None,
            llm_category=row.get("llm_category"),
            llm_raw_response=row.get("llm_raw_response"),
            llm_failed=bool(row.get("llm_failed", False)),
        )
        transactions.append(txn)

    db.bulk_save_objects(transactions)
    db.commit()
    logger.info(f"Saved {len(transactions)} transactions for job {job_id}")


def _save_summary(db, job_id: str, summary_data: dict):
    """Persist the LLM narrative summary to the database."""
    summary = JobSummary(
        job_id=job_id,
        total_spend_inr=summary_data.get("total_spend_inr"),
        total_spend_usd=summary_data.get("total_spend_usd"),
        top_merchants=summary_data.get("top_merchants"),
        anomaly_count=summary_data.get("anomaly_count"),
        narrative=summary_data.get("narrative"),
        risk_level=summary_data.get("risk_level"),
    )
    db.add(summary)
    db.commit()
    logger.info(f"Saved job summary for job {job_id}")
