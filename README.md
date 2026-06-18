# 🚀 Transaction Processing Pipeline

Ever looked at a raw bank statement and wondered where your money actually went? This project takes messy, uncategorized transaction CSVs and turns them into clean, structured, and insightful data. 

Built with **FastAPI**, **Redis**, and powered by **LLaMA 3.1 (via Groq)**, this pipeline doesn't just parse rows—it acts like an AI financial analyst. It cleans your data, flags weird spending anomalies, categorizes your purchases, and gives you a summary of your spending habits.

## ✨ What It Does

1. **Ingests & Cleans**: Upload a CSV and the system normalizes dates, strips weird currency symbols, and removes duplicates.
2. **Flags Anomalies**: Automatically detects suspicious activity, like a transaction that's 3x your median spend, or a random USD charge at a local domestic merchant.
3. **AI Categorization**: Uncategorized transactions? No problem. The pipeline batches them up and asks an LLM (LLaMA 3.1 via Groq) to accurately tag them (Food, Travel, Utilities, etc.).
4. **Smart Summaries**: Generates a high-level spending summary, calculating total spend in different currencies, top merchants, and assigning a "risk level" based on the anomalies found.
5. **Asynchronous by Design**: Long CSV? We handle the heavy lifting in the background using **Redis and RQ** so the API stays lightning-fast.

## 🛠️ Tech Stack

- **Web Framework:** FastAPI
- **Database:** SQLite (via SQLAlchemy)
- **Background Jobs:** Redis + RQ (Redis Queue)
- **Data Processing:** Pandas
- **AI/LLM:** Groq API (llama-3.1-8b-instant)

## 🚦 Getting Started

### 1. Prerequisites
- Docker and Docker Compose (Easiest way)
- **OR** Python 3.9+ and a running Redis instance.
- A **Groq API Key** (Get one for free at [console.groq.com](https://console.groq.com/)).

### 2. Setup your Environment

Create a `.env` file in the root directory and add your Groq API key and Redis URL:

```env
GROQ_API_KEY=your_groq_api_key_here
REDIS_URL=redis://redis:6379/0  # If using Docker
```

### 3. Run the Application

**The Easy Way (Docker):**
```bash
docker-compose up --build
```

**The Manual Way (Local Python):**
1. Create a virtual environment: `python -m venv venv`
2. Activate it: `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
3. Install dependencies: `pip install -r requirements.txt`
4. Start the FastAPI server: `uvicorn app.main:app --reload`
5. In a new terminal, start the background worker: `rq worker -u redis://localhost:6379/0`

## 📡 How to Use It (API Endpoints)

Once the app is running (usually at `http://localhost:8000`), you can interact with the pipeline:

- **`POST /jobs/upload`**: Send your `.csv` file here. You'll get back a `job_id`.
- **`GET /jobs`**: See a list of all your uploaded files and their current status (`pending`, `processing`, `completed`, `failed`).
- **`GET /jobs/{job_id}/status`**: Check on a specific job. If it's done, you'll see a quick summary of the results.
- **`GET /jobs/{job_id}/results`**: The goldmine. Returns the fully processed payload, including every cleaned transaction, the AI's categories, a list of anomalies, and your personalized spending narrative.

*(You can also visit `http://localhost:8000/docs` for the interactive Swagger UI!)*

## 🧠 How the AI Works

Instead of relying on fragile regex rules to guess if "SBX*123" is coffee, we batch up unknown merchants and send them to Groq's insanely fast inference engine. We use a strictly formatted prompt to ensure the LLM returns exactly the JSON we need to map transactions to our internal categories safely. If the LLM fails or times out, we use exponential backoff to try again gracefully without crashing your job.

---
*Happy analyzing! 💸*
