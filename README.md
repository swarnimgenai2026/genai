# AI Customer Complaint & Case Processing System

An automated pipeline that ingests customer complaint documents (`.txt`, `.pdf`, `.docx`),
runs them through a 3-stage LLM workflow powered by OpenAI, and produces
structured data, a customer-facing reply email, an internal case summary,
and a consolidated CSV report — with a Streamlit UI on top.

---

## Architecture

```
data/ (input documents)
      │
      ▼
┌─────────────────────┐
│ document_reader.py   │  Ingestion: txt / pdf / docx → raw text
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│ extraction.py         │  Task 1: LLM → Pydantic-validated ExtractedComplaint
│ (schemas.py)           │
└─────────┬────────────┘
          │
   ┌──────┴──────┐
   ▼             ▼
┌─────────────┐ ┌─────────────────────┐
│email_        │ │summary_generator.py  │
│generator.py  │ │ Task 3: internal      │
│ Task 2:       │ │ case summary          │
│ customer email│ └─────────────────────┘
└─────────────┘
          │
          ▼
┌─────────────────────┐
│ workflow.py           │  Orchestrates tasks 1→2/3 per doc,
│                        │  batches + parallelises across documents,
│                        │  writes output/ + final_report.csv
└─────────────────────┘
          │
          ▼
┌─────────────────────┐
│ app.py (Streamlit)    │  UI: upload/run/browse/download
│ main.py (CLI)          │  Headless batch runner
└─────────────────────┘
```

All OpenAI calls flow through `LLMClient` (`src/llm_client.py`); no other
module imports the SDK directly. Each document runs through three
independent LLM calls, each with its own prompt and output contract:

1. **Extraction** — output is forced into JSON and validated against the
   `ExtractedComplaint` Pydantic model, with automatic retry on invalid
   output. The raw LLM response is never persisted directly.
2. **Customer email** — grounded in the *validated* structured data, not
   the raw document, so it cannot surface facts that failed validation.
3. **Case summary** — same grounding, different audience/tone.

Within one document, tasks 2 and 3 run concurrently (both depend only on
task 1). Across documents, whole pipelines run concurrently up to
`MAX_PARALLEL_WORKERS` via `ThreadPoolExecutor`.

---

## Project structure

```
ai-complaint-system/
├── data/                       # input documents (5 samples: txt/docx/pdf)
├── output/
│   ├── structured_data/        # <file>.json per case (validated schema)
│   ├── customer_emails/        # <file>_email.txt per case
│   ├── case_summaries/         # <file>_summary.txt per case
│   └── final_report.csv        # consolidated view of every processed doc
├── logs/
│   └── pipeline.log            # rotating log file
├── src/
│   ├── config.py                 # paths, OpenAI settings, env loading
│   ├── schemas.py                 # Pydantic models
│   ├── document_reader.py         # ingestion: txt/pdf/docx
│   ├── llm_client.py              # OpenAI wrapper: retries, JSON validation
│   ├── prompts.py                  # prompt templates
│   ├── extraction.py               # Task 1
│   ├── email_generator.py          # Task 2
│   ├── summary_generator.py        # Task 3
│   ├── workflow.py                 # orchestration + batching
│   └── utils/logger.py             # logging setup
├── app.py                        # Streamlit UI
├── main.py                       # CLI entry point
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

**Requirements:** Python 3.10+

```bash
cd ai-complaint-system
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# set OPENAI_API_KEY in .env (https://platform.openai.com/api-keys)
```

### CLI (batch mode)

```bash
python main.py
```

Reads everything in `data/`, runs the pipeline, and writes
`output/final_report.csv` plus per-case files.

### Streamlit app (local)

```bash
streamlit run app.py
```

Open `http://localhost:8501`, enter your API key in the sidebar (or
leave blank if already set in `.env`), and click **Process all documents**.



## Git

```bash
git init
git add .
git commit -m "Initial commit: AI complaint & case processing system"
git branch -M main
git remote add origin https://github.com/<username>/ai-complaint-system.git
git push -u origin main
```

---

## Error handling

- **File-level errors** (corrupt PDF, empty file, bad encoding,
  unsupported extension) are caught in `document_reader.py`, logged, and
  the file is marked `Failed` in the report — the batch continues.
- **LLM-level errors** (timeouts, rate limits, malformed JSON) are
  retried up to `MAX_RETRIES` times with backoff in `llm_client.py`.
  Persistent failures are recorded with their error message; the batch
  is not aborted.
- Logging defaults to `INFO`; set `LOG_LEVEL=DEBUG` in `.env` for a
  fine-grained trace of every step (prompt sizes, retry attempts,
  response lengths). Output goes to the console and to
  `logs/pipeline.log` (rotating, capped at ~2 MB × 3 files).

---

## Configuration

Change the model in `.env`:

```env
MODEL_NAME=gpt-4o-mini
```

`gpt-4o-mini` is the default. Any chat-completion model your API key
has access to (e.g. `gpt-4o`) works without code changes — override it
in `.env` or directly in the Streamlit sidebar.

---

## Technical skills demonstrated

Python · LLM API integration (OpenAI) · Prompt engineering · Structured
outputs via Pydantic · Batch processing · Sequential + parallel workflow
orchestration (`ThreadPoolExecutor`) · Error handling · Modular package
layout · Multi-format file processing (txt/pdf/docx) · Logging ·
Streamlit UI · Git/GitHub · 
