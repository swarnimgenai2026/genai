"""Centralised configuration: paths, OpenAI settings, and tuning knobs.

All values are read from environment variables (via .env locally, or
the deployment platform's secrets manager in production), so nothing
sensitive is hard-coded and behaviour can be tuned without touching code.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "output"))

STRUCTURED_DATA_DIR = OUTPUT_DIR / "structured_data"
CUSTOMER_EMAILS_DIR = OUTPUT_DIR / "customer_emails"
CASE_SUMMARIES_DIR = OUTPUT_DIR / "case_summaries"
FINAL_REPORT_PATH = OUTPUT_DIR / "final_report.csv"
LOG_DIR = BASE_DIR / "logs"

for _folder in (DATA_DIR, STRUCTURED_DATA_DIR, CUSTOMER_EMAILS_DIR,
                CASE_SUMMARIES_DIR, LOG_DIR):
    _folder.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------
# Supported input formats
# --------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}

# --------------------------------------------------------------------
# OpenAI settings
# --------------------------------------------------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "").strip() or "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --------------------------------------------------------------------
# Generation controls
# --------------------------------------------------------------------
TEMPERATURE_EXTRACTION = 0.0   # deterministic, factual JSON output
TEMPERATURE_GENERATION = 0.4   # natural but still grounded prose
MAX_TOKENS = 1200

# --------------------------------------------------------------------
# Reliability controls
# --------------------------------------------------------------------
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2

# --------------------------------------------------------------------
# Batch / concurrency controls
# --------------------------------------------------------------------
MAX_PARALLEL_WORKERS = int(os.getenv("MAX_PARALLEL_WORKERS", "4"))

# --------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
