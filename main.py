#!/usr/bin/env python3
"""CLI entry point: batch-process every document in data/ without the UI.

Usage:
    python main.py
    python main.py --data-dir path/to/other_data --workers 8
"""

import argparse
import sys
from pathlib import Path

from src.config import DATA_DIR, MAX_PARALLEL_WORKERS, FINAL_REPORT_PATH
from src.llm_client import LLMClient, LLMError
from src.workflow import run_batch
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _cli_progress_printer(done: int, total: int, current_file: str) -> None:
    print(f"[{done}/{total}] processed: {current_file}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AI Customer Complaint & Case Processing System (OpenAI-powered)"
    )
    parser.add_argument("--data-dir", default=str(DATA_DIR),
                         help="Folder containing complaint documents (default: data/).")
    parser.add_argument("--workers", type=int, default=MAX_PARALLEL_WORKERS,
                         help=f"Max documents processed in parallel (default: {MAX_PARALLEL_WORKERS}).")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    try:
        client = LLMClient()
    except LLMError as e:
        logger.error("Could not initialise LLM client: %s", e)
        print(f"\nERROR: {e}\n")
        print("Copy .env.example to .env and set OPENAI_API_KEY.")
        return 1

    print(f"Starting batch run on: {data_dir}")
    print(f"Using OpenAI model='{client.model}'\n")

    records = run_batch(
        client, data_dir=data_dir, max_workers=args.workers,
        progress_callback=_cli_progress_printer,
    )

    if not records:
        print("\nNo documents were processed. Check that the data folder "
              "contains .txt / .pdf / .docx files.")
        return 0

    succeeded = sum(1 for r in records if r.processing_status == "Success")
    failed = len(records) - succeeded

    print(f"\nDone. {succeeded} succeeded, {failed} failed.")
    print(f"Final report: {FINAL_REPORT_PATH}")
    print("Per-case outputs: output/structured_data/, output/customer_emails/, "
          "output/case_summaries/")

    if failed:
        print("\nSome documents failed -- check logs/pipeline.log for details.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
