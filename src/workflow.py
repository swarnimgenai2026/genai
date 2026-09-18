"""Orchestrates the 3-task pipeline per document and runs it as a batch.

Dependency graph per document:

    Document text
         |
         v
    [Task 1: Extraction]          -- must run first
         |
    +----+----+
    v         v
  [Task 2]  [Task 3]               -- both depend only on Task 1,
   Email     Summary                  so they run concurrently

Across documents, whole pipelines run concurrently (bounded by
MAX_PARALLEL_WORKERS) via ThreadPoolExecutor -- appropriate here since
each task is I/O-bound (waiting on the OpenAI API), not CPU-bound.
"""

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Optional

from src.config import (
    DATA_DIR, STRUCTURED_DATA_DIR, CUSTOMER_EMAILS_DIR, CASE_SUMMARIES_DIR,
    FINAL_REPORT_PATH, MAX_PARALLEL_WORKERS,
)
from src.document_reader import (
    DocumentReadError, RawDocument, discover_documents, read_document,
)
from src.extraction import extract_structured_data
from src.email_generator import generate_customer_email
from src.summary_generator import generate_case_summary
from src.llm_client import LLMClient, LLMError
from src.schemas import CaseRecord
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _safe_stem(file_name: str) -> str:
    """Filesystem-safe base name for a document's output files."""
    return Path(file_name).stem.replace(" ", "_")


def process_single_document(client: LLMClient, raw_doc: RawDocument) -> CaseRecord:
    """Run the full 3-task pipeline for one already-read document.

    Returns a CaseRecord describing success (with all generated content)
    or failure (with an error message). Never raises -- failures are
    captured so a batch run can continue past a single bad document.
    """
    source_file = raw_doc.file_name
    logger.info("Processing document: %s", source_file)

    try:
        extracted = extract_structured_data(client, raw_doc.text, source_file)

        with ThreadPoolExecutor(max_workers=2) as inner_pool:
            email_future = inner_pool.submit(
                generate_customer_email, client, raw_doc.text, extracted, source_file
            )
            summary_future = inner_pool.submit(
                generate_case_summary, client, raw_doc.text, extracted, source_file
            )
            customer_email_text = email_future.result()
            case_summary_text = summary_future.result()

        record = CaseRecord(
            source_file=source_file,
            extracted=extracted,
            customer_email_text=customer_email_text,
            case_summary_text=case_summary_text,
            processing_status="Success",
        )
        _persist_case_outputs(record)
        logger.info("Pipeline succeeded: %s", source_file)
        return record

    except LLMError as e:
        logger.error("Pipeline failed (LLM error) for %s: %s", source_file, e)
        return CaseRecord(source_file=source_file, processing_status="Failed",
                           error_message=str(e))
    except Exception as e:
        logger.exception("Pipeline failed (unexpected error) for %s", source_file)
        return CaseRecord(source_file=source_file, processing_status="Failed",
                           error_message=str(e))


def _persist_case_outputs(record: CaseRecord) -> None:
    """Write structured JSON, email, and summary files for one case."""
    stem = _safe_stem(record.source_file)

    (STRUCTURED_DATA_DIR / f"{stem}.json").write_text(
        record.extracted.model_dump_json(indent=2), encoding="utf-8"
    )
    (CUSTOMER_EMAILS_DIR / f"{stem}_email.txt").write_text(
        record.customer_email_text or "", encoding="utf-8"
    )
    (CASE_SUMMARIES_DIR / f"{stem}_summary.txt").write_text(
        record.case_summary_text or "", encoding="utf-8"
    )
    logger.debug("Persisted outputs for %s", record.source_file)


def write_final_report(records: List[CaseRecord],
                        report_path: Path = FINAL_REPORT_PATH) -> Path:
    """Write a consolidated CSV report, one row per processed document."""
    rows = [r.to_report_row() for r in records]
    if not rows:
        logger.warning("No records to write -- final report skipped.")
        return report_path

    fieldnames = list(rows[0].keys())
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Final report written: %s (%d rows)", report_path, len(rows))
    return report_path


def run_batch(
    client: LLMClient,
    data_dir: Path = DATA_DIR,
    max_workers: int = MAX_PARALLEL_WORKERS,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[CaseRecord]:
    """Process every eligible document in data_dir and write the final report.

    Args:
        client: Configured LLM client.
        data_dir: Folder to scan for documents.
        max_workers: Max documents processed concurrently.
        progress_callback: Optional callback(done, total, current_file),
            invoked as each document finishes (used by the Streamlit UI
            to drive a progress bar).

    Returns:
        One CaseRecord per discovered document (successes and failures).
    """
    logger.info("Batch run starting | data_dir=%s | max_workers=%d", data_dir, max_workers)

    file_paths = discover_documents(data_dir)
    total = len(file_paths)
    records: List[CaseRecord] = []

    if total == 0:
        logger.warning("No eligible documents found in %s", data_dir)
        return records

    # Read files first (sequential, cheap) so unreadable files are
    # recorded immediately without wasting an LLM call.
    raw_docs: List[RawDocument] = []
    for path in file_paths:
        try:
            raw_docs.append(read_document(path))
        except DocumentReadError as e:
            logger.error("Failed to read %s: %s", path.name, e)
            records.append(
                CaseRecord(source_file=path.name, processing_status="Failed",
                           error_message=str(e))
            )

    if not raw_docs:
        logger.warning("Every document failed to read -- skipping LLM processing.")
        write_final_report(records)
        return records

    done_count = 0

    def _run_one(raw_doc: RawDocument) -> CaseRecord:
        return process_single_document(client, raw_doc)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_doc = {pool.submit(_run_one, doc): doc for doc in raw_docs}

        for future in as_completed(future_to_doc):
            doc = future_to_doc[future]
            try:
                record = future.result()
            except Exception as e:
                logger.exception("Unhandled failure processing %s", doc.file_name)
                record = CaseRecord(source_file=doc.file_name,
                                     processing_status="Failed", error_message=str(e))

            records.append(record)
            done_count += 1
            if progress_callback:
                progress_callback(done_count, total, doc.file_name)

    # Restore original file-listing order (parallel completion order varies).
    original_order = {p.name: i for i, p in enumerate(file_paths)}
    records.sort(key=lambda r: original_order.get(r.source_file, 999999))

    write_final_report(records)

    succeeded = sum(1 for r in records if r.processing_status == "Success")
    logger.info("Batch run finished | %d succeeded, %d failed, %d total",
                succeeded, len(records) - succeeded, len(records))
    return records
