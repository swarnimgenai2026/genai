"""Task 3: (document + extracted data) -> internal case summary.

Mirrors email_generator.py's grounding pattern but targets a different
audience (internal ops team) and a different system prompt.
"""

from src.llm_client import LLMClient
from src.prompts import CASE_SUMMARY_SYSTEM_PROMPT, build_case_summary_user_prompt
from src.schemas import ExtractedComplaint
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_case_summary(
    client: LLMClient, document_text: str, extracted: ExtractedComplaint,
    source_file: str,
) -> str:
    """Generate a concise internal case summary for one case."""
    logger.info("Generating case summary: %s", source_file)

    extracted_json = extracted.model_dump_json(indent=2)
    user_prompt = build_case_summary_user_prompt(document_text, extracted_json)
    summary_text = client.complete_text(system=CASE_SUMMARY_SYSTEM_PROMPT, user=user_prompt)

    logger.debug("Case summary generated (%d chars): %s", len(summary_text), source_file)
    return summary_text
