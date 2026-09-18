"""Task 1: document text -> structured extraction."""

from src.llm_client import LLMClient
from src.prompts import EXTRACTION_SYSTEM_PROMPT, build_extraction_user_prompt
from src.schemas import ExtractedComplaint
from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_structured_data(
    client: LLMClient, document_text: str, source_file: str
) -> ExtractedComplaint:
    """Run structured extraction for one document.

    Args:
        client: Configured LLM client.
        document_text: Raw text extracted from the complaint document.
        source_file: Original filename (used for prompt context and logging).

    Returns:
        A validated ExtractedComplaint instance.
    """
    logger.info("Extracting structured data: %s", source_file)
    user_prompt = build_extraction_user_prompt(document_text, source_file)

    result = client.complete_structured(
        system=EXTRACTION_SYSTEM_PROMPT,
        user=user_prompt,
        schema=ExtractedComplaint,
    )

    logger.info(
        "Extraction complete: %s | category=%s | status=%s | escalation=%s",
        source_file, result.complaint_category, result.overall_case_status,
        result.escalation_required,
    )
    return result
