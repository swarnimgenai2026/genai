"""Task 2: (document + extracted data) -> customer-facing reply email.

Depends on Task 1's validated ExtractedComplaint as the source of truth
for facts; the raw document is passed only for tone/context.
"""

from src.llm_client import LLMClient
from src.prompts import EMAIL_SYSTEM_PROMPT, build_email_user_prompt
from src.schemas import ExtractedComplaint
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_customer_email(
    client: LLMClient, document_text: str, extracted: ExtractedComplaint,
    source_file: str,
) -> str:
    """Generate a professional customer-facing reply email for one case."""
    logger.info("Generating customer email: %s", source_file)

    extracted_json = extracted.model_dump_json(indent=2)
    user_prompt = build_email_user_prompt(document_text, extracted_json)
    email_text = client.complete_text(system=EMAIL_SYSTEM_PROMPT, user=user_prompt)

    logger.debug("Customer email generated (%d chars): %s", len(email_text), source_file)
    return email_text
