"""Thin wrapper around the OpenAI SDK -- the only file that talks to
OpenAI directly.

Exposes two capabilities:
  * complete_text(system, user)              -> plain string
  * complete_structured(system, user, Model)  -> validated Pydantic instance

Both retry on transient failures (network errors, rate limits, invalid
JSON) with exponential backoff. Centralising this here means retry
logic and error handling are written once, and every task module
(extraction/email/summary) stays free of SDK-specific code.
"""

import json
import time
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError
from openai import OpenAI

from src.config import (
    MODEL_NAME, OPENAI_API_KEY,
    TEMPERATURE_EXTRACTION, TEMPERATURE_GENERATION, MAX_TOKENS,
    MAX_RETRIES, RETRY_BACKOFF_SECONDS,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Raised when a request to OpenAI cannot be completed after retries."""


class LLMClient:
    """Construct one instance per session with the desired model/API key."""

    def __init__(self, model: str = MODEL_NAME, openai_api_key: str = OPENAI_API_KEY):
        self.model = model

        if not openai_api_key:
            raise LLMError(
                "OPENAI_API_KEY is not set. Add it to your .env file "
                "(see .env.example) or enter it in the Streamlit sidebar."
            )

        self._client = OpenAI(api_key=openai_api_key)
        logger.info("LLMClient initialised (model=%s)", self.model)

    # ------------------------------------------------------------------
    # Plain text generation -- used by email_generator.py, summary_generator.py
    # ------------------------------------------------------------------
    def complete_text(self, system: str, user: str,
                       temperature: float = TEMPERATURE_GENERATION) -> str:
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    max_tokens=MAX_TOKENS,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                last_error = e
                logger.warning("Text generation attempt %d/%d failed: %s",
                                attempt, MAX_RETRIES, e)
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)

        raise LLMError(f"Text generation failed after {MAX_RETRIES} attempts: {last_error}")

    # ------------------------------------------------------------------
    # Structured (schema-validated) generation -- used by extraction.py
    # ------------------------------------------------------------------
    def complete_structured(self, system: str, user: str, schema: Type[T],
                             temperature: float = TEMPERATURE_EXTRACTION) -> T:
        """Request JSON matching `schema` and return a validated instance.

        The raw response is never trusted or persisted as-is: it is
        always parsed and run through schema.model_validate(), with
        automatic retries on invalid JSON or schema mismatches.
        """
        schema_description = json.dumps(schema.model_json_schema(), indent=2)
        json_system_prompt = (
            f"{system}\n\n"
            "Respond with ONLY a single valid JSON object -- no markdown "
            "code fences, no commentary before or after it. The JSON "
            f"object must conform to this JSON Schema:\n{schema_description}"
        )

        last_error = None
        current_user_prompt = user

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw_text = self._openai_json_call(json_system_prompt, current_user_prompt, temperature)
                cleaned_text = self._strip_code_fences(raw_text)
                parsed = json.loads(cleaned_text)
                validated = schema.model_validate(parsed)

                logger.debug("Structured extraction validated on attempt %d/%d (%s).",
                             attempt, MAX_RETRIES, schema.__name__)
                return validated

            except (json.JSONDecodeError, ValidationError) as e:
                last_error = e
                logger.warning(
                    "Structured output attempt %d/%d invalid (%s). Retrying...",
                    attempt, MAX_RETRIES, type(e).__name__,
                )
                current_user_prompt = (
                    user + "\n\nIMPORTANT: Your previous response was not valid "
                    "JSON matching the required schema. Return ONLY the raw JSON "
                    "object, with no extra text, no markdown, and no explanation."
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)

            except Exception as e:
                last_error = e
                logger.warning("Structured generation attempt %d/%d failed: %s",
                                attempt, MAX_RETRIES, e)
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)

        raise LLMError(f"Structured extraction failed after {MAX_RETRIES} attempts: {last_error}")

    def _openai_json_call(self, system: str, user: str, temperature: float) -> str:
        call_kwargs = dict(
            model=self.model,
            temperature=temperature,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        try:
            response = self._client.chat.completions.create(
                response_format={"type": "json_object"}, **call_kwargs
            )
        except Exception:
            # A few older/smaller models reject response_format entirely;
            # fall back to a plain call since the prompt already demands JSON.
            response = self._client.chat.completions.create(**call_kwargs)
        return response.choices[0].message.content

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Strip ```json fences and any stray text around the JSON object."""
        stripped = text.strip()

        if stripped.startswith("```"):
            parts = stripped.split("```")
            stripped = parts[1] if len(parts) >= 2 else stripped.strip("`")
            if stripped.lower().startswith("json"):
                stripped = stripped[4:]

        first_brace = stripped.find("{")
        last_brace = stripped.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            stripped = stripped[first_brace:last_brace + 1]

        return stripped.strip()
