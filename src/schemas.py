"""Pydantic data contracts for the pipeline.

ExtractedComplaint is the schema the LLM's output must validate against
before it is used or persisted anywhere -- raw LLM text is never saved
as-is. CaseRecord bundles one document's full result (structured data +
generated email + generated summary) for the final report.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator

YesNo = Literal["Yes", "No"]
CaseStatus = Literal["Open", "In Progress", "Resolved", "Escalated", "Closed", "Unknown"]


class ExtractedComplaint(BaseModel):
    """Structured fields extracted from a single complaint document."""

    customer_name: Optional[str] = Field(default=None, description="Full name of the customer, if present.")
    email: Optional[str] = Field(default=None, description="Customer email address, if present.")
    phone_number: Optional[str] = Field(default=None, description="Customer phone number, if present.")
    complaint_category: Optional[str] = Field(
        default=None,
        description="Short category, e.g. Billing, Product Defect, Delivery Delay, "
                    "Service Quality, Technical Issue, Refund, Other.",
    )
    issue_description: Optional[str] = Field(default=None, description="Concise description of the issue raised.")
    resolution_provided: Optional[str] = Field(
        default=None, description="Resolution/action already provided, if any was mentioned."
    )
    is_complaint: YesNo = Field(description="Whether the document is genuinely a complaint.")
    escalation_required: YesNo = Field(description="Whether the case needs escalation to a higher team.")
    supporting_document_available: YesNo = Field(
        description="Whether supporting evidence/attachments are referenced."
    )
    overall_case_status: CaseStatus = Field(
        description="Overall status of the case based on the document content."
    )

    @field_validator("email")
    @classmethod
    def _normalise_email(cls, value: Optional[str]) -> Optional[str]:
        if value:
            value = value.strip()
            if value.lower() in {"n/a", "na", "none", "not provided", ""}:
                return None
        return value or None

    @field_validator(
        "customer_name", "phone_number", "complaint_category",
        "issue_description", "resolution_provided",
    )
    @classmethod
    def _blank_placeholder_to_none(cls, value: Optional[str]) -> Optional[str]:
        """Normalise LLM placeholder text ("N/A", "None", ...) to a real None."""
        if value is None:
            return None
        value = value.strip()
        if value.lower() in {"n/a", "na", "none", "not provided", ""}:
            return None
        return value


class CaseRecord(BaseModel):
    """One fully-processed case: source file, structured data, generated
    artefacts, and processing status. Flattens into one CSV report row.
    """

    source_file: str
    extracted: Optional[ExtractedComplaint] = None
    customer_email_text: Optional[str] = None
    case_summary_text: Optional[str] = None
    processing_status: Literal["Success", "Failed"] = "Success"
    error_message: Optional[str] = None

    def to_report_row(self) -> dict:
        row = {
            "source_file": self.source_file,
            "processing_status": self.processing_status,
            "error_message": self.error_message or "",
        }
        if self.extracted:
            row.update(
                {
                    "customer_name": self.extracted.customer_name or "",
                    "email": self.extracted.email or "",
                    "phone_number": self.extracted.phone_number or "",
                    "complaint_category": self.extracted.complaint_category or "",
                    "issue_description": self.extracted.issue_description or "",
                    "resolution_provided": self.extracted.resolution_provided or "",
                    "is_complaint": self.extracted.is_complaint,
                    "escalation_required": self.extracted.escalation_required,
                    "supporting_document_available": self.extracted.supporting_document_available,
                    "overall_case_status": self.extracted.overall_case_status,
                }
            )
        else:
            row.update(
                {
                    "customer_name": "", "email": "", "phone_number": "",
                    "complaint_category": "", "issue_description": "",
                    "resolution_provided": "", "is_complaint": "",
                    "escalation_required": "", "supporting_document_available": "",
                    "overall_case_status": "",
                }
            )
        row["has_customer_email"] = bool(self.customer_email_text)
        row["has_case_summary"] = bool(self.case_summary_text)
        return row
