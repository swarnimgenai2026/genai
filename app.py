"""Streamlit UI for the AI Customer Complaint & Case Processing System.

Run locally:  streamlit run app.py
Deploy:       push to GitHub, deploy on share.streamlit.io (see README).
"""

import os
import shutil

import pandas as pd
import streamlit as st

from src.config import (
    DATA_DIR, STRUCTURED_DATA_DIR, CUSTOMER_EMAILS_DIR,
    CASE_SUMMARIES_DIR, FINAL_REPORT_PATH,
)
from src.llm_client import LLMClient, LLMError
from src.workflow import run_batch
from src.schemas import CaseRecord
from src.utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(
    page_title="AI Complaint & Case Processing System",
    page_icon="📋",
    layout="wide",
)

# --------------------------------------------------------------------
# Sidebar: model / API key configuration
# --------------------------------------------------------------------
st.sidebar.title("⚙️ Configuration")

model_name = st.sidebar.text_input("OpenAI model", value="gpt-4o-mini")

# Pre-fill from env var / Streamlit secrets if available; otherwise the
# visitor enters their own key.
_prefilled_key = os.getenv("OPENAI_API_KEY", "")
try:
    _prefilled_key = st.secrets.get("OPENAI_API_KEY", _prefilled_key)
except Exception:
    pass

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    value=_prefilled_key,
    type="password",
    help="Kept only in this browser session; never written to disk.",
)

max_workers = st.sidebar.slider(
    "Parallel workers", min_value=1, max_value=8, value=4,
    help="Documents processed concurrently. Higher is faster but sends "
         "more concurrent API requests.",
)

st.sidebar.markdown("---")
st.sidebar.caption("🔑 Get an API key at platform.openai.com/api-keys")

# --------------------------------------------------------------------
# Header
# --------------------------------------------------------------------
st.title("📋 AI Customer Complaint & Case Processing System")
st.caption(
    "Batch-processes complaint documents through a 3-stage LLM workflow: "
    "structured extraction → customer email → internal case summary."
)

# --------------------------------------------------------------------
# Step 1: document source
# --------------------------------------------------------------------
st.subheader("1. Provide documents")

tab_upload, tab_existing = st.tabs(["📤 Upload files", "📁 Use data/ folder"])

with tab_upload:
    uploaded_files = st.file_uploader(
        "Upload complaint documents (.txt, .pdf, .docx)",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for uploaded_file in uploaded_files:
            target_path = DATA_DIR / uploaded_file.name
            with open(target_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        logger.info("Saved %d uploaded file(s) to %s", len(uploaded_files), DATA_DIR)
        st.success(f"Saved {len(uploaded_files)} file(s) to {DATA_DIR}/")

with tab_existing:
    existing_files = sorted(
        p.name for p in DATA_DIR.glob("*")
        if p.suffix.lower() in {".txt", ".pdf", ".docx"}
    )
    if existing_files:
        st.write(f"Found **{len(existing_files)}** document(s) in `data/`:")
        st.code("\n".join(existing_files))
    else:
        st.info("No documents found in data/ yet. Upload some in the other tab.")

# --------------------------------------------------------------------
# Step 2: run the pipeline
# --------------------------------------------------------------------
st.subheader("2. Run the workflow")

run_col, clear_col = st.columns(2)
run_clicked = run_col.button("🚀 Process all documents", type="primary")
clear_clicked = clear_col.button("🗑️ Clear previous outputs")

if clear_clicked:
    for folder in (STRUCTURED_DATA_DIR, CUSTOMER_EMAILS_DIR, CASE_SUMMARIES_DIR):
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True, exist_ok=True)
    if FINAL_REPORT_PATH.exists():
        FINAL_REPORT_PATH.unlink()
    st.session_state.pop("records", None)
    st.success("Cleared previous outputs.")

if run_clicked:
    if not api_key:
        st.error("Please provide your OpenAI API key in the sidebar.")
    else:
        client = None
        try:
            client = LLMClient(model=model_name, openai_api_key=api_key)
        except LLMError as e:
            st.error(f"Could not initialise LLM client: {e}")

        if client:
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def _update_progress_ui(done: int, total: int, current_file: str) -> None:
                progress_bar.progress(done / total)
                status_text.text(f"Processed {done}/{total}: {current_file}")

            with st.spinner("Running document ingestion, extraction, and generation..."):
                records = run_batch(
                    client, data_dir=DATA_DIR, max_workers=max_workers,
                    progress_callback=_update_progress_ui,
                )

            st.session_state["records"] = records

            if records:
                succeeded = sum(1 for r in records if r.processing_status == "Success")
                st.success(f"Finished: {succeeded}/{len(records)} document(s) processed successfully.")
            else:
                st.warning("No eligible documents were found to process.")

# --------------------------------------------------------------------
# Step 3 & 4: results
# --------------------------------------------------------------------
records: list[CaseRecord] = st.session_state.get("records", [])

if records:
    st.subheader("3. Results")

    results_dataframe = pd.DataFrame([r.to_report_row() for r in records])
    st.dataframe(results_dataframe, use_container_width=True)

    if FINAL_REPORT_PATH.exists():
        with open(FINAL_REPORT_PATH, "rb") as f:
            st.download_button(
                "⬇️ Download final_report.csv", f, file_name="final_report.csv",
                mime="text/csv",
            )

    st.markdown("---")
    st.subheader("4. Browse individual cases")

    case_file_names = [r.source_file for r in records]
    selected_file_name = st.selectbox("Select a document", case_file_names)
    selected_record = next(r for r in records if r.source_file == selected_file_name)

    if selected_record.processing_status == "Failed":
        st.error(f"Processing failed for this file: {selected_record.error_message}")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🧩 Structured Data**")
            st.json(selected_record.extracted.model_dump())

        with col2:
            st.markdown("**✉️ Customer Email**")
            st.text_area("Email", selected_record.customer_email_text,
                         height=320, label_visibility="collapsed")

        with col3:
            st.markdown("**🗂️ Internal Case Summary**")
            st.text_area("Summary", selected_record.case_summary_text,
                         height=320, label_visibility="collapsed")
else:
    st.info("Provide documents above and click **Process all documents** to begin.")

st.markdown("---")
st.caption("AI Customer Complaint & Case Processing System · Python, Pydantic, OpenAI, Streamlit")
