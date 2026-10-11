# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pydantic schemas, PDF extraction utilities, and external dispatch tools for the AI in 5 Days Content Customizer Agent."""

from __future__ import annotations

import datetime
import gzip
import json
import logging
import os
import re
import smtplib
import urllib.request
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

VALID_DAYS = ("day1", "day2", "day3", "day4", "day5")
DEFAULT_WORKSPACE_ROOT = os.environ.get("CONTENT_PDFS_DIR", "./content_pdfs")


# ---------------------------------------------------------------------------
# Structured Observability & Telemetry Logging
# ---------------------------------------------------------------------------


def emit_telemetry_log(
    severity: str,
    node: str,
    event_type: str,
    message: str,
    **fields: Any,
) -> dict[str, Any]:
    """Emit a structured JSON log entry compatible with Google Cloud Logging and Cloud Trace."""
    entry: dict[str, Any] = {
        "severity": severity.upper(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "agent": "ai_in_5_days_agent",
        "node": node,
        "event_type": event_type,
        "message": message,
        **fields,
    }
    print(json.dumps(entry, default=str), flush=True)
    return entry


# ---------------------------------------------------------------------------
# Strict Pydantic Schemas for Node I/O & Workflow State
# ---------------------------------------------------------------------------


class UserProfile(BaseModel):
    """Learner profile collected during session setup."""

    user_name: str = Field(description="Full name of the learner")
    user_email: str = Field(description="Target email address for daily brief delivery")
    technical_track: str = Field(
        description="Preferred technical track, e.g., Engineer or Architect"
    )
    target_day: str = Field(
        default="day1",
        description="Target training day folder (day1 through day5)",
    )


class WorkspaceStatus(BaseModel):
    """Status of the local ./content_pdfs/day1..day5 workspace."""

    workspace_root: str = Field(description="Root directory for training PDFs")
    created_days: list[str] = Field(
        description="List of day directories ensured in the workspace"
    )
    target_day: str = Field(description="Selected training day (day1..day5)")
    target_day_dir: str = Field(description="Path to the selected day's folder")
    pdf_files: list[str] = Field(
        description="PDF filenames discovered in the target day directory"
    )
    seeded_sample_pdf: bool = Field(
        description="Whether a starter training PDF was auto-seeded"
    )
    guidance_message: str = Field(
        description="User-facing guidance on where to drop custom training PDFs"
    )
    user_name: str = Field(default="", description="Learner full name")
    user_email: str = Field(default="", description="Learner target email")
    technical_track: str = Field(default="", description="Learner technical track")


class ExtractedPdfContent(BaseModel):
    """Strict Pydantic schema for PDF text and metadata extracted via pypdf."""

    day: str = Field(description="Target training day (day1..day5)")
    file_path: str = Field(description="Path to the extracted PDF file")
    file_name: str = Field(description="Filename of the extracted PDF")
    page_count: int = Field(ge=1, description="Number of pages in the PDF")
    extracted_text: str = Field(
        min_length=1, description="Full text extracted across all PDF pages"
    )
    char_count: int = Field(ge=1, description="Character count of extracted text")
    metadata: dict[str, str] = Field(
        default_factory=dict, description="PDF document metadata (Title, Author, etc.)"
    )
    user_name: str = Field(default="", description="Learner full name")
    user_email: str = Field(default="", description="Learner target email")
    technical_track: str = Field(default="", description="Learner technical track")


class SummarizerInput(BaseModel):
    """Structured input schema passed from pdf_reader into content_summarizer."""

    user_name: str = Field(description="Full name of the learner")
    user_email: str = Field(description="Target email address of the learner")
    technical_track: str = Field(
        description="Preferred technical track (Engineer or Architect)"
    )
    target_day: str = Field(description="Training day identifier (day1..day5)")
    pdf_file_name: str = Field(description="Source PDF filename")
    page_count: int = Field(description="Number of pages in the source PDF")
    raw_content: str = Field(description="Raw training text extracted from the PDF")


class DailyBrief(BaseModel):
    """Structured daily brief generated by content_summarizer and reviewed by human_review."""

    day: str = Field(description="Training day identifier (e.g., day1)")
    technical_track: str = Field(
        description="Target technical track (Engineer or Architect)"
    )
    user_name: str = Field(description="Learner's name")
    user_email: str = Field(description="Learner's target email address")
    title: str = Field(description="Engaging headline for the personalized daily brief")
    executive_summary: str = Field(
        description="Track-personalized synthesis of the day's PDF content"
    )
    key_takeaways: list[str] = Field(
        description="3 to 5 actionable takeaways tailored to the learner's track"
    )
    curiosity_hooks: list[str] = Field(
        description="Thought-provoking curiosity hooks connecting concepts to real-world systems"
    )
    micro_challenge: str = Field(
        description="A concrete 15-minute hands-on micro-challenge for the learner's track"
    )


class DispatchReceipt(BaseModel):
    """Structured confirmation receipt produced after dispatching or cancelling a brief."""

    status: str = Field(description="Overall outcome: dispatched or cancelled")
    user_name: str = Field(description="Learner name")
    user_email: str = Field(description="Target email address")
    technical_track: str = Field(description="Technical track")
    day: str = Field(description="Training day")
    email_status: str = Field(description="Email delivery status")
    chat_webhook_status: str = Field(description="Google Chat webhook delivery status")
    summary_title: str = Field(description="Title of the brief")
    message: str = Field(description="Human-readable summary of the dispatch outcome")


class WorkflowState(BaseModel):
    """Typed session state schema for the AI in 5 Days Workflow."""

    user_name: str = Field(default="", description="Persisted learner name")
    user_email: str = Field(default="", description="Persisted learner email")
    technical_track: str = Field(
        default="", description="Persisted technical track (Engineer or Architect)"
    )
    target_day: str = Field(default="day1", description="Current target training day")
    pending_interrupt: str = Field(
        default="", description="Active HITL interrupt_id awaiting user input"
    )
    pending_review_decision: str = Field(
        default="", description="Pending human review decision if supplied via text"
    )
    auto_confirm: bool = Field(
        default=False,
        description="Whether human review confirmation was pre-authorized",
    )
    preferred_language: str = Field(
        default="English", description="Optional user preferred language"
    )
    visit_count: int = Field(
        default=1, description="Optional session visit counter"
    )
    workspace_status: dict[str, Any] = Field(
        default_factory=dict, description="Latest workspace initialization status"
    )
    extracted_pdf: dict[str, Any] = Field(
        default_factory=dict, description="Latest extracted PDF payload"
    )
    daily_brief: dict[str, Any] = Field(
        default_factory=dict, description="Latest drafted DailyBrief dictionary"
    )
    review_decision: str = Field(
        default="", description="Human review decision (approved or rejected)"
    )
    dispatch_receipt: dict[str, Any] = Field(
        default_factory=dict, description="Latest dispatch receipt dictionary"
    )
    completed_days: list[str] = Field(
        default_factory=list, description="List of days completed in this session"
    )


# ---------------------------------------------------------------------------
# Pure-Python Starter PDF Generator & Workspace Helpers
# ---------------------------------------------------------------------------

STARTER_CURRICULUM: dict[str, tuple[str, list[str]]] = {
    "day1": (
        "AI in 5 Days - Day 1: Tool & Interface Design for Agentic Systems",
        [
            "AI in 5 Days Training Curriculum - Day 1: Tool & Interface Design",
            "Module 1: Deterministic Boundaries and Pydantic Contracts",
            "Production AI agents separate deterministic code from LLM reasoning.",
            "Every tool interface must declare explicit parameter types, clear docstrings,",
            "and strict Pydantic input/output schemas so downstream nodes never guess shapes.",
            "For Engineers: Focus on idempotent tool functions, retryable HTTP clients,",
            "structured error payloads, and zero-default-arg signatures for ADK FunctionTools.",
            "For Architects: Focus on API gateway boundaries, least-privilege IAM,",
            "service-to-service contracts, and decoupling external side effects behind review gates.",
            "Key Pattern: Validate inputs at the boundary with BaseModel before invoking models.",
        ],
    ),
    "day2": (
        "AI in 5 Days - Day 2: Context Engineering & Session Memory",
        [
            "AI in 5 Days Training Curriculum - Day 2: Context & Memory Architecture",
            "Module 2: Short-Term Session State vs. Long-Term Memory Bank",
            "Effective agents maintain continuity across turns without overflowing context windows.",
            "In ADK 2.0, Workflow state_schema and Event(state=...) persist replayable state deltas.",
            "For Engineers: Implement typed state schemas, token-based context compaction,",
            "and PreloadMemoryTool callbacks to hydrate user preferences automatically.",
            "For Architects: Design multi-tenant session storage using Agent Platform Sessions",
            "or Cloud SQL, balancing retention policies, privacy boundaries, and cache TTLs.",
        ],
    ),
    "day3": (
        "AI in 5 Days - Day 3: Graph Orchestration & Human-in-the-Loop",
        [
            "AI in 5 Days Training Curriculum - Day 3: ADK 2.0 Graph Workflows & HITL",
            "Module 3: Deterministic Routing, Parallel Fan-Out, and RequestInput Gates",
            "ADK 2.0 introduces graph-based Workflows combining FunctionNodes and LlmAgents.",
            "High-consequence actions require explicit Human-in-the-Loop pauses via RequestInput.",
            "For Engineers: Use @node(rerun_on_resume=True), inspect ctx.resume_inputs,",
            "and emit Event(route='APPROVED' | 'REJECTED') for conditional edge branching.",
            "For Architects: Orchestrate multi-stage pipelines where LLMs draft artifacts",
            "and human reviewers hold cryptographic or session-bound approval authority.",
        ],
    ),
    "day4": (
        "AI in 5 Days - Day 4: Observability, Cloud Trace & Quality Evaluation",
        [
            "AI in 5 Days Training Curriculum - Day 4: Observability & The Quality Flywheel",
            "Module 4: Structured Telemetry, OpenTelemetry Spans, and LLM-as-Judge Eval",
            "You cannot improve an agent you cannot measure in production.",
            "Emit structured JSON logs with severity, node, and trace correlation fields.",
            "For Engineers: Instrument nodes with Cloud Trace (otel_to_cloud) and run",
            "agents-cli eval with trajectory and response quality rubrics before every release.",
            "For Architects: Establish continuous evaluation gates and BigQuery Agent Analytics",
            "dashboards to monitor latency, token budgets, and safety compliance at scale.",
        ],
    ),
    "day5": (
        "AI in 5 Days - Day 5: Production Infrastructure, CI/CD & A2A Interop",
        [
            "AI in 5 Days Training Curriculum - Day 5: Enterprise Deployment & CI/CD",
            "Module 5: Vertex AI Agent Runtime, Terraform IaC, and A2A Protocol",
            "Moving from local prototype to enterprise production requires reproducible infra.",
            "Expose your agent over the Agent-to-Agent (A2A) protocol for fleet composition.",
            "For Engineers: Containerize with multi-stage Dockerfiles, pin dependencies with uv,",
            "and automate pytest + eval gates inside Cloud Build or GitHub Actions.",
            "For Architects: Select between Vertex AI Agent Runtime, Cloud Run, and GKE,",
            "and register production endpoints in Google Cloud Agent Registry.",
        ],
    ),
}


def build_simple_pdf_bytes(
    lines: list[str],
    title: str,
    pages: list[str] | None = None,
) -> bytes:
    """Create a valid multi-page PDF byte stream readable by pypdf.PdfReader."""

    def _escape_pdf_text(text: str) -> str:
        clean = (
            text.replace("\u2013", "-")
            .replace("\u2014", "--")
            .replace("\u2018", "'")
            .replace("\u2019", "'")
            .replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2022", "*")
            .replace("\u2192", "->")
        )
        clean = clean.encode("latin-1", errors="replace").decode("latin-1")
        return (
            clean.replace("\\", "\\\\")
            .replace("(", "\\(")
            .replace(")", "\\)")
        )

    if pages:
        pages_lines = [
            [ln.strip() for ln in page_text.splitlines() if ln.strip()] or [title]
            for page_text in pages
        ]
    else:
        # Split lines into chunks of 4 lines per page so starter PDFs are multi-page documents
        chunk_size = 4
        pages_lines = [
            lines[i : i + chunk_size] for i in range(0, max(1, len(lines)), chunk_size)
        ]
    num_pages = len(pages_lines)

    # Object IDs:
    # 1: Catalog, 2: Pages, 3: Font, 4: Info
    # Then for each page p (0..num_pages-1):
    #   page_obj_id = 5 + 2*p
    #   content_obj_id = 6 + 2*p
    kids_refs = " ".join(f"{5 + 2 * p} 0 R" for p in range(num_pages))
    objects: list[bytes] = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        f"2 0 obj\n<< /Type /Pages /Kids [{kids_refs}] /Count {num_pages} >>\nendobj\n".encode(
            "ascii"
        ),
        b"3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        (
            f"4 0 obj\n<< /Title ({_escape_pdf_text(title)}) "
            f"/Author (AI in 5 Days Training Material) /Subject (ADK 2.0 Multi-Page Courseware) >>\nendobj\n"
        ).encode("latin-1", errors="replace"),
    ]

    for p_idx, p_lines in enumerate(pages_lines):
        page_obj_id = 5 + 2 * p_idx
        content_obj_id = 6 + 2 * p_idx
        ops = ["BT", "/F1 10 Tf", "36 756 Td", "13 TL"]
        ops.append(f"(Page {p_idx + 1} of {num_pages} - {_escape_pdf_text(title)}) Tj T*")
        for line in p_lines:
            ops.append(f"({_escape_pdf_text(line)}) Tj T*")
        ops.append("ET")
        stream_bytes = "\n".join(ops).encode("latin-1", errors="replace")

        page_obj = (
            f"{page_obj_id} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_obj_id} 0 R /Resources << /Font << /F1 3 0 R >> >> >>\nendobj\n"
        ).encode("ascii")
        content_obj = (
            f"{content_obj_id} 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n".encode(
                "ascii"
            )
            + stream_bytes
            + b"\nendstream\nendobj\n"
        )
        objects.append(page_obj)
        objects.append(content_obj)

    output = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for obj in objects:
        offsets.append(len(output))
        output.extend(obj)

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info 4 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(output)


def normalize_day(day_value: str) -> str:
    """Normalize a user-supplied day string into 'day1'..'day5'."""
    cleaned = str(day_value or "day1").strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    if cleaned in VALID_DAYS:
        return cleaned
    for d in ("1", "2", "3", "4", "5"):
        if d in cleaned:
            return f"day{d}"
    return "day1"


def normalize_track(track_value: str) -> str:
    """Normalize a user-supplied track into 'Engineer' or 'Architect' (or title-cased custom track)."""
    cleaned = str(track_value or "").strip()
    lower = cleaned.lower()
    if "arch" in lower:
        return "Architect"
    if "eng" in lower or "dev" in lower:
        return "Engineer"
    return cleaned.title() if cleaned else "Engineer"


def _load_bundled_corpus() -> dict[str, dict[str, Any]]:
    """Load the compressed multi-page training corpus if bundled alongside app/tools.py."""
    corpus_path = Path(__file__).resolve().parent / "training_corpus.json.gz"
    if corpus_path.exists():
        try:
            with gzip.open(corpus_path, "rt", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def initialize_pdf_workspace(
    target_day: str,
    workspace_root: str = DEFAULT_WORKSPACE_ROOT,
    user_name: str = "",
    user_email: str = "",
    technical_track: str = "",
) -> WorkspaceStatus:
    """Ensure ./content_pdfs/day1 through day5 exist and seed multi-page starter PDFs if empty."""
    normalized_day = normalize_day(target_day)
    root_path = Path(workspace_root)
    created_days: list[str] = []
    seeded_for_target = False
    bundled_corpus = _load_bundled_corpus()

    for day_key in VALID_DAYS:
        day_dir = root_path / day_key
        day_dir.mkdir(parents=True, exist_ok=True)
        created_days.append(day_key)

        existing_pdfs = sorted(p.name for p in day_dir.glob("*.pdf"))
        if not existing_pdfs:
            if day_key in bundled_corpus and bundled_corpus[day_key].get("pages"):
                entry = bundled_corpus[day_key]
                sample_path = day_dir / str(entry.get("file_name") or f"{day_key}_training_guide.pdf")
                sample_path.write_bytes(
                    build_simple_pdf_bytes(
                        lines=[],
                        title=str(entry.get("title") or day_key.upper()),
                        pages=list(entry["pages"]),
                    )
                )
            else:
                title, lines = STARTER_CURRICULUM[day_key]
                sample_path = day_dir / f"{day_key}_training_guide.pdf"
                sample_path.write_bytes(build_simple_pdf_bytes(lines=lines, title=title))
            if day_key == normalized_day:
                seeded_for_target = True

    target_dir = root_path / normalized_day
    pdf_files = sorted(p.name for p in target_dir.glob("*.pdf"))
    guidance = (
        f"Workspace initialized at '{root_path}' with folders {', '.join(created_days)}. "
        f"Drop your multi-page training material PDFs into '{target_dir}' (active file(s): {', '.join(pdf_files) if pdf_files else 'none'})."
    )

    status = WorkspaceStatus(
        workspace_root=str(root_path),
        created_days=created_days,
        target_day=normalized_day,
        target_day_dir=str(target_dir),
        pdf_files=pdf_files,
        seeded_sample_pdf=seeded_for_target,
        guidance_message=guidance,
        user_name=user_name,
        user_email=user_email,
        technical_track=technical_track,
    )
    emit_telemetry_log(
        severity="INFO",
        node="init_workspace",
        event_type="workspace_initialized",
        message=guidance,
        target_day=normalized_day,
        pdf_files=pdf_files,
        seeded_sample_pdf=seeded_for_target,
    )
    return status


def extract_pdf_from_workspace(
    target_day: str,
    workspace_root: str = DEFAULT_WORKSPACE_ROOT,
    user_name: str = "",
    user_email: str = "",
    technical_track: str = "",
) -> ExtractedPdfContent:
    """Discover and extract multi-page training material text from ./content_pdfs/<target_day> using pypdf."""
    normalized_day = normalize_day(target_day)
    day_dir = Path(workspace_root) / normalized_day
    if not day_dir.exists() or not list(day_dir.glob("*.pdf")):
        initialize_pdf_workspace(
            target_day=normalized_day,
            workspace_root=workspace_root,
            user_name=user_name,
            user_email=user_email,
            technical_track=technical_track,
        )

    all_pdfs = sorted(day_dir.glob("*.pdf"))
    if not all_pdfs:
        raise FileNotFoundError(f"No PDF files found in '{day_dir}'.")

    starter_name = f"{normalized_day}_training_guide.pdf"
    custom_pdfs = [p for p in all_pdfs if p.name != starter_name]
    target_pdfs = custom_pdfs if custom_pdfs else all_pdfs

    pages_text: list[str] = []
    total_pages = 0
    clean_meta: dict[str, str] = {}
    for pdf_path in target_pdfs:
        reader = PdfReader(str(pdf_path))
        doc_pages = len(reader.pages)
        total_pages += max(1, doc_pages)
        if not clean_meta and reader.metadata:
            clean_meta = {
                str(k).lstrip("/"): str(v)
                for k, v in reader.metadata.items()
                if v is not None
            }
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if page_text:
                header = (
                    f"[Document: {pdf_path.name} | Page {page_num} of {doc_pages}]\n"
                    if len(target_pdfs) > 1 or doc_pages > 1
                    else ""
                )
                pages_text.append(f"{header}{page_text}")

    primary_pdf = target_pdfs[0]
    combined_text = "\n\n".join(pages_text).strip()
    if not combined_text:
        # Fallback if a user dropped a scanned/image-only PDF without extractable glyphs
        title, lines = STARTER_CURRICULUM.get(normalized_day, STARTER_CURRICULUM["day1"])
        combined_text = f"{title}\n" + "\n".join(lines)

    # Cap at 120,000 chars so very large multi-page PDFs fit comfortably in context
    if len(combined_text) > 120_000:
        combined_text = combined_text[:120_000] + "\n...[truncated after 120,000 chars]"

    extracted = ExtractedPdfContent(
        day=normalized_day,
        file_path=str(primary_pdf),
        file_name=primary_pdf.name if len(target_pdfs) == 1 else ", ".join(p.name for p in target_pdfs),
        page_count=max(1, total_pages),
        extracted_text=combined_text,
        char_count=len(combined_text),
        metadata=clean_meta,
        user_name=user_name,
        user_email=user_email,
        technical_track=technical_track,
    )
    emit_telemetry_log(
        severity="INFO",
        node="pdf_reader",
        event_type="pdf_extracted",
        message=f"Extracted {extracted.char_count} chars across {extracted.page_count} page(s) from {extracted.file_name}",
        day=normalized_day,
        file_name=extracted.file_name,
        page_count=extracted.page_count,
        char_count=extracted.char_count,
    )
    return extracted


# ---------------------------------------------------------------------------
# External Dispatch Tools (Email & Google Chat Card v2 Webhook)
# ---------------------------------------------------------------------------


def send_email_brief(
    recipient_email: str,
    user_name: str,
    day: str,
    track: str,
    subject: str,
    body_markdown: str,
) -> dict[str, Any]:
    """Send the confirmed daily brief to the learner's target email address.

    Args:
        recipient_email: Target email address of the learner.
        user_name: Full name of the learner.
        day: Training day identifier (e.g., day1).
        track: Learner's preferred technical track (Engineer or Architect).
        subject: Email subject line for the daily brief.
        body_markdown: Full formatted daily brief body content.

    Returns:
        A dictionary containing delivery status, recipient, subject, and transport mode.
    """
    smtp_host = os.environ.get("SMTP_HOST", "").strip()
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER", "").strip()
    smtp_password = os.environ.get("SMTP_PASSWORD", "").strip()
    sender_email = os.environ.get("SMTP_FROM", smtp_user or "ai-in-5-days@google.com")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg.set_content(body_markdown)

    # Always save a local copy in ./dispatched_briefs/ so local prototype testing can inspect the exact email
    outbox_dir = Path(os.environ.get("DISPATCHED_BRIEFS_DIR", "./dispatched_briefs"))
    outbox_dir.mkdir(parents=True, exist_ok=True)
    safe_email = re.sub(r"[^a-zA-Z0-9_.-]", "_", recipient_email)
    md_path = outbox_dir / f"{day}_email_{safe_email}.md"
    eml_path = outbox_dir / f"{day}_email_{safe_email}.eml"
    md_path.write_text(
        f"To: {recipient_email}\nFrom: {sender_email}\nSubject: {subject}\nDay: {day.upper()} ({track} Track)\n\n---\n\n{body_markdown}\n",
        encoding="utf-8",
    )
    eml_path.write_bytes(bytes(msg))

    delivery_mode = "dry_run_saved_to_outbox"
    if smtp_host:
        try:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
                if smtp_user and smtp_password:
                    server.starttls()
                    server.login(smtp_user, smtp_password)
                server.send_message(msg)
            delivery_mode = "smtp_sent"
        except Exception as exc:
            delivery_mode = f"dry_run_fallback ({type(exc).__name__})"

    result = {
        "status": "sent",
        "delivery_mode": delivery_mode,
        "recipient_email": recipient_email,
        "user_name": user_name,
        "day": day,
        "track": track,
        "subject": subject,
        "saved_email_path": str(md_path),
    }
    emit_telemetry_log(
        severity="INFO",
        node="dispatch_brief",
        event_type="email_dispatched",
        message=f"Daily brief email dispatched to {recipient_email} ({delivery_mode}, saved={md_path})",
        **result,
    )
    return result


def post_google_chat_card(
    user_name: str,
    day: str,
    track: str,
    title: str,
    summary: str,
    curiosity_hook: str,
    micro_challenge: str,
) -> dict[str, Any]:
    """Post an interactive Google Chat Card v2 payload to a Google Chat Space webhook.

    Args:
        user_name: Full name of the learner.
        day: Training day identifier (e.g., day1).
        track: Learner's preferred technical track (Engineer or Architect).
        title: Headline title of the daily brief.
        summary: Executive summary personalized to the learner's track.
        curiosity_hook: Curiosity hook question or insight from the brief.
        micro_challenge: Concrete hands-on micro-challenge for the day.

    Returns:
        A dictionary containing the webhook delivery status and Card v2 metadata.
    """
    card_payload = {
        "cardsV2": [
            {
                "cardId": f"ai-in-5-days-{day}-{track.lower()}",
                "card": {
                    "header": {
                        "title": title,
                        "subtitle": f"AI in 5 Days ({day.upper()}) • {track} Track • For {user_name}",
                    },
                    "sections": [
                        {
                            "header": "Executive Summary",
                            "widgets": [
                                {"textParagraph": {"text": summary}},
                            ],
                        },
                        {
                            "header": "Curiosity Hook & Micro-Challenge",
                            "widgets": [
                                {
                                    "decoratedText": {
                                        "topLabel": "Curiosity Hook",
                                        "text": curiosity_hook,
                                        "wrapText": True,
                                    }
                                },
                                {
                                    "decoratedText": {
                                        "topLabel": f"{track} Micro-Challenge",
                                        "text": micro_challenge,
                                        "wrapText": True,
                                    }
                                },
                            ],
                        },
                    ],
                },
            }
        ]
    }

    outbox_dir = Path(os.environ.get("DISPATCHED_BRIEFS_DIR", "./dispatched_briefs"))
    outbox_dir.mkdir(parents=True, exist_ok=True)
    card_path = outbox_dir / f"{day}_google_chat_card_{track.lower()}.json"
    card_path.write_text(json.dumps(card_payload, indent=2), encoding="utf-8")

    webhook_url = os.environ.get("GOOGLE_CHAT_WEBHOOK_URL", "").strip()
    delivery_mode = "dry_run_card_v2_saved"
    if webhook_url and webhook_url.startswith("https://"):
        try:
            req = urllib.request.Request(
                webhook_url,
                data=json.dumps(card_payload).encode("utf-8"),
                headers={"Content-Type": "application/json; charset=UTF-8"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                delivery_mode = f"webhook_posted_http_{resp.status}"
        except Exception as exc:
            delivery_mode = f"dry_run_fallback ({type(exc).__name__})"

    result = {
        "status": "posted",
        "delivery_mode": delivery_mode,
        "card_id": f"ai-in-5-days-{day}-{track.lower()}",
        "user_name": user_name,
        "day": day,
        "track": track,
        "title": title,
        "saved_card_path": str(card_path),
        "card_v2": card_payload,
    }
    emit_telemetry_log(
        severity="INFO",
        node="dispatch_brief",
        event_type="google_chat_card_dispatched",
        message=f"Google Chat Card v2 dispatched for {user_name} ({day}, {track} track, mode={delivery_mode})",
        status="posted",
        delivery_mode=delivery_mode,
        card_id=result["card_id"],
        saved_card_path=str(card_path),
    )
    return result
