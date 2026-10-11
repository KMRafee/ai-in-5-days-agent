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

"""Pydantic schemas, PII redaction, semantic compaction, async memory consolidation, and tools for the AI in 5 Days Agent."""

from __future__ import annotations

import asyncio
import datetime
import gzip
import hashlib
import json
import logging
import os
import re
import smtplib
import time
import urllib.request
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from opentelemetry import trace
from pydantic import BaseModel, Field
from pypdf import PdfReader

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

VALID_DAYS = ("day1", "day2", "day3", "day4", "day5")
DEFAULT_WORKSPACE_ROOT = os.environ.get("CONTENT_PDFS_DIR", "./content_pdfs")

# Track active background memory consolidation tasks so they are not garbage-collected mid-flight
_BACKGROUND_MEMORY_TASKS: set[asyncio.Task[Any]] = set()
_CONSOLIDATED_MEMORY_STORE: dict[str, list[dict[str, Any]]] = {}

# Regex patterns for automatic PII redaction across logs and OpenTelemetry traces
_EMAIL_PII_RE = re.compile(r"\b([a-zA-Z0-9_.+-])[a-zA-Z0-9_.+-]*@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)\b")
_SECRET_TOKEN_RE = re.compile(r"\b(ghp_[A-Za-z0-9]{20,}|ya29\.[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{30,})\b")
_PHONE_PII_RE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
_NAME_FIELD_KEYS = frozenset({"user_name", "learner_name", "author_name", "full_name", "recipient_name"})
_EMAIL_FIELD_KEYS = frozenset({"user_email", "recipient_email", "email", "sender_email"})
_SECRET_FIELD_KEYS = frozenset({"password", "smtp_password", "api_key", "token", "secret", "authorization"})


# ---------------------------------------------------------------------------
# PII Redaction & Masking Engine for Logs & OpenTelemetry Spans
# ---------------------------------------------------------------------------


def mask_email_pii(email_value: str) -> str:
    """Mask an email address while preserving domain for operational debugging (e.g., k***@google.com)."""
    cleaned = str(email_value or "").strip()
    if not cleaned or "@" not in cleaned:
        return "[REDACTED_EMAIL]" if cleaned else ""
    local_part, domain_part = cleaned.split("@", 1)
    prefix = local_part[0] if local_part else "u"
    digest = hashlib.sha256(cleaned.lower().encode("utf-8")).hexdigest()[:6]
    return f"{prefix}***@{domain_part} (pii_hash={digest})"


def mask_name_pii(name_value: str) -> str:
    """Mask a person's full name into initials (e.g., 'Rafee Mohamed' -> 'R*** M***')."""
    cleaned = str(name_value or "").strip()
    if not cleaned:
        return ""
    parts = [p for p in cleaned.split() if p]
    return " ".join(f"{p[0]}***" for p in parts)


def redact_pii_text(text: str, known_names: list[str] | None = None) -> str:
    """Redact emails, tokens, phone numbers, and known learner names from free-form log strings."""
    if not text:
        return ""
    redacted = _EMAIL_PII_RE.sub(r"\1***@\2", str(text))
    redacted = _SECRET_TOKEN_RE.sub("[REDACTED_SECRET_TOKEN]", redacted)
    redacted = _PHONE_PII_RE.sub("[REDACTED_PHONE]", redacted)
    for name in known_names or []:
        clean_name = str(name or "").strip()
        if len(clean_name) >= 3:
            redacted = re.sub(
                re.escape(clean_name),
                mask_name_pii(clean_name),
                redacted,
                flags=re.IGNORECASE,
            )
    return redacted


def redact_pii_payload(payload: Any, known_names: list[str] | None = None) -> Any:
    """Recursively sanitize dictionaries, lists, and strings to remove plain-text PII before logging."""
    detected_names = list(known_names or [])
    if isinstance(payload, dict):
        for k, v in payload.items():
            if str(k).lower() in _NAME_FIELD_KEYS and isinstance(v, str) and v.strip():
                detected_names.append(v.strip())

        sanitized: dict[str, Any] = {}
        for key, value in payload.items():
            key_lower = str(key).lower()
            if key_lower in _SECRET_FIELD_KEYS:
                sanitized[key] = "[REDACTED_SECRET]"
            elif key_lower in _EMAIL_FIELD_KEYS and isinstance(value, str):
                sanitized[key] = mask_email_pii(value)
            elif key_lower in _NAME_FIELD_KEYS and isinstance(value, str):
                sanitized[key] = mask_name_pii(value)
            else:
                sanitized[key] = redact_pii_payload(value, known_names=detected_names)
        return sanitized
    if isinstance(payload, list):
        return [redact_pii_payload(item, known_names=detected_names) for item in payload]
    if isinstance(payload, str):
        return redact_pii_text(payload, known_names=detected_names)
    return payload


class PiiRedactingLogFilter(logging.Filter):
    """Standard library logging.Filter that scrubs PII from all log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_pii_text(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = redact_pii_payload(record.args)
            elif isinstance(record.args, tuple):
                record.args = tuple(redact_pii_payload(a) for a in record.args)
        return True


logger.addFilter(PiiRedactingLogFilter())


# ---------------------------------------------------------------------------
# Structured Observability, Intent vs. Outcome Auditing & Telemetry Logging
# ---------------------------------------------------------------------------


def emit_telemetry_log(
    severity: str,
    node: str,
    event_type: str,
    message: str,
    **fields: Any,
) -> dict[str, Any]:
    """Emit a PII-redacted structured JSON log entry correlated with OpenTelemetry spans."""
    known_names: list[str] = []
    for k in _NAME_FIELD_KEYS:
        if isinstance(fields.get(k), str) and fields[k].strip():
            known_names.append(fields[k].strip())

    sanitized_fields = redact_pii_payload(fields, known_names=known_names)
    sanitized_message = redact_pii_text(message, known_names=known_names)

    current_span = trace.get_current_span()
    span_ctx = current_span.get_span_context() if current_span else None
    trace_id = (
        format(span_ctx.trace_id, "032x")
        if span_ctx and getattr(span_ctx, "is_valid", False)
        else ""
    )
    span_id = (
        format(span_ctx.span_id, "016x")
        if span_ctx and getattr(span_ctx, "is_valid", False)
        else ""
    )

    entry: dict[str, Any] = {
        "severity": severity.upper(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "agent": "ai_in_5_days_agent",
        "node": node,
        "event_type": event_type,
        "message": sanitized_message,
        "pii_redacted": True,
        "trace_id": trace_id,
        "span_id": span_id,
        **sanitized_fields,
    }
    print(json.dumps(entry, default=str), flush=True)
    return entry


def log_tool_intent_before(
    tool_name: str,
    declared_intent: str,
    tool_args: dict[str, Any],
    node: str = "tool_execution",
) -> dict[str, Any]:
    """Capture explicit tool-level BEFORE log with declared intent and PII-redacted arguments."""
    return emit_telemetry_log(
        severity="INFO",
        node=node,
        event_type="tool_execution_before_intent",
        message=f"Tool '{tool_name}' invoked with intent: {declared_intent}",
        tool_name=tool_name,
        declared_intent=declared_intent,
        lifecycle_stage="before_tool",
        sanitized_tool_args=redact_pii_payload(tool_args),
    )


def log_tool_outcome_after(
    tool_name: str,
    declared_intent: str,
    actual_outcome: str,
    intent_vs_outcome_match: bool,
    latency_ms: float,
    result_summary: dict[str, Any],
    node: str = "tool_execution",
) -> dict[str, Any]:
    """Capture explicit tool-level AFTER log auditing declared intent vs. actual outcome."""
    return emit_telemetry_log(
        severity="INFO" if intent_vs_outcome_match else "WARNING",
        node=node,
        event_type="tool_execution_after_outcome",
        message=(
            f"Tool '{tool_name}' completed in {latency_ms:.2f}ms: "
            f"intent='{declared_intent}' -> outcome='{actual_outcome}' (match={intent_vs_outcome_match})"
        ),
        tool_name=tool_name,
        declared_intent=declared_intent,
        actual_outcome=actual_outcome,
        intent_vs_outcome_match=intent_vs_outcome_match,
        latency_ms=round(latency_ms, 2),
        lifecycle_stage="after_tool",
        sanitized_result=redact_pii_payload(result_summary),
    )


# ---------------------------------------------------------------------------
# Strict Pydantic Schemas for Node I/O, Error Recovery & Workflow State
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
    recovery_instructions: str = Field(
        default="",
        description="Explicit self-correction instructions for the LLM if workspace errors occur",
    )


class ExtractedPdfContent(BaseModel):
    """Strict Pydantic schema for PDF text and metadata extracted and compacted via pypdf."""

    day: str = Field(description="Target training day (day1..day5)")
    file_path: str = Field(description="Path to the extracted PDF file")
    file_name: str = Field(description="Filename of the extracted PDF")
    page_count: int = Field(ge=1, description="Number of pages in the PDF")
    extracted_text: str = Field(
        min_length=1, description="Compacted high-signal text across all PDF pages"
    )
    char_count: int = Field(ge=1, description="Character count of compacted text")
    raw_char_count: int = Field(
        default=0, description="Original uncompacted character count before semantic compaction"
    )
    compaction_strategy: str = Field(
        default="hierarchical_semantic_window",
        description="Compaction strategy applied to prevent context window bloat",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict, description="PDF document metadata (Title, Author, etc.)"
    )
    user_name: str = Field(default="", description="Learner full name")
    user_email: str = Field(default="", description="Learner target email")
    technical_track: str = Field(default="", description="Learner technical track")
    recovery_instructions: str = Field(
        default="",
        description="Explicit self-correction instructions for the LLM if PDF extraction fails",
    )


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
    raw_content: str = Field(description="Compacted training text extracted from the PDF")


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


class ConsolidatedLearnerMemory(BaseModel):
    """Long-term consolidated learner memory entry generated asynchronously in the background."""

    user_id_hash: str = Field(description="PII-safe hash of learner email identifier")
    technical_track: str = Field(description="Preferred learner track (Engineer or Architect)")
    completed_days: list[str] = Field(description="Days completed by the learner")
    synthesized_concepts: list[str] = Field(
        description="Consolidated key architectural/engineering takeaways mastered so far"
    )
    active_micro_challenge: str = Field(
        default="", description="Most recent hands-on micro-challenge assigned to the learner"
    )
    consolidated_at: str = Field(description="ISO-8601 UTC timestamp of background consolidation")


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
    model_routing_decision: dict[str, Any] = Field(
        default_factory=dict,
        description="Strategic multi-model routing metadata for cost/latency/complexity optimization",
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
    consolidated_memories: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Long-term learner memories consolidated asynchronously via background tasks",
    )


# ---------------------------------------------------------------------------
# Asynchronous Background Memory Generation & Consolidation Pipeline
# ---------------------------------------------------------------------------


async def consolidate_learner_memory_async(
    state: dict[str, Any],
    session: Any = None,
    memory_service: Any = None,
) -> ConsolidatedLearnerMemory:
    """Asynchronously generate and consolidate episodic session state into long-term Memory Bank."""
    await asyncio.sleep(0)  # Yield control so execution is non-blocking on the event loop
    user_email = str(state.get("user_email") or "anonymous@example.com")
    user_hash = hashlib.sha256(user_email.lower().encode("utf-8")).hexdigest()[:12]
    track = str(state.get("technical_track") or "Architect")
    completed_days = list(state.get("completed_days") or [])
    brief = state.get("daily_brief") if isinstance(state.get("daily_brief"), dict) else {}

    day = str(brief.get("day") or state.get("target_day") or "day1")
    if day and day not in completed_days:
        completed_days.append(day)

    takeaways = [str(t)[:180] for t in (brief.get("key_takeaways") or [])[:5]]
    micro_challenge = str(brief.get("micro_challenge") or "")

    consolidated = ConsolidatedLearnerMemory(
        user_id_hash=user_hash,
        technical_track=track,
        completed_days=completed_days,
        synthesized_concepts=takeaways,
        active_micro_challenge=micro_challenge,
        consolidated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    record_dict = consolidated.model_dump()
    _CONSOLIDATED_MEMORY_STORE.setdefault(user_hash, []).append(record_dict)

    existing_memories = list(state.get("consolidated_memories") or [])
    existing_memories.append(record_dict)
    state["consolidated_memories"] = existing_memories[-10:]

    # Persist into ADK BaseMemoryService (VertexAiMemoryBankService or InMemoryMemoryService)
    if memory_service is not None and session is not None:
        try:
            await memory_service.add_session_to_memory(session)
        except Exception as exc:
            emit_telemetry_log(
                severity="WARNING",
                node="memory_consolidator",
                event_type="memory_bank_sync_fallback",
                message=f"Memory service sync completed via local store ({type(exc).__name__})",
                user_id_hash=user_hash,
            )

    emit_telemetry_log(
        severity="INFO",
        node="memory_consolidator",
        event_type="async_memory_consolidation_completed",
        message=(
            f"Background memory consolidation completed for learner hash={user_hash} "
            f"({len(completed_days)} day(s) consolidated)"
        ),
        user_id_hash=user_hash,
        technical_track=track,
        completed_days=completed_days,
        concept_count=len(takeaways),
    )
    return consolidated


def schedule_background_memory_consolidation(
    state: dict[str, Any],
    session: Any = None,
    memory_service: Any = None,
) -> asyncio.Task[ConsolidatedLearnerMemory] | None:
    """Schedule non-blocking background memory generation & consolidation via asyncio.create_task."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None

    task = loop.create_task(
        consolidate_learner_memory_async(
            state=state,
            session=session,
            memory_service=memory_service,
        ),
        name="consolidate_learner_memory_bg",
    )
    _BACKGROUND_MEMORY_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_MEMORY_TASKS.discard)
    return task


# ---------------------------------------------------------------------------
# Hierarchical Semantic Compaction for Multi-Page PDF Context Management
# ---------------------------------------------------------------------------


def compact_pdf_pages_hierarchically(
    pages_text: list[str],
    max_token_budget: int = 16_000,
) -> tuple[str, str]:
    """Perform section-aware hierarchical semantic compaction across multi-page PDF documents.

    Instead of naive character slicing, this compaction pipeline:
    1. Deduplicates repetitive slide headers, footers, and copyright boilerplate across pages.
    2. Preserves high-signal structural headings, architectural patterns, bullet takeaways, and code schemas.
    3. Applies a sliding semantic window across all pages so every section of a 50+ page deck is represented
       within the target token budget (~4 chars/token).
    """
    if not pages_text:
        return "", "empty_document"

    max_char_budget = max_token_budget * 4
    seen_lines: set[str] = set()
    compacted_pages: list[str] = []

    boilerplate_patterns = (
        "copyright 202",
        "all rights reserved",
        "confidential",
        "google cloud",
    )

    for page_raw in pages_text:
        unique_lines: list[str] = []
        for raw_line in page_raw.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line:
                continue
            line_lower = line.lower()
            if any(bp in line_lower for bp in boilerplate_patterns) and len(line) < 60:
                continue
            # Always keep [Document: ... | Page ...] headers, section titles, and unique content lines
            norm_key = re.sub(r"[^a-z0-9]", "", line_lower)
            if line.startswith("[Document:") or len(norm_key) < 12 or norm_key not in seen_lines:
                if len(norm_key) >= 12:
                    seen_lines.add(norm_key)
                unique_lines.append(line)
        if unique_lines:
            compacted_pages.append("\n".join(unique_lines))

    joined = "\n\n".join(compacted_pages).strip()
    if len(joined) <= max_char_budget:
        return joined, "hierarchical_dedup_full_coverage"

    # Proportional per-page semantic window allocation so all pages remain represented
    per_page_budget = max(400, max_char_budget // max(1, len(compacted_pages)))
    windowed_pages: list[str] = []
    for page_block in compacted_pages:
        if len(page_block) <= per_page_budget:
            windowed_pages.append(page_block)
        else:
            lines = page_block.splitlines()
            header = lines[0] if lines else ""
            high_signal = [
                ln
                for ln in lines[1:]
                if any(
                    kw in ln.lower()
                    for kw in (
                        "architect",
                        "engineer",
                        "agent",
                        "tool",
                        "memory",
                        "workflow",
                        "mcp",
                        "a2a",
                        "security",
                        "guardrail",
                        "telemetry",
                        "eval",
                        "pattern",
                        ":",
                    )
                )
            ]
            selected = [header] + (high_signal if high_signal else lines[1:])
            chunk = "\n".join(selected)[:per_page_budget]
            windowed_pages.append(chunk)

    return "\n\n".join(windowed_pages), "hierarchical_semantic_sliding_window"


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
        chunk_size = 4
        pages_lines = [
            lines[i : i + chunk_size] for i in range(0, max(1, len(lines)), chunk_size)
        ]
    num_pages = len(pages_lines)

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
    start_ts = time.perf_counter()
    normalized_day = normalize_day(target_day)
    declared_intent = f"Initialize training PDF directories day1..day5 and verify PDF availability for {normalized_day}"
    log_tool_intent_before(
        tool_name="initialize_pdf_workspace",
        declared_intent=declared_intent,
        tool_args={
            "target_day": normalized_day,
            "workspace_root": workspace_root,
            "user_name": user_name,
            "user_email": user_email,
            "technical_track": technical_track,
        },
        node="init_workspace",
    )

    root_path = Path(workspace_root)
    created_days: list[str] = []
    seeded_for_target = False
    recovery_instructions = ""
    bundled_corpus = _load_bundled_corpus()

    try:
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
    except Exception as exc:
        recovery_instructions = (
            f"SELF-CORRECTION REQUIRED: Workspace directory initialization failed ({type(exc).__name__}: {exc}). "
            "Verify that 'workspace_root' points to a writable local path (e.g. '/tmp/content_pdfs' or './content_pdfs') "
            "and retry 'initialize_pdf_workspace' with a valid 'target_day' in ('day1', 'day2', 'day3', 'day4', 'day5')."
        )

    target_dir = root_path / normalized_day
    pdf_files = sorted(p.name for p in target_dir.glob("*.pdf")) if target_dir.exists() else []
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
        recovery_instructions=recovery_instructions,
    )
    latency_ms = (time.perf_counter() - start_ts) * 1000.0
    log_tool_outcome_after(
        tool_name="initialize_pdf_workspace",
        declared_intent=declared_intent,
        actual_outcome=f"Workspace initialized with {len(pdf_files)} PDF(s) in {normalized_day}",
        intent_vs_outcome_match=bool(pdf_files) and not recovery_instructions,
        latency_ms=latency_ms,
        result_summary=status.model_dump(),
        node="init_workspace",
    )
    return status


def extract_pdf_from_workspace(
    target_day: str,
    workspace_root: str = DEFAULT_WORKSPACE_ROOT,
    user_name: str = "",
    user_email: str = "",
    technical_track: str = "",
) -> ExtractedPdfContent:
    """Discover, extract, and semantically compact multi-page training material text from ./content_pdfs/<target_day>."""
    start_ts = time.perf_counter()
    normalized_day = normalize_day(target_day)
    declared_intent = f"Extract and semantically compact multi-page PDF training material for {normalized_day}"
    log_tool_intent_before(
        tool_name="extract_pdf_from_workspace",
        declared_intent=declared_intent,
        tool_args={
            "target_day": normalized_day,
            "workspace_root": workspace_root,
            "user_name": user_name,
            "user_email": user_email,
            "technical_track": technical_track,
        },
        node="pdf_reader",
    )

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
        raise FileNotFoundError(
            f"No PDF files found in '{day_dir}'. SELF-CORRECTION: Call initialize_pdf_workspace(target_day='{normalized_day}') first."
        )

    starter_name = f"{normalized_day}_training_guide.pdf"
    custom_pdfs = [p for p in all_pdfs if p.name != starter_name]
    target_pdfs = custom_pdfs if custom_pdfs else all_pdfs

    pages_text: list[str] = []
    total_pages = 0
    clean_meta: dict[str, str] = {}
    recovery_instructions = ""

    for pdf_path in target_pdfs:
        try:
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
        except Exception as exc:
            recovery_instructions = (
                f"SELF-CORRECTION REQUIRED: Corrupt or unreadable PDF '{pdf_path.name}' ({type(exc).__name__}: {exc}). "
                f"Remove or replace the corrupted file in '{day_dir}' or re-run initialize_pdf_workspace(target_day='{normalized_day}') "
                "to restore the verified starter training PDF."
            )

    primary_pdf = target_pdfs[0]
    raw_combined = "\n\n".join(pages_text).strip()
    raw_char_count = len(raw_combined)

    if not raw_combined:
        title, lines = STARTER_CURRICULUM.get(normalized_day, STARTER_CURRICULUM["day1"])
        pages_text = [f"{title}\n" + "\n".join(lines)]
        raw_char_count = len(pages_text[0])
        if not recovery_instructions:
            recovery_instructions = (
                "Fallback curriculum loaded because the PDF contained scanned images without extractable text glyphs. "
                "To summarize custom material, provide a text-searchable PDF in the target day directory."
            )

    compacted_text, compaction_strategy = compact_pdf_pages_hierarchically(
        pages_text=pages_text,
        max_token_budget=16_000,
    )

    extracted = ExtractedPdfContent(
        day=normalized_day,
        file_path=str(primary_pdf),
        file_name=primary_pdf.name if len(target_pdfs) == 1 else ", ".join(p.name for p in target_pdfs),
        page_count=max(1, total_pages),
        extracted_text=compacted_text,
        char_count=len(compacted_text),
        raw_char_count=raw_char_count,
        compaction_strategy=compaction_strategy,
        metadata=clean_meta,
        user_name=user_name,
        user_email=user_email,
        technical_track=technical_track,
        recovery_instructions=recovery_instructions,
    )
    latency_ms = (time.perf_counter() - start_ts) * 1000.0
    log_tool_outcome_after(
        tool_name="extract_pdf_from_workspace",
        declared_intent=declared_intent,
        actual_outcome=(
            f"Extracted {extracted.char_count} compacted chars (from {raw_char_count} raw chars) "
            f"across {extracted.page_count} page(s) using {compaction_strategy}"
        ),
        intent_vs_outcome_match=extracted.char_count > 0,
        latency_ms=latency_ms,
        result_summary={
            "day": extracted.day,
            "file_name": extracted.file_name,
            "page_count": extracted.page_count,
            "char_count": extracted.char_count,
            "raw_char_count": extracted.raw_char_count,
            "compaction_strategy": extracted.compaction_strategy,
            "user_name": extracted.user_name,
            "user_email": extracted.user_email,
        },
        node="pdf_reader",
    )
    return extracted


# ---------------------------------------------------------------------------
# External Dispatch Tools (Email & Google Chat Card v2 Webhook) with Self-Correction
# ---------------------------------------------------------------------------


def send_email_brief(
    recipient_email: str,
    user_name: str,
    day: str,
    track: str,
    subject: str,
    body_markdown: str,
) -> dict[str, Any]:
    """Send the confirmed daily brief to the learner's target email address with explicit LLM self-correction instructions on error.

    Args:
        recipient_email: Target RFC-5322 email address of the learner (e.g. user@domain.com).
        user_name: Full name of the learner.
        day: Training day identifier (day1 through day5).
        track: Learner's preferred technical track (Engineer or Architect).
        subject: Non-empty email subject line for the daily brief.
        body_markdown: Full formatted daily brief markdown body content.

    Returns:
        A dictionary containing delivery status, recipient, subject, transport mode, and
        explicit `recovery_instructions` for LLM self-correction if parameter validation or SMTP fails.
    """
    start_ts = time.perf_counter()
    declared_intent = f"Deliver approved {day.upper()} ({track}) daily brief email to learner"

    with tracer.start_as_current_span("send_email_brief") as span:
        span.set_attribute("gen_ai.tool.name", "send_email_brief")
        span.set_attribute("gen_ai.tool.declared_intent", declared_intent)
        span.set_attribute("gen_ai.security.pii_redacted", True)

        log_tool_intent_before(
            tool_name="send_email_brief",
            declared_intent=declared_intent,
            tool_args={
                "recipient_email": recipient_email,
                "user_name": user_name,
                "day": day,
                "track": track,
                "subject": subject,
            },
            node="dispatch_brief",
        )

        # Parameter validation with explicit self-correction instructions back to the LLM
        clean_email = str(recipient_email or "").strip()
        if not clean_email or "@" not in clean_email or "." not in clean_email.split("@")[-1]:
            recovery_msg = (
                f"SELF-CORRECTION REQUIRED: Invalid 'recipient_email' value '{clean_email}'. "
                "You MUST extract the valid RFC-5322 email address from the input DailyBrief.user_email field "
                "(format: 'local@domain.tld') and re-invoke 'send_email_brief' with that corrected email."
            )
            error_res = {
                "status": "error",
                "error_type": "InvalidRecipientEmailError",
                "recipient_email": clean_email,
                "recovery_instructions": recovery_msg,
                "self_correction_required": True,
            }
            span.set_attribute("gen_ai.tool.actual_outcome", "validation_error")
            span.set_attribute("gen_ai.tool.intent_outcome_match", False)
            log_tool_outcome_after(
                tool_name="send_email_brief",
                declared_intent=declared_intent,
                actual_outcome="Validation failed: malformed recipient_email",
                intent_vs_outcome_match=False,
                latency_ms=(time.perf_counter() - start_ts) * 1000.0,
                result_summary=error_res,
                node="dispatch_brief",
            )
            return error_res

        if not str(subject or "").strip() or not str(body_markdown or "").strip():
            recovery_msg = (
                "SELF-CORRECTION REQUIRED: Both 'subject' and 'body_markdown' must be non-empty strings. "
                "Re-invoke 'send_email_brief' passing the DailyBrief.title in 'subject' and the full formatted "
                "summary (executive_summary, key_takeaways, curiosity_hooks, and micro_challenge) in 'body_markdown'."
            )
            error_res = {
                "status": "error",
                "error_type": "EmptyEmailContentError",
                "recipient_email": clean_email,
                "recovery_instructions": recovery_msg,
                "self_correction_required": True,
            }
            span.set_attribute("gen_ai.tool.actual_outcome", "validation_error")
            span.set_attribute("gen_ai.tool.intent_outcome_match", False)
            log_tool_outcome_after(
                tool_name="send_email_brief",
                declared_intent=declared_intent,
                actual_outcome="Validation failed: empty subject or body_markdown",
                intent_vs_outcome_match=False,
                latency_ms=(time.perf_counter() - start_ts) * 1000.0,
                result_summary=error_res,
                node="dispatch_brief",
            )
            return error_res

        normalized_day = normalize_day(day)
        smtp_host = os.environ.get("SMTP_HOST", "").strip()
        smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        smtp_user = os.environ.get("SMTP_USER", "").strip()
        smtp_password = os.environ.get("SMTP_PASSWORD", "").strip()
        sender_email = os.environ.get("SMTP_FROM", smtp_user or "ai-in-5-days@google.com")

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = clean_email
        msg.set_content(body_markdown)

        outbox_dir = Path(os.environ.get("DISPATCHED_BRIEFS_DIR", "./dispatched_briefs"))
        outbox_dir.mkdir(parents=True, exist_ok=True)
        safe_email = re.sub(r"[^a-zA-Z0-9_.-]", "_", clean_email)
        md_path = outbox_dir / f"{normalized_day}_email_{safe_email}.md"
        eml_path = outbox_dir / f"{normalized_day}_email_{safe_email}.eml"
        md_path.write_text(
            f"To: {clean_email}\nFrom: {sender_email}\nSubject: {subject}\nDay: {normalized_day.upper()} ({track} Track)\n\n---\n\n{body_markdown}\n",
            encoding="utf-8",
        )
        eml_path.write_bytes(bytes(msg))

        delivery_mode = "dry_run_saved_to_outbox"
        recovery_instructions = (
            "No recovery action required: email payload validated and persisted to local outbox. "
            "Proceed to call 'post_google_chat_card' if not yet invoked, then report completion."
        )
        if smtp_host:
            try:
                with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
                    if smtp_user and smtp_password:
                        server.starttls()
                        server.login(smtp_user, smtp_password)
                    server.send_message(msg)
                delivery_mode = "smtp_sent"
                recovery_instructions = "SMTP delivery succeeded; no retry or self-correction needed."
            except Exception as exc:
                delivery_mode = f"dry_run_fallback ({type(exc).__name__})"
                recovery_instructions = (
                    f"SMTP transport encountered {type(exc).__name__} ({exc}) and automatically fell back to outbox persistence "
                    f"at '{md_path}'. Do NOT retry 'send_email_brief' with identical arguments to avoid duplicate outbox writes; "
                    "instead inform the user that the brief was archived to the local outbox and proceed with 'post_google_chat_card'."
                )

        result = {
            "status": "sent",
            "delivery_mode": delivery_mode,
            "recipient_email": clean_email,
            "user_name": user_name,
            "day": normalized_day,
            "track": track,
            "subject": subject,
            "saved_email_path": str(md_path),
            "recovery_instructions": recovery_instructions,
            "self_correction_required": False,
        }
        latency_ms = (time.perf_counter() - start_ts) * 1000.0
        span.set_attribute("gen_ai.tool.actual_outcome", delivery_mode)
        span.set_attribute("gen_ai.tool.intent_outcome_match", True)
        log_tool_outcome_after(
            tool_name="send_email_brief",
            declared_intent=declared_intent,
            actual_outcome=f"Email dispatched via {delivery_mode}",
            intent_vs_outcome_match=True,
            latency_ms=latency_ms,
            result_summary=result,
            node="dispatch_brief",
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
    """Post an interactive Google Chat Card v2 payload to a Google Chat Space webhook with LLM self-correction guidance.

    Args:
        user_name: Full name of the learner.
        day: Training day identifier (e.g., day1).
        track: Learner's preferred technical track (Engineer or Architect).
        title: Headline title of the daily brief.
        summary: Executive summary personalized to the learner's track.
        curiosity_hook: Curiosity hook question or insight from the brief.
        micro_challenge: Concrete hands-on micro-challenge for the day.

    Returns:
        A dictionary containing the webhook delivery status, Card v2 metadata, and explicit
        `recovery_instructions` for LLM self-correction if validation or webhook transport fails.
    """
    start_ts = time.perf_counter()
    normalized_day = normalize_day(day)
    declared_intent = f"Post interactive Google Chat Card v2 for {normalized_day.upper()} ({track} track)"

    with tracer.start_as_current_span("post_google_chat_card") as span:
        span.set_attribute("gen_ai.tool.name", "post_google_chat_card")
        span.set_attribute("gen_ai.tool.declared_intent", declared_intent)
        span.set_attribute("gen_ai.security.pii_redacted", True)

        log_tool_intent_before(
            tool_name="post_google_chat_card",
            declared_intent=declared_intent,
            tool_args={
                "user_name": user_name,
                "day": normalized_day,
                "track": track,
                "title": title,
            },
            node="dispatch_brief",
        )

        if not str(title or "").strip() or not str(summary or "").strip():
            recovery_msg = (
                "SELF-CORRECTION REQUIRED: 'title' and 'summary' cannot be empty when building a Google Chat Card v2 payload. "
                "Re-invoke 'post_google_chat_card' with the non-empty DailyBrief.title and DailyBrief.executive_summary fields."
            )
            error_res = {
                "status": "error",
                "error_type": "InvalidChatCardPayloadError",
                "recovery_instructions": recovery_msg,
                "self_correction_required": True,
            }
            span.set_attribute("gen_ai.tool.actual_outcome", "validation_error")
            span.set_attribute("gen_ai.tool.intent_outcome_match", False)
            log_tool_outcome_after(
                tool_name="post_google_chat_card",
                declared_intent=declared_intent,
                actual_outcome="Validation failed: empty title or summary",
                intent_vs_outcome_match=False,
                latency_ms=(time.perf_counter() - start_ts) * 1000.0,
                result_summary=error_res,
                node="dispatch_brief",
            )
            return error_res

        card_payload = {
            "cardsV2": [
                {
                    "cardId": f"ai-in-5-days-{normalized_day}-{track.lower()}",
                    "card": {
                        "header": {
                            "title": title,
                            "subtitle": f"AI in 5 Days ({normalized_day.upper()}) • {track} Track • For {user_name}",
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
        card_path = outbox_dir / f"{normalized_day}_google_chat_card_{track.lower()}.json"
        card_path.write_text(json.dumps(card_payload, indent=2), encoding="utf-8")

        webhook_url = os.environ.get("GOOGLE_CHAT_WEBHOOK_URL", "").strip()
        delivery_mode = "dry_run_card_v2_saved"
        recovery_instructions = (
            "No recovery action required: Google Chat Card v2 payload validated and saved to outbox."
        )
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
                    recovery_instructions = "Webhook POST succeeded; no retry needed."
            except Exception as exc:
                delivery_mode = f"dry_run_fallback ({type(exc).__name__})"
                recovery_instructions = (
                    f"Google Chat webhook POST encountered {type(exc).__name__} ({exc}) and fell back to local Card v2 "
                    f"persistence at '{card_path}'. Do NOT re-invoke 'post_google_chat_card'; report fallback status to the user."
                )

        result = {
            "status": "posted",
            "delivery_mode": delivery_mode,
            "card_id": f"ai-in-5-days-{normalized_day}-{track.lower()}",
            "user_name": user_name,
            "day": normalized_day,
            "track": track,
            "title": title,
            "saved_card_path": str(card_path),
            "card_v2": card_payload,
            "recovery_instructions": recovery_instructions,
            "self_correction_required": False,
        }
        latency_ms = (time.perf_counter() - start_ts) * 1000.0
        span.set_attribute("gen_ai.tool.actual_outcome", delivery_mode)
        span.set_attribute("gen_ai.tool.intent_outcome_match", True)
        log_tool_outcome_after(
            tool_name="post_google_chat_card",
            declared_intent=declared_intent,
            actual_outcome=f"Chat Card v2 dispatched via {delivery_mode}",
            intent_vs_outcome_match=True,
            latency_ms=latency_ms,
            result_summary={
                "status": result["status"],
                "delivery_mode": result["delivery_mode"],
                "card_id": result["card_id"],
                "user_name": result["user_name"],
                "day": result["day"],
                "track": result["track"],
                "saved_card_path": result["saved_card_path"],
            },
            node="dispatch_brief",
        )
        return result
