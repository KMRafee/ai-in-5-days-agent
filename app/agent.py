# ruff: noqa
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

"""AI in 5 Days Content Customizer Agent implemented with ADK 2.0 Graph Workflow."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from google.adk import Agent, Context, Event, Workflow
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps import App
from google.adk.events import RequestInput
from google.adk.models import Gemini
from google.adk.workflow import node
from google.genai import types

from app.tools import (
    DEFAULT_WORKSPACE_ROOT,
    DailyBrief,
    DispatchReceipt,
    ExtractedPdfContent,
    SummarizerInput,
    UserProfile,
    WorkflowState,
    WorkspaceStatus,
    emit_telemetry_log,
    extract_pdf_from_workspace,
    initialize_pdf_workspace,
    normalize_day,
    normalize_track,
    post_google_chat_card,
    send_email_brief,
)

MODEL = "gemini-3.8-flash"


# ---------------------------------------------------------------------------
# Helper utilities for parsing user input & formatting model events
# ---------------------------------------------------------------------------


def _model_content(text: str) -> types.Content:
    """Create a model-role Content object for user-visible workflow messages."""
    return types.Content(role="model", parts=[types.Part.from_text(text=text)])


def _extract_text_from_input(node_input: Any, ctx: Context | None = None) -> str:
    """Extract plain text from node_input or ctx.user_content."""
    for candidate in (node_input, getattr(ctx, "user_content", None)):
        if candidate is None:
            continue
        if isinstance(candidate, str):
            return candidate.strip()
        if isinstance(candidate, dict):
            return json.dumps(candidate)
        parts = getattr(candidate, "parts", None)
        if parts:
            texts = [
                 getattr(p, "text", "")
                for p in parts
                if getattr(p, "text", None)
            ]
            if texts:
                return " ".join(texts).strip()
    return ""


def _parse_profile_hints(text: str) -> dict[str, Any]:
    """Extract optional user_name, user_email, technical_track, target_day, and auto_confirm from text or JSON."""
    if not text:
        return {}

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            hints: dict[str, Any] = {}
            for k in ("user_name", "name"):
                if parsed.get(k):
                    hints["user_name"] = str(parsed[k]).strip()
            for k in ("user_email", "email", "target_email"):
                if parsed.get(k):
                    hints["user_email"] = str(parsed[k]).strip()
            for k in ("technical_track", "track"):
                if parsed.get(k):
                    hints["technical_track"] = normalize_track(str(parsed[k]))
            for k in ("target_day", "day"):
                if parsed.get(k):
                    hints["target_day"] = normalize_day(str(parsed[k]))
            if "auto_confirm" in parsed or "decision" in parsed:
                hints["decision"] = str(
                    parsed.get("decision")
                    or ("approve" if parsed.get("auto_confirm") else "")
                )
            return hints
    except (json.JSONDecodeError, TypeError):
        pass

    hints = {}
    email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    if email_match:
        hints["user_email"] = email_match.group(0)

    day_match = re.search(r"\bday\s*[-_]?([1-5])\b", text, re.IGNORECASE)
    if day_match:
        hints["target_day"] = f"day{day_match.group(1)}"

    track_match = re.search(
        r"\b(architect|engineer|developer|data\s+scientist)\b",
        text,
        re.IGNORECASE,
    )
    if track_match:
        hints["technical_track"] = normalize_track(track_match.group(1))

    name_patterns = [
        r"(?i:my\s+name\s+is|user_name\s*[:=]\s*|name\s*[:=]\s*|for\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
    ]
    for pat in name_patterns:
        m = re.search(pat, text)
        if m:
            candidate_name = m.group(1).strip()
            if candidate_name.lower() not in {
                "day",
                "engineer",
                "architect",
                "track",
                "email",
            }:
                hints["user_name"] = candidate_name
                break

    lower = text.lower()
    if any(
        phrase in lower
        for phrase in (
            "auto-approve",
            "auto_approve",
            "auto approve",
            "confirm_dispatch: approve",
            "confirmed",
        )
    ):
        hints["decision"] = "approve"

    return hints


def format_daily_brief_markdown(brief: DailyBrief) -> str:
    """Format a DailyBrief Pydantic model into readable CLI/UI Markdown."""
    takeaways_md = "\n".join(f"- {item}" for item in brief.key_takeaways)
    hooks_md = "\n".join(f"- {item}" for item in brief.curiosity_hooks)
    return (
        f"### {brief.title}\n"
        f"**Learner:** {brief.user_name} (`{brief.user_email}`) | "
        f"**Track:** {brief.technical_track} | **Day:** {brief.day.upper()}\n\n"
        f"#### Executive Summary\n{brief.executive_summary}\n\n"
        f"#### Key Takeaways ({brief.technical_track} Track)\n{takeaways_md}\n\n"
        f"#### Curiosity Hooks\n{hooks_md}\n\n"
        f"#### Daily Micro-Challenge\n{brief.micro_challenge}"
    )


# ---------------------------------------------------------------------------
# Node 1: setup_session (Conversational RequestInput Loop)
# ---------------------------------------------------------------------------


@node(name="setup_session", rerun_on_resume=True)
async def setup_session(ctx: Context, node_input: Any = None):
    """Collect or restore Learner Name, Target Email, and Technical Track via RequestInput."""
    raw_text = _extract_text_from_input(node_input, ctx)
    resume_inputs = dict(getattr(ctx, "resume_inputs", None) or {})
    pending_interrupt = str(ctx.state.get("pending_interrupt", "") or "")

    # If human_review is awaiting confirmation and the user replied via plain text,
    # route directly to human_review with the user's decision.
    if (
        pending_interrupt == "confirm_dispatch"
        and ctx.state.get("daily_brief")
        and raw_text
    ):
        decision_lower = raw_text.strip().lower()
        if any(
            tok in decision_lower
            for tok in ("approve", "reject", "yes", "no", "confirm", "cancel")
        ):
            state_delta = {
                "pending_review_decision": raw_text.strip(),
                "pending_interrupt": "",
            }
            ctx.state.update(state_delta)
            yield Event(
                route="RESUME_REVIEW",
                output=ctx.state["daily_brief"],
                state=state_delta,
            )
            return

    hints = _parse_profile_hints(raw_text)

    # Apply plain-text reply to whichever setup interrupt was pending
    if pending_interrupt == "ask_user_name" and "ask_user_name" not in resume_inputs and raw_text:
        resume_inputs["ask_user_name"] = raw_text
    elif pending_interrupt == "ask_user_email" and "ask_user_email" not in resume_inputs and raw_text:
        resume_inputs["ask_user_email"] = raw_text
    elif pending_interrupt == "ask_technical_track" and "ask_technical_track" not in resume_inputs and raw_text:
        resume_inputs["ask_technical_track"] = raw_text

    user_name = (
        str(resume_inputs.get("ask_user_name") or "").strip()
        or str(ctx.state.get("user_name") or "").strip()
        or str(hints.get("user_name") or "").strip()
    )
    raw_email_input = (
        str(resume_inputs.get("ask_user_email") or "").strip()
        or str(ctx.state.get("user_email") or "").strip()
        or str(hints.get("user_email") or "").strip()
    )
    email_extract = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", raw_email_input)
    user_email = email_extract.group(0) if email_extract else raw_email_input

    technical_track = (
        str(resume_inputs.get("ask_technical_track") or "").strip()
        or str(ctx.state.get("technical_track") or "").strip()
        or str(hints.get("technical_track") or "").strip()
    )

    # Check if the user included the Day alongside their email, track, or name input
    combined_resume_text = " ".join(
        str(v) for v in resume_inputs.values() if v is not None
    )
    resume_day_match = re.search(
        r"\bday\s*[-_]?([1-5])\b", combined_resume_text, re.IGNORECASE
    )
    inferred_resume_day = (
        f"day{resume_day_match.group(1)}" if resume_day_match else ""
    )

    target_day = normalize_day(
        str(
            resume_inputs.get("ask_target_day")
            or inferred_resume_day
            or hints.get("target_day")
            or ctx.state.get("target_day")
            or "day1"
        )
    )

    # If the user provided both Name and Email upfront in their prompt without specifying
    # a track, default to Architect so summarization proceeds immediately.
    if (
        not resume_inputs
        and not pending_interrupt
        and user_name
        and user_email
        and not technical_track
    ):
        technical_track = "Architect"

    # Support non-interactive server smoke tests when INTEGRATION_TEST=TRUE
    if os.environ.get("INTEGRATION_TEST", "").upper() == "TRUE":
        user_name = user_name or "Alex Rivera"
        user_email = user_email or "alex.rivera@example.com"
        technical_track = technical_track or "Architect"
        ctx.state["auto_confirm"] = True

    if hints.get("decision") == "approve":
        ctx.state["auto_confirm"] = True

    # Step 1 of RequestInput loop: Ask for Learner Name
    if not user_name:
        prompt_msg = (
            "Welcome to the AI in 5 Days Content Customizer! "
            "To personalize your daily training material brief, what is your full name?"
        )
        state_delta = {"pending_interrupt": "ask_user_name", "target_day": target_day}
        ctx.state.update(state_delta)
        emit_telemetry_log(
            severity="INFO",
            node="setup_session",
            event_type="request_input",
            message="Requesting user_name via RequestInput",
            interrupt_id="ask_user_name",
        )
        yield Event(content=_model_content(prompt_msg), state=state_delta)
        yield RequestInput(
            interrupt_id="ask_user_name",
            message=prompt_msg,
            payload={"step": 1, "field": "user_name"},
        )
        return

    # Step 2 of RequestInput loop: Ask for Target Email Address (and Target Day)
    if not user_email:
        prompt_msg = (
            f"Nice to meet you, {user_name}! "
            "Please share your target email address and the Training Day (e.g., Day 1–Day 5) "
            "whose training material PDF you would like me to summarize and send:"
        )
        state_delta = {
            "user_name": user_name,
            "pending_interrupt": "ask_user_email",
            "target_day": target_day,
        }
        ctx.state.update(state_delta)
        emit_telemetry_log(
            severity="INFO",
            node="setup_session",
            event_type="request_input",
            message="Requesting user_email and target_day via RequestInput",
            interrupt_id="ask_user_email",
            user_name=user_name,
        )
        yield Event(content=_model_content(prompt_msg), state=state_delta)
        yield RequestInput(
            interrupt_id="ask_user_email",
            message=prompt_msg,
            payload={
                "step": 2,
                "field": "user_email",
                "user_name": user_name,
                "supports_day_selection": True,
            },
        )
        return

    # Step 3 of RequestInput loop: Ask for Preferred Technical Track
    if not technical_track:
        prompt_msg = (
            f"Got it (Email: `{user_email}`, Target Day: `{target_day.upper()}`). "
            "Which technical track do you prefer for your training material summary "
            "and micro-challenges: Engineer or Architect?"
        )
        state_delta = {
            "user_name": user_name,
            "user_email": user_email,
            "pending_interrupt": "ask_technical_track",
            "target_day": target_day,
        }
        ctx.state.update(state_delta)
        emit_telemetry_log(
            severity="INFO",
            node="setup_session",
            event_type="request_input",
            message="Requesting technical_track via RequestInput",
            interrupt_id="ask_technical_track",
            user_name=user_name,
            user_email=user_email,
            target_day=target_day,
        )
        yield Event(content=_model_content(prompt_msg), state=state_delta)
        yield RequestInput(
            interrupt_id="ask_technical_track",
            message=prompt_msg,
            payload={
                "step": 3,
                "field": "technical_track",
                "options": ["Engineer", "Architect"],
                "user_name": user_name,
                "user_email": user_email,
                "target_day": target_day,
            },
        )
        return

    normalized_track = normalize_track(technical_track)
    profile = UserProfile(
        user_name=user_name,
        user_email=user_email,
        technical_track=normalized_track,
        target_day=target_day,
    )
    state_delta = {
        "user_name": profile.user_name,
        "user_email": profile.user_email,
        "technical_track": profile.technical_track,
        "target_day": profile.target_day,
        "pending_interrupt": "",
    }
    ctx.state.update(state_delta)

    emit_telemetry_log(
        severity="INFO",
        node="setup_session",
        event_type="session_configured",
        message=(
            f"Session configured for {profile.user_name} ({profile.user_email}) "
            f"on {profile.technical_track} track ({profile.target_day})"
        ),
        **profile.model_dump(),
    )
    yield Event(
        route="INIT_WORKSPACE",
        output=profile.model_dump(),
        state=state_delta,
    )


# ---------------------------------------------------------------------------
# Node 2: init_workspace (Scaffolds ./content_pdfs/day1..day5)
# ---------------------------------------------------------------------------


def init_workspace(node_input: dict[str, Any] | UserProfile, ctx: Context) -> Event:
    """Create ./content_pdfs/day1 through day5 if missing and guide the user."""
    data = (
        node_input.model_dump()
        if isinstance(node_input, UserProfile)
        else dict(node_input or {})
    )
    target_day = normalize_day(
        data.get("target_day") or ctx.state.get("target_day") or "day1"
    )
    user_name = str(data.get("user_name") or ctx.state.get("user_name") or "Learner")
    user_email = str(
        data.get("user_email") or ctx.state.get("user_email") or "learner@example.com"
    )
    technical_track = normalize_track(
        str(data.get("technical_track") or ctx.state.get("technical_track") or "Engineer")
    )

    workspace_status: WorkspaceStatus = initialize_pdf_workspace(
        target_day=target_day,
        workspace_root=DEFAULT_WORKSPACE_ROOT,
        user_name=user_name,
        user_email=user_email,
        technical_track=technical_track,
    )
    status_dict = workspace_status.model_dump()
    ctx.state["workspace_status"] = status_dict
    return Event(
        output=status_dict,
        state={"workspace_status": status_dict, "target_day": target_day},
    )


# ---------------------------------------------------------------------------
# Node 3: pdf_reader (pypdf Discovery & Strict Pydantic Validation)
# ---------------------------------------------------------------------------


def pdf_reader(node_input: dict[str, Any] | WorkspaceStatus, ctx: Context) -> Event:
    """Extract text from the target day's PDF using pypdf and validate with Pydantic schemas."""
    data = (
        node_input.model_dump()
        if isinstance(node_input, WorkspaceStatus)
        else dict(node_input or {})
    )
    target_day = normalize_day(
        data.get("target_day") or ctx.state.get("target_day") or "day1"
    )
    workspace_root = str(data.get("workspace_root") or DEFAULT_WORKSPACE_ROOT)
    user_name = str(data.get("user_name") or ctx.state.get("user_name") or "Learner")
    user_email = str(
        data.get("user_email") or ctx.state.get("user_email") or "learner@example.com"
    )
    technical_track = normalize_track(
        str(data.get("technical_track") or ctx.state.get("technical_track") or "Engineer")
    )

    extracted: ExtractedPdfContent = extract_pdf_from_workspace(
        target_day=target_day,
        workspace_root=workspace_root,
        user_name=user_name,
        user_email=user_email,
        technical_track=technical_track,
    )
    summarizer_input = SummarizerInput(
        user_name=user_name,
        user_email=user_email,
        technical_track=technical_track,
        target_day=extracted.day,
        pdf_file_name=extracted.file_name,
        page_count=extracted.page_count,
        raw_content=extracted.extracted_text,
    )

    extracted_dict = extracted.model_dump()
    summarizer_dict = summarizer_input.model_dump()
    ctx.state["extracted_pdf"] = extracted_dict
    return Event(
        output=summarizer_dict,
        state={"extracted_pdf": extracted_dict},
    )


# ---------------------------------------------------------------------------
# Node 4: content_summarizer (Gemini Flash Track-Personalized Summarizer)
# ---------------------------------------------------------------------------

content_summarizer = Agent(
    name="content_summarizer",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    mode="single_turn",
    description=(
        "Summarizes multi-page training material PDFs into an engaging daily brief "
        "personalized to the user's technical track with curiosity hooks and micro-challenges."
    ),
    instruction="""You are the "AI in 5 Days" Content Customizer Agent.
You receive full multi-page **Training Material** extracted from the learner's PDF (`SummarizerInput`, spanning `page_count` pages in `raw_content`) and synthesize the substantive training content into an engaging, track-specific `DailyBrief` ready to be sent via email.

IMPORTANT: The PDF is comprehensive multi-page **Training Material** (deep technical concepts, architectures, code/design patterns, and lessons), NOT a superficial agenda or schedule. Summarize the actual subject matter taught across all pages of the document.

Follow these strict personalization rules:
1. Preserve `user_name`, `user_email`, `technical_track`, and set `day` to `target_day`.
2. Write a thorough, multi-sentence `executive_summary` that synthesizes the core technical material covered across the pages of `raw_content`, tailored to `technical_track`:
   - If `Engineer`: focus on implementation mechanics, code/SDK primitives, schemas, tool contracts, and debugging/testing patterns from the training material.
   - If `Architect`: focus on system architecture, service boundaries, governance/security, scalability, observability, and production deployment patterns from the training material.
3. Include 3 to 5 substantive `key_takeaways` grounded directly in the multi-page `raw_content`.
4. Include 2 to 3 compelling `curiosity_hooks` ("What happens when...", "Why do production systems...") that connect the training material to real-world engineering challenges.
5. Design 1 actionable 15-minute `micro_challenge` tailored to `technical_track` based on the day's training material.""",
    input_schema=SummarizerInput,
    output_schema=DailyBrief,
    output_key="daily_brief",
)


# ---------------------------------------------------------------------------
# Node 5: human_review (CLI Display + RequestInput Approval Gate)
# ---------------------------------------------------------------------------


def _coerce_daily_brief(raw_brief: Any, ctx: Context) -> DailyBrief:
    """Coerce predecessor output or state into a validated DailyBrief instance."""
    if isinstance(raw_brief, DailyBrief):
        return raw_brief

    candidate: dict[str, Any] = {}
    if isinstance(raw_brief, dict):
        candidate = dict(raw_brief)
    elif isinstance(raw_brief, str):
        try:
            parsed = json.loads(raw_brief)
            if isinstance(parsed, dict):
                candidate = parsed
        except json.JSONDecodeError:
            pass

    if not candidate or "title" not in candidate:
        state_brief = ctx.state.get("daily_brief")
        if isinstance(state_brief, dict) and "title" in state_brief:
            candidate = dict(state_brief)

    user_name = str(
        candidate.get("user_name") or ctx.state.get("user_name") or "Learner"
    )
    user_email = str(
        candidate.get("user_email")
        or ctx.state.get("user_email")
        or "learner@example.com"
    )
    technical_track = normalize_track(
        str(
            candidate.get("technical_track")
            or ctx.state.get("technical_track")
            or "Engineer"
        )
    )
    day = normalize_day(
        str(candidate.get("day") or ctx.state.get("target_day") or "day1")
    )

    return DailyBrief(
        day=day,
        technical_track=technical_track,
        user_name=user_name,
        user_email=user_email,
        title=str(
            candidate.get("title")
            or f"AI in 5 Days ({day.upper()}): {technical_track} Daily Brief"
        ),
        executive_summary=str(
            candidate.get("executive_summary")
            or f"Personalized {technical_track} synthesis for {day.upper()}."
        ),
        key_takeaways=list(
            candidate.get("key_takeaways")
            or [
                "Use strict Pydantic schemas at every node and tool boundary.",
                "Separate deterministic workflow routing from LLM reasoning.",
                "Gate external dispatches behind explicit RequestInput confirmation.",
            ]
        ),
        curiosity_hooks=list(
            candidate.get("curiosity_hooks")
            or [
                "What happens if an agent dispatches external notifications without a human review checkpoint?",
                "How does typed state_schema improve replayability across multi-turn sessions?",
            ]
        ),
        micro_challenge=str(
            candidate.get("micro_challenge")
            or f"Inspect the {day.upper()} workflow graph and trace how RequestInput pauses execution before external tool calls."
        ),
    )


@node(name="human_review", rerun_on_resume=True)
async def human_review(node_input: Any, ctx: Context):
    """Display the drafted DailyBrief and pause via RequestInput for explicit human confirmation."""
    brief = _coerce_daily_brief(node_input, ctx)
    brief_dict = brief.model_dump()
    formatted_preview = format_daily_brief_markdown(brief)

    resume_inputs = dict(getattr(ctx, "resume_inputs", None) or {})
    decision_raw: Any = resume_inputs.get("confirm_dispatch")
    if decision_raw is None and isinstance(node_input, dict) and "decision" in node_input:
        decision_raw = node_input["decision"]
    if decision_raw is None and ctx.state.get("pending_review_decision"):
        decision_raw = ctx.state.get("pending_review_decision")
    if decision_raw is None and ctx.state.get("auto_confirm"):
        decision_raw = "approve"

    if decision_raw is None:
        review_prompt = (
            f"{formatted_preview}\n\n---\n"
            f"**Human Review Required:** Please confirm before external dispatch.\n"
            f"Reply **`approve`** to send this brief to `{brief.user_email}` and post the Google Chat card, "
            f"or **`reject`** to cancel dispatch."
        )
        state_delta = {
            "daily_brief": brief_dict,
            "pending_interrupt": "confirm_dispatch",
        }
        ctx.state.update(state_delta)
        emit_telemetry_log(
            severity="WARNING",
            node="human_review",
            event_type="awaiting_human_review",
            message=f"Drafted brief '{brief.title}' paused for human confirmation before external dispatch",
            day=brief.day,
            technical_track=brief.technical_track,
            user_email=brief.user_email,
        )
        yield Event(content=_model_content(review_prompt), state=state_delta)
        yield RequestInput(
            interrupt_id="confirm_dispatch",
            message=(
                f"Approve dispatch of '{brief.title}' to {brief.user_email} "
                "and Google Chat Space webhook? (approve / reject)"
            ),
            payload=brief_dict,
        )
        return

    decision_str = str(decision_raw).strip().lower()
    if isinstance(decision_raw, dict):
        decision_str = str(
            decision_raw.get("decision") or decision_raw.get("result") or ""
        ).strip().lower()

    approved = any(
        tok in decision_str
        for tok in ("approve", "yes", "confirm", "send", "ok", "true")
    ) and not any(tok in decision_str for tok in ("reject", "no", "cancel", "deny"))
    route = "APPROVED" if approved else "REJECTED"
    status_label = "approved" if approved else "rejected"

    state_delta = {
        "daily_brief": brief_dict,
        "review_decision": status_label,
        "pending_interrupt": "",
        "pending_review_decision": "",
    }
    ctx.state.update(state_delta)
    emit_telemetry_log(
        severity="INFO" if approved else "WARNING",
        node="human_review",
        event_type="human_review_decision",
        message=f"Human reviewer {status_label} daily brief '{brief.title}' (route={route})",
        decision=status_label,
        route=route,
        user_email=brief.user_email,
        day=brief.day,
    )
    yield Event(
        route=route,
        output=brief_dict,
        state=state_delta,
    )


# ---------------------------------------------------------------------------
# Node 6: dispatch_brief (Email + Google Chat Card v2 Tools) & cancel_dispatch
# ---------------------------------------------------------------------------


def execute_dispatch_tools(
    brief_data: dict[str, Any] | DailyBrief,
    ctx: Context | CallbackContext | None = None,
) -> DispatchReceipt:
    """Deterministically execute send_email_brief and post_google_chat_card for an approved brief."""
    brief = (
        brief_data
        if isinstance(brief_data, DailyBrief)
        else DailyBrief(**brief_data)
    )
    markdown_body = format_daily_brief_markdown(brief)
    curiosity_hook = (
        brief.curiosity_hooks[0]
        if brief.curiosity_hooks
        else "How will you apply today's AI pattern in your next architecture review?"
    )

    email_res = send_email_brief(
        recipient_email=brief.user_email,
        user_name=brief.user_name,
        day=brief.day,
        track=brief.technical_track,
        subject=f"[AI in 5 Days - {brief.day.upper()}] {brief.title}",
        body_markdown=markdown_body,
    )
    chat_res = post_google_chat_card(
        user_name=brief.user_name,
        day=brief.day,
        track=brief.technical_track,
        title=brief.title,
        summary=brief.executive_summary,
        curiosity_hook=curiosity_hook,
        micro_challenge=brief.micro_challenge,
    )

    receipt = DispatchReceipt(
        status="dispatched",
        user_name=brief.user_name,
        user_email=brief.user_email,
        technical_track=brief.technical_track,
        day=brief.day,
        email_status=str(email_res.get("delivery_mode", "sent")),
        chat_webhook_status=str(chat_res.get("delivery_mode", "posted")),
        summary_title=brief.title,
        message=(
            f"Dispatched '{brief.title}' ({brief.day.upper()}, {brief.technical_track} track) "
            f"to {brief.user_email} and posted interactive card to Google Chat Space."
        ),
    )
    if ctx is not None:
        ctx.state["dispatch_receipt"] = receipt.model_dump()
        completed = list(ctx.state.get("completed_days") or [])
        if brief.day not in completed:
            completed.append(brief.day)
        ctx.state["completed_days"] = completed
    return receipt


async def _ensure_dispatch_completed(
    callback_context: CallbackContext,
) -> types.Content | None:
    """Guarantee dispatch receipt is recorded in session state after dispatch_brief runs."""
    if not callback_context.state.get("dispatch_receipt"):
        brief_dict = callback_context.state.get("daily_brief")
        if isinstance(brief_dict, dict) and brief_dict.get("user_email"):
            execute_dispatch_tools(brief_dict, callback_context)
    return None


dispatch_brief = Agent(
    name="dispatch_brief",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    mode="single_turn",
    description=(
        "Dispatches the human-approved daily brief to the learner's email address "
        "and posts an interactive card to the Google Chat Space webhook."
    ),
    instruction="""You are the Dispatch Coordinator for the "AI in 5 Days" Content Customizer Agent.
You are invoked ONLY after the learner has reviewed and explicitly approved their `DailyBrief`.

You MUST perform both dispatches using your tools:
1. Call `send_email_brief` with the learner's `user_email`, `user_name`, `day`, `technical_track`, an engaging subject line, and the formatted markdown body.
2. Call `post_google_chat_card` with `user_name`, `day`, `technical_track`, `title`, `executive_summary`, the primary `curiosity_hook`, and `micro_challenge`.
3. Confirm both deliveries clearly to the user, including their email address, technical track, day, and brief title.""",
    input_schema=DailyBrief,
    tools=[send_email_brief, post_google_chat_card],
    after_agent_callback=_ensure_dispatch_completed,
)


def cancel_dispatch(node_input: dict[str, Any] | DailyBrief, ctx: Context) -> Event:
    """Safely cancel external dispatch when the human reviewer rejects the drafted brief."""
    brief = _coerce_daily_brief(node_input, ctx)
    receipt = DispatchReceipt(
        status="cancelled",
        user_name=brief.user_name,
        user_email=brief.user_email,
        technical_track=brief.technical_track,
        day=brief.day,
        email_status="skipped_by_human_review",
        chat_webhook_status="skipped_by_human_review",
        summary_title=brief.title,
        message=(
            f"Dispatch cancelled for '{brief.title}' ({brief.day.upper()}). "
            f"No email was sent to {brief.user_email} and no Google Chat card was posted."
        ),
    )
    receipt_dict = receipt.model_dump()
    ctx.state["dispatch_receipt"] = receipt_dict
    emit_telemetry_log(
        severity="WARNING",
        node="cancel_dispatch",
        event_type="dispatch_cancelled",
        **receipt_dict,
    )
    return Event(
        content=_model_content(receipt.message),
        output=receipt_dict,
        state={"dispatch_receipt": receipt_dict},
    )


# ---------------------------------------------------------------------------
# Root Agent: ADK 2.0 Graph Workflow
# ---------------------------------------------------------------------------

root_agent = Workflow(
    # Keep in sync with agents-cli-manifest.yaml: agents-cli derives this name
    # from the project `name:` recorded there, and telemetry reports it as
    # gen_ai.agent.name.
    name="ai_in_5_days_agent",
    description=(
        "AI in 5 Days Content Customizer Agent that sets up learner sessions via RequestInput, "
        "initializes ./content_pdfs/day1..day5, extracts training PDFs with pypdf and Pydantic schemas, "
        "summarizes personalized daily briefs with curiosity hooks and micro-challenges, "
        "gates external dispatch on human review, and sends Email + Google Chat Space cards."
    ),
    state_schema=WorkflowState,
    edges=[
        ("START", setup_session),
        (
            setup_session,
            {
                "INIT_WORKSPACE": init_workspace,
                "RESUME_REVIEW": human_review,
            },
        ),
        (init_workspace, pdf_reader, content_summarizer, human_review),
        (
            human_review,
            {
                "APPROVED": dispatch_brief,
                "REJECTED": cancel_dispatch,
            },
        ),
    ],
)

app = App(
    root_agent=root_agent,
    name="app",
)
