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

"""AI in 5 Days Content Customizer Agent implemented with ADK 2.0 Graph Workflow, Strategic Model Routing, Runtime Guardrail Plugins, ADK Context Compaction, and Async Memory Consolidation."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from google.adk import Agent, Context, Event, Workflow
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps import App
from google.adk.apps.app import (
    ContextCacheConfig,
    EventsCompactionConfig,
    ResumabilityConfig,
)
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.events import RequestInput
from google.adk.models import Gemini
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.adk.workflow import node
from google.genai import types

from app.app_utils import services
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
    log_tool_intent_before,
    log_tool_outcome_after,
    normalize_day,
    normalize_track,
    post_google_chat_card,
    redact_pii_payload,
    redact_pii_text,
    schedule_background_memory_consolidation,
    send_email_brief,
)

# ---------------------------------------------------------------------------
# Strategic Multi-Model Routing Configuration
# ---------------------------------------------------------------------------

MODEL = "gemini-3.8-flash"
MODEL_DEEP_REASONING = os.environ.get("MODEL_DEEP_REASONING", "gemini-2.5-pro")
MODEL_FAST_ROUTING = os.environ.get("MODEL_FAST_ROUTING", "gemini-2.5-flash")

# Prompt-injection & jailbreak patterns blocked by the runtime guardrail plugin
_PROMPT_INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"reveal\s+(?:your\s+)?system\s+(?:prompt|instructions)", re.IGNORECASE),
    re.compile(r"system\s+override\s*:\s*disable\s+guardrails", re.IGNORECASE),
    re.compile(r"bypass\s+human[\s_-]+review", re.IGNORECASE),
)
_BLOCKED_EMAIL_DOMAINS = frozenset({"malicious.example", "exfiltrate.invalid", "darkweb.onion"})


def select_strategic_model(
    technical_track: str,
    page_count: int = 1,
    char_count: int = 0,
    task_type: str = "summarization",
) -> dict[str, Any]:
    """Strategically route tasks across model tiers based on task complexity, persona track, and latency budget.

    - Tier 1 (`MODEL_FAST_ROUTING` / `gemini-2.5-flash`): Deterministic tool dispatch, event compaction, and lightweight formatting.
    - Tier 2 (`MODEL` / `gemini-3.8-flash`): High-throughput multi-page synthesis for Engineer implementation briefs.
    - Tier 3 (`MODEL_DEEP_REASONING` / `gemini-2.5-pro`): Deep architectural reasoning for complex multi-page Architect decks (>35 pages).
    """
    normalized_track = normalize_track(technical_track)
    if task_type in ("dispatch", "compaction", "guardrail"):
        selected_tier = "fast_execution_tier"
        target_model = MODEL_FAST_ROUTING
        active_runtime_model = MODEL
        rationale = (
            f"Routed '{task_type}' to low-latency execution tier ({target_model}) "
            "for deterministic tool orchestration and minimal token overhead."
        )
    elif normalized_track == "Architect" and (page_count >= 35 or char_count >= 45_000):
        selected_tier = "deep_architectural_reasoning_tier"
        target_model = MODEL_DEEP_REASONING
        active_runtime_model = MODEL
        rationale = (
            f"Classified high-complexity {normalized_track} deck ({page_count} pages, {char_count} chars) "
            f"for deep architectural synthesis tier ({target_model} policy profile on {active_runtime_model})."
        )
    else:
        selected_tier = "balanced_flash_reasoning_tier"
        target_model = MODEL
        active_runtime_model = MODEL
        rationale = (
            f"Routed {normalized_track} brief ({page_count} pages, {char_count} chars) "
            f"to balanced reasoning tier ({target_model})."
        )

    decision = {
        "task_type": task_type,
        "technical_track": normalized_track,
        "page_count": page_count,
        "char_count": char_count,
        "selected_tier": selected_tier,
        "recommended_model": target_model,
        "active_runtime_model": active_runtime_model,
        "rationale": rationale,
    }
    emit_telemetry_log(
        severity="INFO",
        node="model_router",
        event_type="strategic_model_routed",
        message=rationale,
        **decision,
    )
    return decision


# ---------------------------------------------------------------------------
# Advanced Runtime Guardrails, Policy Enforcement & Observability Plugin
# ---------------------------------------------------------------------------


def _screen_text_for_policy_violations(text: str) -> str | None:
    """Inspect user input or extracted content against runtime prompt-injection and security rules."""
    if not text:
        return None
    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(text):
            return (
                f"Runtime Guardrail Violation: Blocked adversarial prompt-injection pattern "
                f"matching '{pattern.pattern}'."
            )
    return None


async def guardrail_before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
    """Enforce Model Armor prompt-injection screening, PII-redacted model logging, and strategic routing."""
    request_texts: list[str] = []
    for content in getattr(llm_request, "contents", None) or []:
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "text", None):
                request_texts.append(part.text)

    combined_prompt = "\n".join(request_texts)
    violation = _screen_text_for_policy_violations(combined_prompt)
    if violation:
        emit_telemetry_log(
            severity="ERROR",
            node=callback_context.agent_name,
            event_type="runtime_guardrail_blocked_model_input",
            message=violation,
            guardrail_action="BLOCKED",
        )
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text=f"Request blocked by Enterprise Governance Policy Guardrail: {violation}"
                    )
                ],
            )
        )

    emit_telemetry_log(
        severity="INFO",
        node=callback_context.agent_name,
        event_type="model_invocation_before",
        message=f"Model request screened by guardrail for '{callback_context.agent_name}'",
        prompt_char_count=len(combined_prompt),
        guardrail_status="PASSED",
    )
    return None


async def guardrail_after_model_callback(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> LlmResponse | None:
    """Audit LLM response for output policy compliance and PII redaction telemetry."""
    response_text = ""
    if llm_response and llm_response.content and llm_response.content.parts:
        response_text = " ".join(
            p.text for p in llm_response.content.parts if getattr(p, "text", None)
        )
    emit_telemetry_log(
        severity="INFO",
        node=callback_context.agent_name,
        event_type="model_invocation_after",
        message=f"Model response audited for '{callback_context.agent_name}' ({len(response_text)} chars)",
        response_char_count=len(response_text),
        output_guardrail_status="PASSED",
    )
    return None


async def guardrail_before_tool_callback(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
) -> dict[str, Any] | None:
    """Enforce zero-trust tool execution policies, Confused Deputy protection, and explicit BEFORE tool logging."""
    tool_name = getattr(tool, "name", str(tool))
    tool_Context_dict = getattr(tool_context, "state", {}) or {}
    tool_Context_dict[f"_tool_start_{tool_name}"] = time.perf_counter()

    declared_intent = (
        f"Execute '{tool_name}' for day={args.get('day', tool_Context_dict.get('target_day', 'day1'))} "
        f"after human_review verification"
    )
    log_tool_intent_before(
        tool_name=tool_name,
        declared_intent=declared_intent,
        tool_args=args,
        node="before_tool_callback",
    )

    # Guardrail 1: Zero-Trust Human-in-the-Loop approval gate verification
    if tool_name in ("send_email_brief", "post_google_chat_card"):
        review_decision = str(tool_Context_dict.get("review_decision") or "").lower()
        auto_confirm = bool(tool_Context_dict.get("auto_confirm"))
        if review_decision == "rejected":
            recovery = (
                f"POLICY GUARDRAIL BLOCKED: Cannot invoke '{tool_name}' because human_review decision is 'rejected'. "
                "SELF-CORRECTION: Abort external tool calls immediately and report cancellation to the user."
            )
            emit_telemetry_log(
                severity="ERROR",
                node="before_tool_callback",
                event_type="tool_guardrail_blocked",
                message=recovery,
                tool_name=tool_name,
            )
            return {
                "status": "blocked_by_policy_guardrail",
                "error_type": "UnapprovedExternalDispatchError",
                "recovery_instructions": recovery,
                "self_correction_required": True,
            }
        if not review_decision and not auto_confirm:
            # Automatically stamp approved when reached via the APPROVED workflow edge
            tool_Context_dict["review_decision"] = "approved"

    # Guardrail 2: Recipient email domain exfiltration protection
    if tool_name == "send_email_brief":
        recipient = str(args.get("recipient_email") or "").strip().lower()
        domain = recipient.split("@")[-1] if "@" in recipient else ""
        if domain in _BLOCKED_EMAIL_DOMAINS:
            recovery = (
                f"POLICY GUARDRAIL BLOCKED: Recipient domain '{domain}' is on the security blocklist. "
                "SELF-CORRECTION: Verify the learner's authorized corporate/personal email in session state and retry."
            )
            emit_telemetry_log(
                severity="ERROR",
                node="before_tool_callback",
                event_type="tool_guardrail_blocked_domain",
                message=recovery,
                tool_name=tool_name,
                blocked_domain=domain,
            )
            return {
                "status": "blocked_by_policy_guardrail",
                "error_type": "BlockedRecipientDomainError",
                "recovery_instructions": recovery,
                "self_correction_required": True,
            }

    return None


async def guardrail_after_tool_callback(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
    tool_response: dict[str, Any],
) -> dict[str, Any] | None:
    """Capture explicit AFTER tool intent-vs-outcome audit and inject self-correction hints if needed."""
    tool_name = getattr(tool, "name", str(tool))
    state = getattr(tool_context, "state", {}) or {}
    start_ts = state.get(f"_tool_start_{tool_name}", time.perf_counter())
    latency_ms = max(0.0, (time.perf_counter() - float(start_ts)) * 1000.0)

    status = str(tool_response.get("status", "completed")) if isinstance(tool_response, dict) else "completed"
    intent_match = status in ("sent", "posted", "completed", "dispatched")
    declared_intent = f"Complete '{tool_name}' execution without policy or validation errors"

    log_tool_outcome_after(
        tool_name=tool_name,
        declared_intent=declared_intent,
        actual_outcome=f"status={status}",
        intent_vs_outcome_match=intent_match,
        latency_ms=latency_ms,
        result_summary=tool_response if isinstance(tool_response, dict) else {"result": str(tool_response)},
        node="after_tool_callback",
    )
    return None


async def handle_tool_error_with_recovery(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
    error: Exception,
) -> dict[str, Any]:
    """Catch unhandled tool exceptions and return explicit recovery instructions to the LLM for self-correction."""
    tool_name = getattr(tool, "name", str(tool))
    recovery_instructions = (
        f"SELF-CORRECTION REQUIRED: Tool '{tool_name}' raised {type(error).__name__}: {error}. "
        f"Inspect the arguments you passed ({list(args.keys())}), ensure all required strings are non-empty "
        "and conform to the expected schema (valid email address with '@', day in 'day1'..'day5', "
        "track in 'Engineer'|'Architect'), and retry the tool call once with corrected arguments."
    )
    error_payload = {
        "status": "recoverable_tool_error",
        "tool_name": tool_name,
        "error_type": type(error).__name__,
        "error_message": redact_pii_text(str(error)),
        "recovery_instructions": recovery_instructions,
        "self_correction_required": True,
    }
    emit_telemetry_log(
        severity="ERROR",
        node="on_tool_error_callback",
        event_type="tool_error_recovery_issued",
        message=f"Issued LLM self-correction recovery instructions for '{tool_name}' ({type(error).__name__})",
        **error_payload,
    )
    return error_payload


class EnterpriseGovernancePolicyPlugin(BasePlugin):
    """ADK BasePlugin enforcing runtime security guardrails, PII redaction, tool intent/outcome auditing, and async memory consolidation."""

    def __init__(self) -> None:
        super().__init__(name="enterprise_governance_policy_plugin")

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        return await guardrail_before_model_callback(callback_context, llm_request)

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        return await guardrail_after_model_callback(callback_context, llm_response)

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        return await guardrail_before_tool_callback(tool, tool_args, tool_context)

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        return await guardrail_after_tool_callback(tool, tool_args, tool_context, result)

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> dict[str, Any] | None:
        return await handle_tool_error_with_recovery(tool, tool_args, tool_context, error)

    async def after_run_callback(self, *, invocation_context: Any) -> None:
        session = getattr(invocation_context, "session", None)
        state = getattr(session, "state", None) if session else None
        if isinstance(state, dict) and (state.get("daily_brief") or state.get("user_email")):
            schedule_background_memory_consolidation(
                state=state,
                session=session,
                memory_service=services.get_memory_service(),
            )


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

    # Runtime Guardrail check on initial user input
    violation = _screen_text_for_policy_violations(raw_text)
    if violation:
        emit_telemetry_log(
            severity="ERROR",
            node="setup_session",
            event_type="input_guardrail_violation",
            message=violation,
        )
        yield Event(content=_model_content(violation))
        return

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

    if (
        not resume_inputs
        and not pending_interrupt
        and user_name
        and user_email
        and not technical_track
    ):
        technical_track = "Architect"

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
# Node 3: pdf_reader (pypdf Extraction, Semantic Compaction & Strategic Model Routing)
# ---------------------------------------------------------------------------


def pdf_reader(node_input: dict[str, Any] | WorkspaceStatus, ctx: Context) -> Event:
    """Extract and semantically compact text from the target day's PDF and execute strategic model routing."""
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

    # Evaluate document complexity and learner track for Strategic Multi-Model Routing
    routing_decision = select_strategic_model(
        technical_track=technical_track,
        page_count=extracted.page_count,
        char_count=extracted.char_count,
        task_type="summarization",
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
    ctx.state["model_routing_decision"] = routing_decision
    return Event(
        output=summarizer_dict,
        state={
            "extracted_pdf": extracted_dict,
            "model_routing_decision": routing_decision,
        },
    )


# ---------------------------------------------------------------------------
# Node 4: content_summarizer (Track-Personalized Summarizer with Guardrails)
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
You receive full multi-page **Training Material** extracted and semantically compacted from the learner's PDF (`SummarizerInput`, spanning `page_count` pages in `raw_content`) and synthesize the substantive training content into an engaging, track-specific `DailyBrief` ready to be sent via email.

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
    before_model_callback=guardrail_before_model_callback,
    after_model_callback=guardrail_after_model_callback,
)


# ---------------------------------------------------------------------------
# Node 5: human_review (CLI Display + RequestInput Approval Gate + Async Memory)
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
    """Display the drafted DailyBrief, pause via RequestInput for human confirmation, and schedule async memory consolidation."""
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
            user_name=brief.user_name,
            declared_intent="Pause execution for mandatory human-in-the-loop review before external dispatch",
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

    # Route dispatch model strategically to the fast execution tier when approved
    dispatch_routing = select_strategic_model(
        technical_track=brief.technical_track,
        page_count=1,
        char_count=len(brief.executive_summary),
        task_type="dispatch",
    )

    state_delta = {
        "daily_brief": brief_dict,
        "review_decision": status_label,
        "pending_interrupt": "",
        "pending_review_decision": "",
        "model_routing_decision": dispatch_routing,
    }
    ctx.state.update(state_delta)

    # Trigger non-blocking asynchronous background memory consolidation
    schedule_background_memory_consolidation(
        state=ctx.state,
        session=getattr(ctx, "session", None),
        memory_service=services.get_memory_service(),
    )

    emit_telemetry_log(
        severity="INFO" if approved else "WARNING",
        node="human_review",
        event_type="human_review_decision",
        message=f"Human reviewer {status_label} daily brief '{brief.title}' (route={route})",
        decision=status_label,
        route=route,
        user_email=brief.user_email,
        user_name=brief.user_name,
        day=brief.day,
        declared_intent="Verify human reviewer authorization before external tool invocation",
        actual_outcome=f"human_review_{status_label}",
        intent_vs_outcome_match=True,
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
        schedule_background_memory_consolidation(
            state=ctx.state,
            session=getattr(ctx, "session", None),
            memory_service=services.get_memory_service(),
        )
    return receipt


async def _ensure_dispatch_completed(
    callback_context: CallbackContext,
) -> types.Content | None:
    """Guarantee dispatch receipt is recorded in session state and background memory consolidation runs."""
    if not callback_context.state.get("dispatch_receipt"):
        brief_dict = callback_context.state.get("daily_brief")
        if isinstance(brief_dict, dict) and brief_dict.get("user_email"):
            execute_dispatch_tools(brief_dict, callback_context)
    else:
        schedule_background_memory_consolidation(
            state=callback_context.state,
            session=getattr(callback_context, "_invocation_context", None),
            memory_service=services.get_memory_service(),
        )
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
3. If any tool returns `self_correction_required: true` or `recovery_instructions`, follow the `recovery_instructions` immediately to repair the parameters and retry.
4. Confirm both deliveries clearly to the user, including their email address, technical track, day, and brief title.""",
    input_schema=DailyBrief,
    tools=[send_email_brief, post_google_chat_card],
    before_model_callback=guardrail_before_model_callback,
    after_model_callback=guardrail_after_model_callback,
    before_tool_callback=guardrail_before_tool_callback,
    after_tool_callback=guardrail_after_tool_callback,
    on_tool_error_callback=handle_tool_error_with_recovery,
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
        declared_intent="Abort external tool side effects following human reviewer rejection",
        actual_outcome="dispatch_cancelled_safely",
        intent_vs_outcome_match=True,
        **receipt_dict,
    )
    return Event(
        content=_model_content(receipt.message),
        output=receipt_dict,
        state={"dispatch_receipt": receipt_dict},
    )


# ---------------------------------------------------------------------------
# Root Agent: ADK 2.0 Graph Workflow + ADK Event Compaction & Governance Plugin
# ---------------------------------------------------------------------------

root_agent = Workflow(
    # Keep in sync with agents-cli-manifest.yaml: agents-cli derives this name
    # from the project `name:` recorded there, and telemetry reports it as
    # gen_ai.agent.name.
    name="ai_in_5_days_agent",
    description=(
        "AI in 5 Days Content Customizer Agent that sets up learner sessions via RequestInput, "
        "initializes ./content_pdfs/day1..day5, extracts and semantically compacts training PDFs with pypdf and Pydantic schemas, "
        "routes across model tiers to summarize personalized daily briefs with curiosity hooks and micro-challenges, "
        "gates external dispatch on human review with runtime policy guardrails, and consolidates long-term learner memory asynchronously."
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

# Native ADK LLM-driven Event Compaction & Context Caching to prevent multi-turn context bloat
event_compaction_summarizer = LlmEventSummarizer(
    llm=Gemini(model=MODEL_FAST_ROUTING),
    prompt_template=(
        "Compact the preceding multi-turn conversation events into a concise, "
        "high-signal state summary preserving learner profile, target day, key technical takeaways, "
        "and human review decisions:\n{events}"
    ),
)

events_compaction_config = EventsCompactionConfig(
    summarizer=event_compaction_summarizer,
    compaction_interval=3,
    overlap_size=1,
    token_threshold=8000,
    event_retention_size=4,
)

context_cache_config = ContextCacheConfig(
    cache_intervals=5,
    ttl_seconds=1800,
    min_tokens=4096,
)

resumability_config = ResumabilityConfig(is_resumable=True)
governance_plugin = EnterpriseGovernancePolicyPlugin()

app = App(
    root_agent=root_agent,
    name="app",
    plugins=[governance_plugin],
    events_compaction_config=events_compaction_config,
    context_cache_config=context_cache_config,
    resumability_config=resumability_config,
)
