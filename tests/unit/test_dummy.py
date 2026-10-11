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
"""Unit tests for the AI in 5 Days Content Customizer Agent workflow nodes, schemas, and tools."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from google.adk import Workflow
from google.adk.events import RequestInput

from app.agent import (
    DailyBrief,
    SummarizerInput,
    cancel_dispatch,
    content_summarizer,
    dispatch_brief,
    execute_dispatch_tools,
    human_review,
    init_workspace,
    pdf_reader,
    root_agent,
    setup_session,
)
from app.tools import (
    ExtractedPdfContent,
    extract_pdf_from_workspace,
    initialize_pdf_workspace,
    post_google_chat_card,
    send_email_brief,
)


def test_workflow_graph_topology() -> None:
    """Verify root_agent is an ADK 2.0 Workflow containing all 6 required nodes."""
    assert isinstance(root_agent, Workflow)
    node_names = {n.name for n in root_agent.graph.nodes}
    for expected in (
        "setup_session",
        "init_workspace",
        "pdf_reader",
        "content_summarizer",
        "human_review",
        "dispatch_brief",
        "cancel_dispatch",
    ):
        assert expected in node_names, f"Missing workflow node: {expected}"

    assert content_summarizer.input_schema is SummarizerInput
    assert content_summarizer.output_schema is DailyBrief
    assert dispatch_brief.input_schema is DailyBrief
    tool_names = {t.__name__ for t in dispatch_brief.tools}
    assert "send_email_brief" in tool_names
    assert "post_google_chat_card" in tool_names


@pytest.mark.asyncio
async def test_setup_session_request_input_loop() -> None:
    """Verify setup_session asks for Name, Email, and Technical Track sequentially via RequestInput."""
    ctx = MagicMock()
    ctx.state = {}
    ctx.resume_inputs = {}
    ctx.user_content = None

    # 1. Missing name -> asks for user_name
    events_1 = [ev async for ev in setup_session._func(ctx, "Start session")]
    req_1 = [e for e in events_1 if isinstance(e, RequestInput)]
    assert len(req_1) == 1
    assert req_1[0].interrupt_id == "ask_user_name"

    # 2. Provide name -> asks for user_email
    ctx.resume_inputs = {"ask_user_name": "Alex Rivera"}
    events_2 = [ev async for ev in setup_session._func(ctx, "Start session")]
    req_2 = [e for e in events_2 if isinstance(e, RequestInput)]
    assert len(req_2) == 1
    assert req_2[0].interrupt_id == "ask_user_email"

    # 3. Provide email -> asks for technical_track
    ctx.resume_inputs = {
        "ask_user_name": "Alex Rivera",
        "ask_user_email": "alex@example.com",
    }
    events_3 = [ev async for ev in setup_session._func(ctx, "Start session")]
    req_3 = [e for e in events_3 if isinstance(e, RequestInput)]
    assert len(req_3) == 1
    assert req_3[0].interrupt_id == "ask_technical_track"

    # 4. Provide track -> emits INIT_WORKSPACE route with validated UserProfile
    ctx.resume_inputs = {
        "ask_user_name": "Alex Rivera",
        "ask_user_email": "alex@example.com",
        "ask_technical_track": "Architect",
    }
    events_4 = [ev async for ev in setup_session._func(ctx, "Start session")]
    final_ev = events_4[-1]
    assert final_ev.actions.route == "INIT_WORKSPACE"
    assert final_ev.output["user_name"] == "Alex Rivera"
    assert final_ev.output["user_email"] == "alex@example.com"
    assert final_ev.output["technical_track"] == "Architect"


def test_init_workspace_and_pdf_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify init_workspace creates day1..day5 and pdf_reader extracts valid PDF text with pypdf."""
    workspace_dir = str(tmp_path / "content_pdfs")
    monkeypatch.setattr("app.agent.DEFAULT_WORKSPACE_ROOT", workspace_dir)
    monkeypatch.setattr("app.tools.DEFAULT_WORKSPACE_ROOT", workspace_dir)

    ws_status = initialize_pdf_workspace(
        target_day="day2",
        workspace_root=workspace_dir,
        user_name="Sam Chen",
        user_email="sam@company.com",
        technical_track="Engineer",
    )
    assert ws_status.created_days == ["day1", "day2", "day3", "day4", "day5"]
    for d in ("day1", "day2", "day3", "day4", "day5"):
        assert (tmp_path / "content_pdfs" / d).is_dir()

    extracted = extract_pdf_from_workspace(
        target_day="day2",
        workspace_root=workspace_dir,
        user_name="Sam Chen",
        user_email="sam@company.com",
        technical_track="Engineer",
    )
    assert isinstance(extracted, ExtractedPdfContent)
    assert extracted.page_count >= 1
    assert len(extracted.extracted_text) > 50

    ctx = MagicMock()
    ctx.state = {}
    ws_ev = init_workspace(
        {
            "user_name": "Sam Chen",
            "user_email": "sam@company.com",
            "technical_track": "Engineer",
            "target_day": "day2",
        },
        ctx,
    )
    pdf_ev = pdf_reader(ws_ev.output, ctx)
    validated_input = SummarizerInput(**pdf_ev.output)
    assert validated_input.target_day == "day2"
    assert validated_input.technical_track == "Engineer"
    assert len(validated_input.raw_content) > 50


@pytest.mark.asyncio
async def test_human_review_and_dispatch_routing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify human_review pauses with RequestInput and routes APPROVED vs REJECTED accurately."""
    monkeypatch.setenv("DISPATCHED_BRIEFS_DIR", str(tmp_path / "outbox"))
    sample_brief = DailyBrief(
        day="day1",
        technical_track="Architect",
        user_name="Alex Rivera",
        user_email="alex@example.com",
        title="Day 1 Architect Brief",
        executive_summary="Architectural patterns for ADK 2.0 tool boundaries.",
        key_takeaways=["Use Pydantic schemas", "Enforce HITL approval"],
        curiosity_hooks=["How do you prevent unapproved external side effects?"],
        micro_challenge="Audit a workflow graph for unguarded tool nodes.",
    )

    ctx = MagicMock()
    ctx.state = {}
    ctx.resume_inputs = {}

    # 1. Without resume input -> pauses with RequestInput(interrupt_id="confirm_dispatch")
    pause_events = [ev async for ev in human_review._func(sample_brief.model_dump(), ctx)]
    req_inputs = [e for e in pause_events if isinstance(e, RequestInput)]
    assert len(req_inputs) == 1
    assert req_inputs[0].interrupt_id == "confirm_dispatch"

    # 2. Resume with "approve" -> routes to APPROVED and dispatches Email + Chat Card v2
    ctx.resume_inputs = {"confirm_dispatch": "approve"}
    approve_events = [ev async for ev in human_review._func(sample_brief.model_dump(), ctx)]
    assert approve_events[-1].actions.route == "APPROVED"

    receipt = execute_dispatch_tools(approve_events[-1].output, ctx)
    assert receipt.status == "dispatched"
    assert receipt.user_email == "alex@example.com"

    # 3. Resume with "reject" -> routes to REJECTED and cancels dispatch safely
    ctx.resume_inputs = {"confirm_dispatch": "reject"}
    reject_events = [ev async for ev in human_review._func(sample_brief.model_dump(), ctx)]
    assert reject_events[-1].actions.route == "REJECTED"

    cancel_ev = cancel_dispatch(reject_events[-1].output, ctx)
    assert cancel_ev.output["status"] == "cancelled"
    assert cancel_ev.output["email_status"] == "skipped_by_human_review"


def test_dispatch_tools_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify send_email_brief and post_google_chat_card return structured receipts."""
    monkeypatch.setenv("DISPATCHED_BRIEFS_DIR", str(tmp_path / "outbox"))
    email_res = send_email_brief(
        recipient_email="alex@example.com",
        user_name="Alex Rivera",
        day="day1",
        track="Architect",
        subject="Day 1 Brief",
        body_markdown="# Summary",
    )
    assert email_res["status"] == "sent"
    assert email_res["recipient_email"] == "alex@example.com"

    chat_res = post_google_chat_card(
        user_name="Alex Rivera",
        day="day1",
        track="Architect",
        title="Day 1 Brief",
        summary="Executive summary",
        curiosity_hook="Why use Pydantic?",
        micro_challenge="Write a schema",
    )
    assert chat_res["status"] == "posted"
    assert "cardsV2" in chat_res["card_v2"]
