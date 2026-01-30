"""
State definitions for Plan-Action architecture (v2).

This module defines lightweight state structures for the new Plan-Action workflow,
replacing the monolithic State from state.py with modular, purpose-specific states.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# ============================================================================
# Action Plan Models (Planning Phase)
# ============================================================================


class ActionStep(BaseModel):
    """Individual action step in the execution plan."""

    action_type: Literal[
        "summarize_pdf",
        "extract_html",
        "detect_document_type",
        "generate_initial_overview",
        "score_documents",
        "summarize_selected_documents",
        "integrate_summaries",
        "finalize",
        "post_to_bluesky",
    ] = Field(description="The type of action to perform")
    target: str = Field(description="URL or file path to process")
    params: dict[str, Any] = Field(
        default_factory=dict, description="Additional parameters for the action"
    )
    priority: int = Field(default=0, description="Priority level (lower = higher priority)")
    estimated_tokens: int | None = Field(
        default=None, description="Estimated token consumption for this step"
    )


class ActionPlan(BaseModel):
    """Execution plan generated during Planning phase."""

    steps: list[ActionStep] = Field(description="List of actions to execute in order")
    reasoning: str = Field(description="Explanation of why this plan was chosen")
    total_estimated_tokens: int | None = Field(
        default=None, description="Total estimated token consumption"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Timestamp when plan was created",
    )


# ============================================================================
# Document Discovery Models (v2-specific)
# ============================================================================


class DiscoveredDocument(BaseModel):
    """Document discovered from HTML content (v2 architecture)."""

    url: str = Field(description="Document URL (absolute path)")
    name: str = Field(description="Document name/link text")
    category: str = Field(
        description="Document category: agenda, minutes, executive_summary, material, reference, participants, seating, personal_material, announcement, other"
    )


class DiscoveredDocumentList(BaseModel):
    """List of discovered documents from HTML content."""

    documents: list[DiscoveredDocument] = Field(description="List of discovered documents")


# ============================================================================
# Execution Tracking Models
# ============================================================================


class CompletedAction(BaseModel):
    """Record of a completed action step."""

    step: ActionStep = Field(description="The action step that was executed")
    result: Any = Field(description="Result returned by the action")
    tokens_used: int | None = Field(default=None, description="Actual tokens consumed")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Timestamp when action completed",
    )
    success: bool = Field(default=True, description="Whether the action succeeded")
    error_message: str | None = Field(default=None, description="Error message if failed")


class DocumentSummaryResult(BaseModel):
    """Result from a document summarization sub-agent."""

    url: str = Field(description="URL of the summarized document")
    name: str = Field(description="Name of the document")
    summary: str = Field(description="Generated summary content")
    document_type: str | None = Field(
        default=None, description="Detected document type (PowerPoint/Word/etc)"
    )
    category: str | None = Field(
        default=None, description="Document category (agenda/minutes/material/etc)"
    )
    tokens_used: int | None = Field(default=None, description="Tokens consumed")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Timestamp when summary was generated",
    )


# ============================================================================
# Plan-Action State Definitions
# ============================================================================


class PlanState(TypedDict):
    """
    State for Planning phase - lightweight context for plan generation.

    This state contains only the minimum information needed to generate
    an execution plan, avoiding the state bloat of the v1 architecture.
    """

    # Input information
    input_url: str  # Source URL or file path
    input_type: Literal["html_meeting", "pdf_file"]  # Type of input

    # Planning outputs (no LLM-generated content in Phase 1)
    main_content: str | None  # Extracted main content from HTML (raw)
    discovered_documents: list[DiscoveredDocument] | None  # Related documents found
    action_plan: ActionPlan | None  # Generated execution plan

    # Meeting summary components (extracted from HTML)
    embedded_agenda: str | None  # Agenda content from HTML main content
    embedded_minutes: str | None  # Minutes content from HTML main content

    # Control flags (inherited from CLI)
    batch: bool  # Run without human interaction
    skip_bluesky_posting: bool  # Skip Bluesky posting
    overview_only: bool  # Generate overview only, skip related documents


class ScoredDocument(BaseModel):
    """Document with scoring information for prioritization."""

    url: str
    name: str
    category: str
    score: float = Field(description="Priority score (higher = more important)")
    reason: str = Field(description="Reason for the score")


class ExecutionState(TypedDict):
    """
    State for Execution phase - tracks action execution and results.

    This state accumulates results from sub-agents without storing
    heavy intermediate data like full message histories.
    """

    # Execution plan
    plan: ActionPlan  # The plan being executed

    # Execution tracking
    current_step_index: int  # Index of current step being executed
    completed_actions: list[CompletedAction]  # History of completed actions

    # Context from Phase 1 (carried over from PlanState)
    main_content: str | None  # Main content extracted from HTML
    embedded_agenda: str | None  # Agenda content from HTML
    embedded_minutes: str | None  # Minutes content from HTML
    input_url: str  # Source URL

    # Results storage (lightweight - only final outputs)
    initial_overview: str | None  # Overview generated in Phase 2 Step 1
    document_summaries: list[DocumentSummaryResult]  # Summaries from sub-agents
    scored_documents: list[ScoredDocument] | None  # Documents after scoring
    final_summary: str | None  # Integrated final summary
    final_review_summary: str | None  # Human-reviewed summary (if not batch)

    # Meeting summary (from agenda/minutes)
    meeting_summary: str | None  # Consolidated meeting summary
    meeting_summary_sources: dict | None  # Track what sources were used

    # Review session (for interactive mode)
    review_session: dict | None  # Q&A history and improvements
    review_approved: bool | None  # Human approval status
    review_completed: bool  # Review process completed

    # Bluesky posting
    bluesky_post_content: str | None  # Content posted to Bluesky
    bluesky_post_response: str | None  # Response from Bluesky API

    # Error handling
    errors: list[str]  # List of error messages encountered


# ============================================================================
# Sub-Agent States (Isolated Contexts)
# ============================================================================


class PowerPointState(TypedDict):
    """
    State for PowerPointSummarizer sub-agent.

    This is an isolated context - no dependency on main workflow state.
    """

    # Input
    pdf_pages: list[str]  # Text content of PDF pages
    url: str  # Source URL for reference

    # Intermediate results
    title: str | None  # Extracted presentation title
    scored_slides: list[dict] | None  # Slides with importance scores
    selected_content: str | None  # Content from high-scoring slides

    # Output
    summary: str | None  # Final summary


class WordState(TypedDict):
    """
    State for WordSummarizer sub-agent.

    This is an isolated context for Word document processing.
    """

    # Input
    pdf_pages: list[str]  # Text content of PDF pages
    url: str  # Source URL for reference

    # Intermediate results
    title: str | None  # Extracted document title
    table_of_contents: list[dict] | None  # Extracted TOC structure

    # Output
    summary: str | None  # Final summary


class DocumentTypeDetectorState(TypedDict):
    """
    State for DocumentTypeDetector sub-agent.

    Lightweight state for document classification.
    """

    # Input
    pdf_pages: list[str]  # First 10 pages of PDF
    url: str  # Source URL for reference

    # Output
    document_type: str  # Detected type (PowerPoint/Word/Agenda/etc)
    confidence_scores: dict[str, float]  # Confidence for each type
    detection_detail: str  # Explanation of detection reasoning


class HTMLProcessorState(TypedDict):
    """
    State for HTMLProcessor sub-agent.

    Handles HTML loading, conversion, and main content extraction.
    """

    # Input
    url: str  # HTML page URL

    # Intermediate
    html_bytes: bytes | None  # Raw HTML content
    markdown: str | None  # Converted markdown

    # Output
    main_content: str | None  # Extracted main content (headers/footers removed)
    discovered_documents: list[DiscoveredDocument] | None  # Discovered related documents

    # Meeting summary extraction (from main content)
    agenda_content: str | None  # Extracted agenda section
    minutes_content: str | None  # Extracted minutes section
    has_embedded_agenda: bool  # Flag if agenda found in main content
    has_embedded_minutes: bool  # Flag if minutes found in main content


# ============================================================================
# Backward Compatibility Helpers
# ============================================================================


def convert_v1_to_plan_state(v1_state: dict) -> PlanState:
    """
    Convert v1 State to PlanState for gradual migration.

    Args:
        v1_state: State dict from the v1 architecture

    Returns:
        PlanState compatible with v2 architecture
    """
    return PlanState(
        input_url=v1_state.get("url", ""),
        input_type="html_meeting" if v1_state.get("is_meeting_page") else "pdf_file",
        overview=v1_state.get("overview"),
        discovered_documents=[
            r.url for r in v1_state.get("candidate_reports", {}).get("reports", [])
        ]
        if v1_state.get("candidate_reports")
        else None,
        action_plan=None,
        batch=v1_state.get("batch", False),
        skip_bluesky_posting=v1_state.get("skip_bluesky_posting", False),
        overview_only=v1_state.get("overview_only", False),
    )


def convert_plan_to_v1_state(plan_state: PlanState, execution_state: ExecutionState) -> dict:
    """
    Convert PlanState + ExecutionState to v1 State format.

    This enables gradual migration - v2 agents can output results
    compatible with v1 finalizer/poster agents.

    Args:
        plan_state: PlanState from Planning phase
        execution_state: ExecutionState from Execution phase

    Returns:
        Dict compatible with v1 State structure
    """
    from jpgovsummary.state import Summary

    return {
        "url": plan_state["input_url"],
        "overview": plan_state.get("overview"),
        "final_summary": execution_state.get("final_summary"),
        "final_review_summary": execution_state.get("final_review_summary"),
        "target_report_summaries": [
            Summary(
                url=s.url,
                name=s.name,
                content=s.summary,
                document_type=s.document_type,
            )
            for s in execution_state.get("document_summaries", [])
        ],
        "batch": plan_state.get("batch", False),
        "skip_bluesky_posting": plan_state.get("skip_bluesky_posting", False),
        "review_session": execution_state.get("review_session"),
        "review_approved": execution_state.get("review_approved"),
        "review_completed": execution_state.get("review_completed", False),
        "bluesky_post_content": execution_state.get("bluesky_post_content"),
        "bluesky_post_response": execution_state.get("bluesky_post_response"),
    }
