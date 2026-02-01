"""
Main workflow for jpgovsummary using Plan-Action architecture (v2).

This is the new Plan-Action based workflow that replaces the monolithic
State-based approach in jpgovwatcher.py. It uses lightweight PlanState
and ExecutionState to minimize token consumption.
"""

import sys
from typing import Literal

from jpgovsummary import Model, logger
from jpgovsummary.agents.action_executor import ActionExecutor
from jpgovsummary.agents.action_planner import ActionPlanner
from jpgovsummary.state_v2 import ExecutionState, PlanState


def detect_input_type(url: str) -> Literal["html_meeting", "pdf_file"]:
    """
    Detect input type from URL or file path.

    Args:
        url: URL or file path

    Returns:
        "html_meeting" for HTML pages, "pdf_file" for PDF files
    """
    lower_url = url.lower()

    # Check if it's a PDF file
    if lower_url.endswith(".pdf"):
        return "pdf_file"

    # Check if it's a local file (non-PDF)
    if not (lower_url.startswith("http://") or lower_url.startswith("https://")):
        # Local file that's not PDF - treat as error
        logger.error(f"Unsupported local file type: {url}")
        raise ValueError(f"Local files must be PDF files: {url}")

    # Default: Assume HTML meeting page
    return "html_meeting"


def run_jpgovwatcher_v2(
    url: str,
    model: Model | None = None,
    batch: bool = False,
    skip_bluesky_posting: bool = False,
    overview_only: bool = False,
) -> dict:
    """
    Run jpgovsummary workflow using Plan-Action architecture (v2).

    Args:
        url: URL or file path to process
        model: Model instance for LLM access
        batch: Run without human interaction
        skip_bluesky_posting: Skip Bluesky posting step
        overview_only: Generate overview only, skip related documents

    Returns:
        Dict with final results
    """
    logger.info("="*80)
    logger.info("JPGOVSUMMARY V2 - Plan-Action Architecture")
    logger.info("="*80)

    # Detect input type
    input_type = detect_input_type(url)
    logger.info(f"Input type: {input_type}")
    logger.info(f"Input URL: {url}")

    # Initialize model
    if model is None:
        model = Model()

    # ========================================================================
    # PHASE 1: PLANNING
    # ========================================================================

    logger.info("\n" + "="*80)
    logger.info("PHASE 1: PLANNING")
    logger.info("="*80 + "\n")

    # Initialize planning state
    plan_state: PlanState = {
        "input_url": url,
        "input_type": input_type,
        "main_content": None,
        "structured_overview": None,
        "discovered_documents": None,
        "action_plan": None,
        "batch": batch,
        "skip_bluesky_posting": skip_bluesky_posting,
        "overview_only": overview_only,
    }

    # Create action planner
    planner = ActionPlanner(model=model)

    # Generate action plan
    plan_state = planner.analyze_input_and_plan(plan_state)

    action_plan = plan_state.get("action_plan")
    if not action_plan:
        logger.error("Failed to generate action plan")
        return {
            "success": False,
            "error": "Failed to generate action plan",
        }

    logger.info("\nAction plan generated:")
    logger.info(f"  Steps: {len(action_plan.steps)}")
    logger.info(f"  Reasoning: {action_plan.reasoning}")
    logger.info(f"  Estimated tokens: {action_plan.total_estimated_tokens}")

    # ========================================================================
    # PHASE 2: EXECUTION
    # ========================================================================

    logger.info("\n" + "="*80)
    logger.info("PHASE 2: EXECUTION")
    logger.info("="*80 + "\n")

    # Initialize execution state with context from Phase 1
    execution_state: ExecutionState = {
        "plan": action_plan,
        "current_step_index": 0,
        "completed_actions": [],
        # Context from Phase 1
        "main_content": plan_state.get("main_content"),
        "structured_overview": plan_state.get("structured_overview"),  # Will be generated in Phase 2
        "has_meeting_info": False,  # Determined in Phase 2
        "input_url": url,
        # Results storage
        "initial_summary": None,
        "document_summaries": [],
        "scored_documents": None,
        "final_summary": None,
        "final_review_summary": None,
        # Meeting summary
        "meeting_summary": None,
        "meeting_summary_sources": None,
        # Review
        "review_session": None,
        "review_approved": None,
        "review_completed": False,
        # Bluesky
        "bluesky_post_content": None,
        "bluesky_post_response": None,
        # Errors
        "errors": [],
    }

    # Create action executor (parallel execution enabled by default)
    executor = ActionExecutor(model=model)

    # Execute action plan
    execution_state = executor.execute_plan(execution_state)

    # ========================================================================
    # PHASE 3: OUTPUT
    # ========================================================================

    logger.info("\n" + "="*80)
    logger.info("PHASE 3: OUTPUT")
    logger.info("="*80 + "\n")

    structured_overview = execution_state.get("structured_overview", "")
    initial_summary = execution_state.get("initial_summary", "")
    final_summary = execution_state.get("final_review_summary") or execution_state.get("final_summary")
    document_summaries = execution_state.get("document_summaries", [])

    if not final_summary:
        logger.error("No final summary generated")
        return {
            "success": False,
            "error": "No final summary generated",
            "errors": execution_state.get("errors", []),
        }

    # Build complete output: title → structured_overview → initial_summary → final_summary → document_summaries
    output_parts = []

    # 0. Document title - extract from structured_overview if available
    meeting_title = "会議サマリー"  # Default
    if structured_overview:
        import re
        # Look for "- 会議名: ..." pattern in structured_overview
        match = re.search(r'-\s*会議名[:：]\s*(.+?)(?:\n|$)', structured_overview)
        if match:
            meeting_title = match.group(1).strip()

    output_parts.append(f"# {meeting_title}")

    # 1. Structured overview (bullet point format from Phase 2)
    if structured_overview:
        output_parts.append("\n## 会議概要")
        output_parts.append(structured_overview)

    # 2. Initial summary (prose format from Phase 2)
    if initial_summary:
        output_parts.append("\n## 議事要約")
        output_parts.append(initial_summary)

    # 3. Final summary (integrated overview + URL)
    output_parts.append("\n## 統合要約")
    output_parts.append(final_summary)

    # 4. Individual document summaries with URLs
    if document_summaries:
        output_parts.append("\n## 関連資料")
        for doc_summary in document_summaries:
            output_parts.append(f"\n### {doc_summary.name}")
            output_parts.append(doc_summary.summary)
            if doc_summary.url:
                output_parts.append(f"\n{doc_summary.url}")

    # Output complete summary
    complete_output = "\n".join(output_parts)
    print(complete_output)

    logger.info("\nWorkflow completed successfully")
    logger.info(f"  Structured overview: {len(structured_overview) if structured_overview else 0} characters")
    logger.info(f"  Initial summary: {len(initial_summary) if initial_summary else 0} characters")
    logger.info(f"  Integrated summary: {len(final_summary)} characters")
    logger.info(f"  Document summaries: {len(document_summaries)}")
    logger.info(f"  Complete output: {len(complete_output)} characters")
    logger.info(f"  Errors: {len(execution_state['errors'])}")

    return {
        "success": True,
        "final_summary": final_summary,
        "document_summaries": execution_state["document_summaries"],
        "errors": execution_state["errors"],
    }


def main():
    """Command-line entry point for testing v2 workflow."""
    import argparse

    from jpgovsummary.jpgovwatcher import setup

    setup()

    parser = argparse.ArgumentParser(description="jpgovsummary v2 (Plan-Action architecture)")
    parser.add_argument("url", type=str, help="URL or PDF file path to process")
    parser.add_argument("--batch", action="store_true", help="Run in batch mode (no human interaction)")
    parser.add_argument("--skip-bluesky-posting", action="store_true", help="Skip Bluesky posting")
    parser.add_argument("--overview-only", action="store_true", help="Generate overview only")
    parser.add_argument("--model", type=str, default=None, help="LLM model to use")

    args = parser.parse_args()

    # Initialize model
    if args.model:
        model = Model(args.model)
    else:
        model = Model()

    try:
        result = run_jpgovwatcher_v2(
            url=args.url,
            model=model,
            batch=args.batch,
            skip_bluesky_posting=args.skip_bluesky_posting,
            overview_only=args.overview_only,
        )

        if result["success"]:
            sys.exit(0)
        else:
            logger.error(f"Workflow failed: {result.get('error')}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
