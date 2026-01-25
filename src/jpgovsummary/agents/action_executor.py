"""
Action Executor for Plan-Action architecture.

This module executes ActionPlan steps by invoking appropriate sub-agents
and collecting results in ExecutionState.
"""

from .. import Model, logger
from ..state_v2 import (
    ActionStep,
    CompletedAction,
    DocumentSummaryResult,
    ExecutionState,
)
from ..subagents import DocumentTypeDetector, PowerPointSummarizer, WordSummarizer
from ..tools.pdf_loader import load_pdf_as_text


class ActionExecutor:
    """
    Action execution agent.

    Executes ActionPlan steps by invoking sub-agents and storing results.
    """

    def __init__(self, model: Model | None = None):
        """
        Initialize ActionExecutor.

        Args:
            model: Model instance for LLM access. If None, uses default Model().
        """
        self.model = model if model is not None else Model()

        # Initialize sub-agents
        self.document_type_detector = DocumentTypeDetector(model=self.model)
        self.powerpoint_summarizer = PowerPointSummarizer(model=self.model)
        self.word_summarizer = WordSummarizer(model=self.model)

    def execute_plan(self, state: ExecutionState) -> ExecutionState:
        """
        Execute all steps in the action plan.

        Args:
            state: ExecutionState with plan to execute

        Returns:
            Updated ExecutionState with results
        """
        plan = state["plan"]
        current_index = state.get("current_step_index", 0)

        logger.info(f"Executing action plan with {len(plan.steps)} steps")
        logger.info(f"Plan reasoning: {plan.reasoning}")

        # Execute each step sequentially
        for i, step in enumerate(plan.steps):
            if i < current_index:
                # Already executed
                continue

            logger.info(f"\n{'='*80}")
            logger.info(f"Executing step {i+1}/{len(plan.steps)}: {step.action_type}")
            logger.info(f"Target: {step.target}")
            logger.info(f"{'='*80}\n")

            try:
                result = self._execute_step(step, state)

                # Record completion
                completed_action = CompletedAction(
                    step=step,
                    result=result,
                    tokens_used=result.get("tokens_used") if isinstance(result, dict) else None,
                    success=True,
                )

                # Update state
                state["completed_actions"].append(completed_action)
                state["current_step_index"] = i + 1

                logger.info(f"✅ Step {i+1} completed successfully")

            except Exception as e:
                logger.error(f"❌ Step {i+1} failed: {e}")
                import traceback

                traceback.print_exc()

                # Record failure
                completed_action = CompletedAction(
                    step=step,
                    result=None,
                    success=False,
                    error_message=str(e),
                )

                state["completed_actions"].append(completed_action)
                state["errors"].append(f"Step {i+1} ({step.action_type}): {str(e)}")
                state["current_step_index"] = i + 1

                # Continue to next step (don't fail entire plan)
                continue

        logger.info(f"\n{'='*80}")
        logger.info("Plan execution completed")
        logger.info(f"Successful steps: {sum(1 for a in state['completed_actions'] if a.success)}/{len(plan.steps)}")
        logger.info(f"Failed steps: {len(state['errors'])}")
        logger.info(f"{'='*80}\n")

        return state

    def _execute_step(self, step: ActionStep, state: ExecutionState) -> dict:
        """
        Execute a single action step.

        Args:
            step: ActionStep to execute
            state: Current ExecutionState

        Returns:
            Result dict from the action
        """
        if step.action_type == "summarize_pdf":
            return self._execute_summarize_pdf(step, state)
        elif step.action_type == "integrate_summaries":
            return self._execute_integrate_summaries(step, state)
        elif step.action_type == "finalize":
            return self._execute_finalize(step, state)
        elif step.action_type == "post_to_bluesky":
            return self._execute_post_to_bluesky(step, state)
        else:
            raise ValueError(f"Unknown action type: {step.action_type}")

    def _execute_summarize_pdf(self, step: ActionStep, state: ExecutionState) -> dict:
        """
        Execute PDF summarization step.

        Steps:
        1. Load PDF
        2. Detect document type
        3. Invoke appropriate summarizer (PowerPoint or Word)
        4. Store result in state
        """
        url = step.target

        logger.info(f"Loading PDF from: {url}")
        pdf_pages = load_pdf_as_text(url)
        logger.info(f"Loaded {len(pdf_pages)} pages")

        # Detect document type
        logger.info("Detecting document type...")
        detection_result = self.document_type_detector.invoke(
            {
                "pdf_pages": pdf_pages[:10],  # First 10 pages for detection
                "url": url,
            }
        )

        document_type = detection_result["document_type"]
        confidence_scores = detection_result["confidence_scores"]

        logger.info(f"Detected type: {document_type}")
        logger.info(f"Confidence scores: {confidence_scores}")

        # Select appropriate summarizer
        if document_type == "PowerPoint":
            logger.info("Using PowerPointSummarizer sub-agent")
            summarizer_result = self.powerpoint_summarizer.invoke(
                {
                    "pdf_pages": pdf_pages,
                    "url": url,
                }
            )
        elif document_type == "Word":
            logger.info("Using WordSummarizer sub-agent")
            summarizer_result = self.word_summarizer.invoke(
                {
                    "pdf_pages": pdf_pages,
                    "url": url,
                }
            )
        else:
            # Fallback: Try Word summarizer for other types
            logger.warning(f"Unsupported type '{document_type}', falling back to WordSummarizer")
            summarizer_result = self.word_summarizer.invoke(
                {
                    "pdf_pages": pdf_pages,
                    "url": url,
                }
            )

        # Extract summary
        summary = summarizer_result.get("summary", "")
        title = summarizer_result.get("title", url.split("/")[-1])

        logger.info(f"Generated summary: {len(summary)} characters")

        # Create document summary result
        doc_summary = DocumentSummaryResult(
            url=url,
            name=title,
            summary=summary,
            document_type=document_type,
        )

        # Store in state
        state["document_summaries"].append(doc_summary)

        return {
            "document_type": document_type,
            "summary_length": len(summary),
            "title": title,
        }

    def _execute_integrate_summaries(self, step: ActionStep, state: ExecutionState) -> dict:
        """
        Execute summary integration step.

        Combines overview (if exists) + all document summaries into final summary.
        """
        overview = step.params.get("overview")
        step.params.get("document_count", 0)
        document_summaries = state.get("document_summaries", [])

        logger.info("Integrating summaries:")
        logger.info(f"  - Overview: {'Yes' if overview else 'No'}")
        logger.info(f"  - Document summaries: {len(document_summaries)}")

        # Build integrated summary
        parts = []

        if overview:
            parts.append(f"# 会議概要\n\n{overview}")

        if document_summaries:
            parts.append("\n\n# 関連資料の要約\n")
            for i, doc_summary in enumerate(document_summaries, 1):
                parts.append(f"\n## {i}. {doc_summary.name}\n")
                parts.append(f"**種類:** {doc_summary.document_type}\n\n")
                parts.append(doc_summary.summary)

        if parts:
            integrated_summary = "\n".join(parts)
        else:
            # No overview or summaries - should not happen
            integrated_summary = "(要約なし)"

        state["final_summary"] = integrated_summary

        logger.info(f"Integrated summary: {len(integrated_summary)} characters")

        return {
            "summary_length": len(integrated_summary),
            "document_count": len(document_summaries),
        }

    def _execute_finalize(self, step: ActionStep, state: ExecutionState) -> dict:
        """
        Execute finalization step.

        Handles human review (if not batch mode) and character limit checks.
        """
        batch = step.params.get("batch", False)
        final_summary = state.get("final_summary", "")

        logger.info("Finalizing summary:")
        logger.info(f"  - Batch mode: {batch}")
        logger.info(f"  - Summary length: {len(final_summary)} characters")

        # Character limit check
        max_chars = 2000  # From bluesky_poster.py
        if len(final_summary) > max_chars:
            logger.warning(f"Summary exceeds {max_chars} characters, needs truncation")
            # TODO: Implement smart truncation or re-summarization
            # For now, just truncate
            final_summary = final_summary[:max_chars] + "..."

        if batch:
            # Batch mode - skip human review
            logger.info("Batch mode - skipping human review")
            state["final_review_summary"] = final_summary
            state["review_approved"] = True
            state["review_completed"] = True
        else:
            # Interactive mode - would implement human review here
            # For now, auto-approve
            logger.info("Interactive mode - auto-approving for now")
            # TODO: Implement interactive review using summary_finalizer logic
            state["final_review_summary"] = final_summary
            state["review_approved"] = True
            state["review_completed"] = True

        return {
            "final_summary_length": len(state["final_review_summary"]),
            "approved": state["review_approved"],
        }

    def _execute_post_to_bluesky(self, step: ActionStep, state: ExecutionState) -> dict:
        """
        Execute Bluesky posting step.

        Posts the finalized summary to Bluesky.
        """
        final_summary = state.get("final_review_summary", "")

        if not final_summary:
            logger.warning("No final summary to post")
            return {"posted": False, "reason": "No summary"}

        # TODO: Implement Bluesky posting using bluesky_poster logic
        logger.info("Bluesky posting - not yet implemented in v2")
        logger.info(f"Would post: {len(final_summary)} characters")

        state["bluesky_post_content"] = final_summary
        state["bluesky_post_response"] = "Not implemented"

        return {
            "posted": False,
            "reason": "Not yet implemented in v2",
        }
