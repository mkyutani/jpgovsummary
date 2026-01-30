"""
Action Planner for Plan-Action architecture.

This module contains the planning agent that analyzes input (HTML meeting page
or PDF file) and generates an ActionPlan with prioritized execution steps.
"""

from .. import Model, logger
from ..state_v2 import ActionPlan, ActionStep, PlanState
from ..subagents import HTMLProcessor


class ActionPlanner:
    """
    Action planning agent.

    Analyzes input and generates execution plan with prioritized steps.
    """

    def __init__(self, model: Model | None = None):
        """
        Initialize ActionPlanner.

        Args:
            model: Model instance for LLM access. If None, uses default Model().
        """
        self.model = model if model is not None else Model()
        self.html_processor = HTMLProcessor(model=self.model)

    def analyze_input_and_plan(self, state: PlanState) -> PlanState:
        """
        Analyze input and generate action plan.

        Args:
            state: PlanState with input_url and input_type

        Returns:
            Updated PlanState with action_plan
        """
        state["input_url"]
        input_type = state["input_type"]
        state.get("overview_only", False)

        logger.info(f"Planning for input_type: {input_type}")

        if input_type == "html_meeting":
            return self._plan_html_meeting(state)
        else:  # pdf_file
            return self._plan_pdf_file(state)

    def _plan_html_meeting(self, state: PlanState) -> PlanState:
        """
        Plan for HTML meeting page.

        Phase 1 (lightweight):
        1. Call HTMLProcessor sub-agent to extract content and discover documents
        2. Generate ActionPlan (NO overview generation here - that's Phase 2)
        """
        input_url = state["input_url"]
        overview_only = state.get("overview_only", False)

        logger.info("HTML meeting page processing plan:")
        logger.info("  1. Extract main content from HTML")
        logger.info("  2. Discover related documents")
        logger.info("  3. Generate action plan (no LLM calls)")

        # Step 1: Extract HTML main content (only LLM calls in Phase 1)
        try:
            html_result = self.html_processor.invoke({"url": input_url})

            main_content = html_result.get("main_content")
            discovered_documents = html_result.get("discovered_documents", [])

            # Extract embedded meeting content
            embedded_agenda = html_result.get("agenda_content")
            embedded_minutes = html_result.get("minutes_content")

            # Filter documents by category
            meeting_related_docs = [
                doc for doc in discovered_documents if doc.category in ["agenda", "minutes"]
            ]
            other_docs = [
                doc
                for doc in discovered_documents
                if doc.category not in ["agenda", "minutes", "participants", "seating"]
            ]

            logger.info(f"Discovered {len(discovered_documents)} related documents")
            logger.info(f"  - Meeting-related: {len(meeting_related_docs)} (agenda/minutes)")
            logger.info(f"  - Other documents: {len(other_docs)}")

            if not main_content:
                logger.warning("Failed to extract main content from HTML")
                return {
                    "main_content": None,
                    "discovered_documents": [],
                    "embedded_agenda": None,
                    "embedded_minutes": None,
                    "action_plan": ActionPlan(
                        steps=[],
                        reasoning="Failed to extract main content from HTML meeting page",
                    ),
                }

            # Build action plan - NO overview generation in Phase 1
            steps = []
            priority = 0

            if overview_only:
                # Overview only mode - just create initial overview, no PDF processing
                steps.append(
                    ActionStep(
                        action_type="generate_initial_overview",
                        target=input_url,
                        params={
                            "main_content": main_content,
                            "embedded_agenda": embedded_agenda,
                            "embedded_minutes": embedded_minutes,
                        },
                        priority=priority,
                        estimated_tokens=2000,
                    )
                )
                priority += 1
                reasoning = "Overview only mode - generating overview from HTML content only"
            elif not discovered_documents:
                # No documents found - just generate overview
                steps.append(
                    ActionStep(
                        action_type="generate_initial_overview",
                        target=input_url,
                        params={
                            "main_content": main_content,
                            "embedded_agenda": embedded_agenda,
                            "embedded_minutes": embedded_minutes,
                        },
                        priority=priority,
                        estimated_tokens=2000,
                    )
                )
                priority += 1
                reasoning = "No related documents found - overview only from HTML content"
            else:
                # Full processing flow

                # Step 1: Summarize meeting-related documents (agenda, minutes) FIRST
                # These are processed in parallel
                for doc in meeting_related_docs:
                    steps.append(
                        ActionStep(
                            action_type="summarize_pdf",
                            target=doc.url,
                            params={
                                "source_meeting_url": input_url,
                                "category": doc.category,
                            },
                            priority=priority,  # Same priority for parallel execution
                            estimated_tokens=5000,
                        )
                    )
                # Move to next priority after all meeting docs
                if meeting_related_docs:
                    priority += 1

                # Step 2: Generate initial overview (after agenda/minutes are processed)
                steps.append(
                    ActionStep(
                        action_type="generate_initial_overview",
                        target=input_url,
                        params={
                            "main_content": main_content,
                            "embedded_agenda": embedded_agenda,
                            "embedded_minutes": embedded_minutes,
                        },
                        priority=priority,
                        estimated_tokens=2000,
                    )
                )
                priority += 1

                # Step 3: Score and select other documents
                if other_docs:
                    steps.append(
                        ActionStep(
                            action_type="score_documents",
                            target=input_url,
                            params={
                                "documents": [
                                    {"url": doc.url, "name": doc.name, "category": doc.category}
                                    for doc in other_docs
                                ],
                            },
                            priority=priority,
                            estimated_tokens=1000,
                        )
                    )
                    priority += 1

                    # Step 4: Summarize selected high-score documents
                    # (actual targets determined at runtime by score_documents)
                    steps.append(
                        ActionStep(
                            action_type="summarize_selected_documents",
                            target=input_url,
                            params={
                                "max_documents": 5,  # Limit to top 5
                            },
                            priority=priority,
                            estimated_tokens=15000,
                        )
                    )
                    priority += 1

                reasoning = (
                    f"HTML meeting with {len(discovered_documents)} documents. "
                    f"Flow: Process agenda/minutes PDFs → Generate initial overview → "
                    f"Score other docs → Summarize top docs → Integrate."
                )

            # Add integration step
            steps.append(
                ActionStep(
                    action_type="integrate_summaries",
                    target=input_url,
                    params={},
                    priority=priority,
                    estimated_tokens=2000,
                )
            )
            priority += 1

            # Add finalization step
            steps.append(
                ActionStep(
                    action_type="finalize",
                    target=input_url,
                    params={"batch": state.get("batch", False)},
                    priority=priority,
                    estimated_tokens=500,
                )
            )
            priority += 1

            # Add Bluesky posting step if not skipped
            if not state.get("skip_bluesky_posting", False):
                steps.append(
                    ActionStep(
                        action_type="post_to_bluesky",
                        target=input_url,
                        params={},
                        priority=priority,
                        estimated_tokens=100,
                    )
                )

            action_plan = ActionPlan(
                steps=steps,
                reasoning=reasoning,
                total_estimated_tokens=sum(s.estimated_tokens or 0 for s in steps),
            )

            return {
                "main_content": main_content,
                "discovered_documents": discovered_documents,
                "embedded_agenda": embedded_agenda,
                "embedded_minutes": embedded_minutes,
                "action_plan": action_plan,
            }

        except Exception as e:
            logger.error(f"Error during HTML meeting planning: {e}")
            import traceback

            traceback.print_exc()

            return {
                "main_content": None,
                "discovered_documents": [],
                "embedded_agenda": None,
                "embedded_minutes": None,
                "action_plan": ActionPlan(
                    steps=[],
                    reasoning=f"Error during planning: {str(e)}",
                ),
            }

    def _plan_pdf_file(self, state: PlanState) -> PlanState:
        """
        Plan for single PDF file.

        Steps:
        1. Detect document type
        2. Summarize document
        3. Finalize
        """
        input_url = state["input_url"]

        logger.info("PDF file processing plan:")
        logger.info("  1. Detect document type")
        logger.info("  2. Summarize document")
        logger.info("  3. Finalize summary")

        steps = [
            ActionStep(
                action_type="summarize_pdf",
                target=input_url,
                params={},
                priority=0,
                estimated_tokens=5000,
            ),
            ActionStep(
                action_type="integrate_summaries",
                target=input_url,
                params={},
                priority=1,
                estimated_tokens=1000,
            ),
            ActionStep(
                action_type="finalize",
                target=input_url,
                params={"batch": state.get("batch", False)},
                priority=2,
                estimated_tokens=500,
            ),
        ]

        # Add Bluesky posting step if not skipped
        if not state.get("skip_bluesky_posting", False):
            steps.append(
                ActionStep(
                    action_type="post_to_bluesky",
                    target=input_url,
                    params={},
                    priority=3,
                    estimated_tokens=100,
                )
            )

        action_plan = ActionPlan(
            steps=steps,
            reasoning=(
                "Single PDF file processing. Plan: Detect type → Summarize → Integrate → Finalize."
            ),
            total_estimated_tokens=sum(s.estimated_tokens or 0 for s in steps),
        )

        return {
            "main_content": None,  # No main content for single PDF
            "discovered_documents": [],
            "embedded_agenda": None,
            "embedded_minutes": None,
            "action_plan": action_plan,
        }

