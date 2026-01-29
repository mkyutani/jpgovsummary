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

            logger.info(f"\n{'=' * 80}")
            logger.info(f"Executing step {i + 1}/{len(plan.steps)}: {step.action_type}")
            logger.info(f"Target: {step.target}")
            logger.info(f"{'=' * 80}\n")

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

                logger.info(f"✅ Step {i + 1} completed successfully")

            except Exception as e:
                logger.error(f"❌ Step {i + 1} failed: {e}")
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
                state["errors"].append(f"Step {i + 1} ({step.action_type}): {str(e)}")
                state["current_step_index"] = i + 1

                # Continue to next step (don't fail entire plan)
                continue

        logger.info(f"\n{'=' * 80}")
        logger.info("Plan execution completed")
        logger.info(
            f"Successful steps: {sum(1 for a in state['completed_actions'] if a.success)}/{len(plan.steps)}"
        )
        logger.info(f"Failed steps: {len(state['errors'])}")
        logger.info(f"{'=' * 80}\n")

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
        elif step.action_type == "create_meeting_summary":
            return self._execute_create_meeting_summary(step, state)
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

        # Extract category from step params (passed by planner)
        category = step.params.get("category")

        # Create document summary result
        doc_summary = DocumentSummaryResult(
            url=url,
            name=title,
            summary=summary,
            document_type=document_type,
            category=category,  # Store category
        )

        # Store in state
        state["document_summaries"].append(doc_summary)

        return {
            "document_type": document_type,
            "summary_length": len(summary),
            "title": title,
            "category": category,
        }

    def _execute_create_meeting_summary(self, step: ActionStep, state: ExecutionState) -> dict:
        """
        Execute meeting summary creation step.

        Combines:
        1. Embedded agenda content (from HTML main content)
        2. Embedded minutes content (from HTML main content)
        3. Agenda category document summaries
        4. Minutes category document summaries

        Into a consolidated meeting summary.
        """
        embedded_agenda = step.params.get("embedded_agenda")
        embedded_minutes = step.params.get("embedded_minutes")
        overview = step.params.get("overview")

        # Filter document summaries by category (agenda, minutes only)
        document_summaries = state.get("document_summaries", [])
        meeting_docs = [doc for doc in document_summaries if doc.category in ["agenda", "minutes"]]

        logger.info("Creating meeting summary:")
        logger.info(f"  - Embedded agenda: {'Yes' if embedded_agenda else 'No'}")
        logger.info(f"  - Embedded minutes: {'Yes' if embedded_minutes else 'No'}")
        logger.info(
            f"  - Agenda documents: {len([d for d in meeting_docs if d.category == 'agenda'])}"
        )
        logger.info(
            f"  - Minutes documents: {len([d for d in meeting_docs if d.category == 'minutes'])}"
        )

        # Build combined meeting content
        parts = []

        if embedded_agenda:
            parts.append(f"# 議事次第（HTMLより）\n\n{embedded_agenda}")

        if embedded_minutes:
            parts.append(f"\n\n# 議事録（HTMLより）\n\n{embedded_minutes}")

        # Add agenda document summaries
        agenda_docs = [d for d in meeting_docs if d.category == "agenda"]
        if agenda_docs:
            parts.append("\n\n# 議事次第（関連資料）\n")
            for doc in agenda_docs:
                parts.append(f"\n## {doc.name}\n")
                parts.append(doc.summary)

        # Add minutes document summaries
        minutes_docs = [d for d in meeting_docs if d.category == "minutes"]
        if minutes_docs:
            parts.append("\n\n# 議事録（関連資料）\n")
            for doc in minutes_docs:
                parts.append(f"\n## {doc.name}\n")
                parts.append(doc.summary)

        combined_content = "\n".join(parts) if parts else ""

        if not combined_content:
            logger.warning("No meeting content to summarize")
            return {"meeting_summary_length": 0}

        # Use LLM to create integrated meeting summary
        llm = self.model.llm()

        from langchain.prompts import PromptTemplate

        meeting_summary_prompt = PromptTemplate(
            input_variables=["content", "overview"],
            template="""あなたは会議の議事要約を作成する専門家です。

以下の情報から、会議の議事要約を作成してください。

# 会議概要
{overview}

# 議事関連コンテンツ
{content}

# 要約作成手順

ステップ1: 議事の構造を把握する
- 議題の流れと構成を理解
- 主要な議論項目を特定
- 決定事項とアクションアイテムを抽出

ステップ2: 議事要約を作成する
- 会議で議論された主要なトピック
- 各議題での重要な議論内容
- 決定事項・合意事項
- 今後のアクションアイテム
- 次回会議の予定（あれば）

ステップ3: 簡潔にまとめる
- 冗長な表現を避ける
- 重要な情報を優先
- 議事の流れを保持

# 出力形式
議事要約のみを出力してください（マークダウン見出しは不要、本文のみ）

# 文量
500-1500文字程度

# 制約
- 推測や補完は行わない
- 提供されたコンテンツに記載されている内容のみを使用
- 「について：」などの空虚な表現は避ける
- 具体的な内容を含める
""",
        )

        chain = meeting_summary_prompt | llm

        try:
            result = chain.invoke(
                {
                    "content": combined_content[:15000],  # Limit to 15K chars
                    "overview": overview or "",
                }
            )

            meeting_summary = result.content.strip()

            logger.info(f"Created meeting summary: {len(meeting_summary)} characters")

            # Store in state
            state["meeting_summary"] = meeting_summary
            state["meeting_summary_sources"] = {
                "embedded_agenda": bool(embedded_agenda),
                "embedded_minutes": bool(embedded_minutes),
                "agenda_docs": len(agenda_docs),
                "minutes_docs": len(minutes_docs),
            }

            return {"meeting_summary_length": len(meeting_summary)}

        except Exception as e:
            logger.error(f"Error creating meeting summary: {e}")
            import traceback

            traceback.print_exc()
            return {"meeting_summary_length": 0}

    def _execute_integrate_summaries(self, step: ActionStep, state: ExecutionState) -> dict:
        """
        Execute summary integration step.

        Combines overview + meeting summary + other document summaries into final summary.
        """
        overview = step.params.get("overview")
        step.params.get("document_count", 0)
        document_summaries = state.get("document_summaries", [])
        meeting_summary = state.get("meeting_summary")  # New: meeting summary

        logger.info("Integrating summaries:")
        logger.info(f"  - Overview: {'Yes' if overview else 'No'}")
        logger.info(f"  - Meeting summary: {'Yes' if meeting_summary else 'No'}")
        logger.info(f"  - Document summaries: {len(document_summaries)}")

        # Build integrated summary
        parts = []

        # 1. Overview (if exists)
        if overview:
            parts.append(f"# 会議概要\n\n{overview}")

        # 2. Meeting summary (new section)
        if meeting_summary:
            parts.append(f"\n\n# 議事要約\n\n{meeting_summary}")

        # 3. Related materials summary (excluding agenda/minutes - they're in meeting summary)
        other_docs = [
            doc for doc in document_summaries if doc.category not in ["agenda", "minutes"]
        ]
        if other_docs:
            parts.append("\n\n# 関連資料の要約\n")
            for i, doc_summary in enumerate(other_docs, 1):
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
        logger.info(
            f"  - Agenda/minutes docs excluded: {len(document_summaries) - len(other_docs)}"
        )
        logger.info(f"  - Other docs included: {len(other_docs)}")

        return {
            "summary_length": len(integrated_summary),
            "document_count": len(document_summaries),
            "other_docs_count": len(other_docs),
        }

    def _execute_finalize(self, step: ActionStep, state: ExecutionState) -> dict:
        """
        Execute finalization step.

        Handles human review (if not batch mode) and character limit checks.
        """
        batch = step.params.get("batch", False)
        final_summary = state.get("final_summary") or ""

        logger.info("Finalizing summary:")
        logger.info(f"  - Batch mode: {batch}")
        logger.info(f"  - Summary length: {len(final_summary)} characters")

        if not final_summary:
            logger.warning("No final summary available for finalization")
            # Use empty summary as fallback
            final_summary = "(要約なし)"

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
