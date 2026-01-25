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

        Steps:
        1. Call HTMLProcessor sub-agent
        2. Discover related documents
        3. Generate ActionPlan for document summarization
        """
        input_url = state["input_url"]
        overview_only = state.get("overview_only", False)

        logger.info("HTML meeting page processing plan:")
        logger.info("  1. Extract main content from HTML")
        logger.info("  2. Discover related documents")
        if overview_only:
            logger.info("  3. Overview only mode - skip document summarization")
        else:
            logger.info("  3. Summarize related documents")

        # Step 1: Extract HTML main content
        try:
            html_result = self.html_processor.invoke({"url": input_url})

            main_content = html_result.get("main_content")
            discovered_documents = html_result.get("discovered_documents", [])

            logger.info(f"Discovered {len(discovered_documents)} related documents")

            if not main_content:
                logger.warning("Failed to extract main content from HTML")
                # Create minimal plan with error
                return {
                    "overview": None,
                    "discovered_documents": [],
                    "action_plan": ActionPlan(
                        steps=[],
                        reasoning="Failed to extract main content from HTML meeting page",
                    ),
                }

            # Generate overview from main content
            overview = self._generate_overview_from_html(main_content, input_url)

            # Build action plan
            steps = []

            if overview_only or not discovered_documents:
                # Overview only - no document summarization
                # But still need integrate step to set final_summary
                steps.append(
                    ActionStep(
                        action_type="integrate_summaries",
                        target=input_url,
                        params={
                            "overview": overview,
                            "document_count": 0,
                        },
                        priority=0,
                        estimated_tokens=1000,
                    )
                )
                reasoning = (
                    "Overview only mode - skipping related document summarization"
                    if overview_only
                    else "No related documents found - overview only"
                )
            else:
                # Add document summarization steps
                # Priority: lower number = higher priority
                for i, doc_url in enumerate(discovered_documents):
                    steps.append(
                        ActionStep(
                            action_type="summarize_pdf",
                            target=doc_url,
                            params={"source_meeting_url": input_url},
                            priority=i,  # Process in discovery order
                            estimated_tokens=5000,  # Estimated with sub-agent
                        )
                    )

                # Add integration step
                steps.append(
                    ActionStep(
                        action_type="integrate_summaries",
                        target=input_url,
                        params={
                            "overview": overview,
                            "document_count": len(discovered_documents),
                        },
                        priority=len(discovered_documents),  # After all summaries
                        estimated_tokens=2000,
                    )
                )

                reasoning = (
                    f"HTML meeting page with {len(discovered_documents)} related documents. "
                    f"Plan: Extract main content → Summarize {len(discovered_documents)} documents → "
                    f"Integrate summaries."
                )

            # Add finalization step
            steps.append(
                ActionStep(
                    action_type="finalize",
                    target=input_url,
                    params={"batch": state.get("batch", False)},
                    priority=len(steps),
                    estimated_tokens=500,
                )
            )

            # Add Bluesky posting step if not skipped
            if not state.get("skip_bluesky_posting", False):
                steps.append(
                    ActionStep(
                        action_type="post_to_bluesky",
                        target=input_url,
                        params={},
                        priority=len(steps),
                        estimated_tokens=100,
                    )
                )

            action_plan = ActionPlan(
                steps=steps,
                reasoning=reasoning,
                total_estimated_tokens=sum(s.estimated_tokens or 0 for s in steps),
            )

            return {
                "overview": overview,
                "discovered_documents": discovered_documents,
                "action_plan": action_plan,
            }

        except Exception as e:
            logger.error(f"Error during HTML meeting planning: {e}")
            import traceback

            traceback.print_exc()

            return {
                "overview": None,
                "discovered_documents": [],
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
                params={
                    "overview": None,  # No overview for single PDF
                    "document_count": 1,
                },
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
                "Single PDF file processing. "
                "Plan: Detect type → Summarize → Integrate → Finalize."
            ),
            total_estimated_tokens=sum(s.estimated_tokens or 0 for s in steps),
        )

        return {
            "overview": None,  # No overview for single PDF
            "discovered_documents": [],
            "action_plan": action_plan,
        }

    def _generate_overview_from_html(self, main_content: str, url: str) -> str:
        """
        Generate overview from HTML main content.

        Args:
            main_content: Extracted main content (markdown)
            url: Source URL

        Returns:
            Generated overview text
        """
        from langchain.prompts import PromptTemplate

        llm = self.model.llm()

        logger.info("Generating overview from HTML main content...")

        overview_prompt = PromptTemplate(
            input_variables=["content", "url"],
            template="""あなたは会議情報を要約する専門家です。以下の会議ページの内容から概要を作成してください。

# 会議ページURL
{url}

# メインコンテンツ
{content}

# 要約作成手順

ステップ1: 会議の基本情報を特定する
- 会議名・委員会名
- 開催日時・場所
- 議題・テーマ

ステップ2: 主要な内容を抽出する
- 議論された主要論点
- 決定事項・合意事項
- 今後の予定・方針

ステップ3: 概要を作成する
- 会議の目的と位置づけ
- 主要な議論内容
- 重要な決定や方針
- 配付資料の概要（リストがある場合）

# 出力形式
概要文のみを出力してください（Markdown不要、改行は適宜使用）

# 文量
500-1500文字程度

# 制約
- 推測や補完は行わない
- メインコンテンツに記載されている内容のみを使用
- 会議の性格（定例会議、臨時会議、審議会等）を明記
""",
        )

        chain = overview_prompt | llm
        result = chain.invoke({"content": main_content[:10000], "url": url})  # Limit to 10K chars

        overview = result.content.strip()

        logger.info(f"Generated overview ({len(overview)} characters)")

        return overview
