"""
Action Planner for Plan-Action architecture.

This module contains the planning agent that analyzes input (HTML meeting page
or PDF file) and generates an ActionPlan with prioritized execution steps.
"""

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .. import Model, logger
from ..state import ActionPlan, ActionStep, PlanState, ScoredDocument
from ..subagents import HTMLProcessor


class ActionPlanner:
    """
    Action planning agent.

    Analyzes input and generates execution plan with prioritized steps.
    """

    # Categories eligible for scoring (others are excluded)
    SCORABLE_CATEGORIES = ["material", "executive_summary", "announcement"]

    # Categories excluded from scoring
    EXCLUDED_CATEGORIES = ["personal_material", "participants", "seating", "reference", "other"]

    # Name patterns that indicate low-priority documents (excluded from scoring)
    EXCLUDED_NAME_PATTERNS = [
        "振り返り",
        "振返り",
        "前回",
        "これまで",
        "パブリックコメント",
        "パブコメ",
        "意見募集",
        "意見公募",
        "参考配布",
    ]

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

            # Filter documents by category
            meeting_related_docs = [
                doc for doc in discovered_documents if doc.category in ["agenda", "minutes"]
            ]

            # Filter documents for scoring
            scorable_docs, excluded_docs = self._filter_documents_for_scoring(
                discovered_documents
            )

            logger.info(f"Discovered {len(discovered_documents)} related documents")
            logger.info(f"  - Meeting-related: {len(meeting_related_docs)} (agenda/minutes)")
            logger.info(f"  - Scorable: {len(scorable_docs)}")
            logger.info(f"  - Excluded: {len(excluded_docs)}")

            if not main_content:
                logger.warning("Failed to extract main content from HTML")
                return {
                    "main_content": None,
                    "structured_overview": None,
                    "discovered_documents": [],
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
                        action_type="generate_initial_summary",
                        target=input_url,
                        params={
                            "main_content": main_content,
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
                        action_type="generate_initial_summary",
                        target=input_url,
                        params={
                            "main_content": main_content,
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
                                "doc_name": doc.name,  # Store document name for display
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
                        action_type="generate_initial_summary",
                        target=input_url,
                        params={
                            "main_content": main_content,
                        },
                        priority=priority,
                        estimated_tokens=2000,
                    )
                )
                priority += 1

                # Step 3: Score and select documents in Phase 1
                # (No longer deferred to Phase 2)
                scored_docs: list[ScoredDocument] = []
                selected_docs: list[ScoredDocument] = []
                if scorable_docs:
                    logger.info("Phase 1でスコアリングを実行...")
                    scored_docs = self._score_documents(
                        scorable_docs, excluded_docs, main_content
                    )

                    # Select top 5 documents with score >= 3 (on 5-point scale)
                    selected_docs = [d for d in scored_docs if d.score >= 3][:5]

                    # Step 4: Create summarize_pdf steps for each selected document
                    # These are processed in parallel (same priority)
                    if selected_docs:
                        for doc in selected_docs:
                            steps.append(
                                ActionStep(
                                    action_type="summarize_pdf",
                                    target=doc.url,
                                    params={
                                        "source_meeting_url": input_url,
                                        "category": doc.category,
                                        "doc_name": doc.name,  # Store document name for display
                                        "score": doc.score,
                                    },
                                    priority=priority,  # Same priority for parallel execution
                                    estimated_tokens=5000,
                                )
                            )
                        priority += 1

                reasoning = (
                    f"HTML meeting with {len(discovered_documents)} documents. "
                    f"Flow: Process agenda/minutes PDFs → Generate initial overview → "
                    f"Summarize {len(selected_docs)} selected docs → Integrate."
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

            # Output action plan in Japanese
            self._log_action_plan_japanese(action_plan)

            return {
                "main_content": main_content,
                "structured_overview": None,  # Generated in Phase 2, not Phase 1
                "discovered_documents": discovered_documents,
                "action_plan": action_plan,
            }

        except Exception as e:
            logger.error(f"Error during HTML meeting planning: {e}")
            import traceback

            traceback.print_exc()

            return {
                "main_content": None,
                "structured_overview": None,
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
        ]

        # Add Bluesky posting step if not skipped
        if not state.get("skip_bluesky_posting", False):
            steps.append(
                ActionStep(
                    action_type="post_to_bluesky",
                    target=input_url,
                    params={},
                    priority=2,
                    estimated_tokens=100,
                )
            )

        action_plan = ActionPlan(
            steps=steps,
            reasoning=(
                "Single PDF file processing. Plan: Detect type → Summarize → Integrate."
            ),
            total_estimated_tokens=sum(s.estimated_tokens or 0 for s in steps),
        )

        # Output action plan in Japanese
        self._log_action_plan_japanese(action_plan)

        return {
            "main_content": None,  # No main content for single PDF
            "structured_overview": None,  # No structured overview for PDF files
            "discovered_documents": [],
            "embedded_agenda": None,
            "embedded_minutes": None,
            "action_plan": action_plan,
        }

    def _filter_documents_for_scoring(
        self, documents: list
    ) -> tuple[list, list]:
        """
        Filter documents into scorable and excluded lists.

        Args:
            documents: All discovered documents

        Returns:
            Tuple of (scorable_docs, excluded_docs) with exclusion reason
        """
        scorable = []
        excluded = []

        for doc in documents:
            # Skip meeting-related docs (processed separately)
            if doc.category in ["agenda", "minutes"]:
                continue

            # Check category exclusion
            if doc.category not in self.SCORABLE_CATEGORIES:
                excluded.append((doc, f"カテゴリ除外: {doc.category}"))
                continue

            # Check name pattern exclusion
            excluded_by_pattern = False
            for pattern in self.EXCLUDED_NAME_PATTERNS:
                if pattern in doc.name:
                    excluded.append((doc, f"パターン除外: {pattern}"))
                    excluded_by_pattern = True
                    break

            if not excluded_by_pattern:
                scorable.append(doc)

        return scorable, excluded

    def _score_documents(
        self, documents: list, excluded_docs: list, main_content: str
    ) -> list[ScoredDocument]:
        """
        Score documents in Phase 1 using main content as context.

        Uses 5-point scale:
        - 5: 会議の主要議題に直接関連
        - 4: 政策・方針に関する資料
        - 3: データ・統計資料
        - 2: 背景資料
        - 1: 重要度低

        Args:
            documents: List of DiscoveredDocument to score
            excluded_docs: List of (doc, reason) tuples for excluded documents
            main_content: Main content from HTML for context

        Returns:
            List of ScoredDocument sorted by score (descending)
        """
        # Log all documents including excluded ones
        logger.info("")
        logger.info("=" * 60)
        logger.info("📊 資料スコアリング結果")
        logger.info("=" * 60)

        # Log excluded documents first
        if excluded_docs:
            logger.info("")
            logger.info("除外資料:")
            for doc, reason in excluded_docs:
                logger.info(f"  [除外] {doc.name} ({reason})")

        if not documents:
            logger.info("")
            logger.info("スコアリング対象資料: なし")
            logger.info("=" * 60)
            return []

        logger.info("")
        logger.info(f"スコアリング対象: {len(documents)}件")

        llm = self.model.llm()

        score_prompt = PromptTemplate(
            input_variables=["content", "documents"],
            template="""あなたは会議資料の重要度を判定する専門家です。

# 会議ページの内容
{content}

# 資料リスト
{documents}

# 判定基準（5点満点）
以下の基準でスコアを付けてください：
- 5点: 会議の主要議題に直接関連する資料
- 4点: 政策・方針に関する資料
- 3点: データ・統計資料
- 2点: 背景資料・補足資料
- 1点: 重要度低

# 出力形式
JSON配列で出力してください：
[
  {{"url": "資料URL", "name": "資料名", "score": スコア, "reason": "理由"}},
  ...
]

# 注意
- 全ての資料にスコアを付けてください
- スコアは1-5の整数で
""",
        )

        # Format documents for prompt
        doc_list = "\n".join(
            [f"- [{doc.category}] {doc.name}: {doc.url}" for doc in documents]
        )

        try:
            parser = JsonOutputParser()
            chain = score_prompt | llm | parser

            result = chain.invoke(
                {
                    "content": main_content[:8000] if main_content else "(内容なし)",
                    "documents": doc_list,
                }
            )

            # Convert to ScoredDocument list
            # Only accept results that match documents in the input list
            scored_documents = []
            for item in result:
                # Find original document by URL match
                original_doc = next(
                    (d for d in documents if d.url == item.get("url")), None
                )

                # Skip if URL doesn't match any input document (LLM hallucination)
                if original_doc is None:
                    continue

                scored_documents.append(
                    ScoredDocument(
                        url=original_doc.url,
                        name=original_doc.name,  # Use original name, not LLM's
                        category=original_doc.category,
                        score=float(item.get("score", 0)),
                        reason=item.get("reason", ""),
                    )
                )

            # Sort by score descending
            scored_documents.sort(key=lambda x: x.score, reverse=True)

            # Log all scored documents
            logger.info("")
            logger.info("スコア結果:")
            for doc in scored_documents:
                selected_mark = "→選択" if doc.score >= 3 else ""
                logger.info(f"  [{doc.score:.0f}点] {doc.name} {selected_mark}")

            logger.info("")
            selected_count = len([d for d in scored_documents if d.score >= 3])
            logger.info(f"選択: {selected_count}件 (3点以上)")
            logger.info("=" * 60)

            return scored_documents

        except Exception as e:
            logger.error(f"スコアリング中にエラー: {e}")
            import traceback

            traceback.print_exc()
            return []

    def _log_action_plan_japanese(self, action_plan: ActionPlan) -> None:
        """
        Output action plan as Japanese bullet list.

        Args:
            action_plan: ActionPlan to log
        """
        # Action type to Japanese mapping
        action_type_ja = {
            "summarize_pdf": "PDF要約",
            "generate_initial_summary": "概要生成",
            "integrate_summaries": "要約統合",
            "post_to_bluesky": "Bluesky投稿",
        }

        # Category to Japanese mapping
        category_ja_map = {
            "agenda": "議事次第",
            "minutes": "議事録",
            "executive_summary": "とりまとめ",
            "material": "資料",
            "reference": "参考資料",
        }

        logger.info("")
        logger.info("=" * 60)
        logger.info("📋 アクションプラン")
        logger.info("=" * 60)

        for i, step in enumerate(action_plan.steps, 1):
            action_ja = action_type_ja.get(step.action_type, step.action_type)

            # Build step description
            if step.action_type == "summarize_pdf":
                # Use doc_name from params if available, otherwise fallback to file name
                doc_name = step.params.get("doc_name")
                category = step.params.get("category", "")
                category_ja = category_ja_map.get(category, "")

                if doc_name:
                    # Abbreviate long document names (max 10 chars)
                    display_name = doc_name[:7] + "..." if len(doc_name) > 10 else doc_name
                else:
                    # Fallback to file name
                    display_name = (
                        step.target.split("/")[-1] if "/" in step.target else step.target
                    )

                if category_ja:
                    desc = f"{action_ja}: {display_name} ({category_ja})"
                else:
                    desc = f"{action_ja}: {display_name}"
            else:
                desc = action_ja

            logger.info(f"  {i}. {desc}")

        logger.info("")
        logger.info(f"合計ステップ数: {len(action_plan.steps)}")
        logger.info("=" * 60)
