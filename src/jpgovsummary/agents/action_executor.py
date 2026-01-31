"""
Action Executor for Plan-Action architecture.

This module executes ActionPlan steps by invoking appropriate sub-agents
and collecting results in ExecutionState.

Supports both sequential and parallel execution modes.
"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

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

    def __init__(self, model: Model | None = None, parallel: bool = True, max_workers: int = 3):
        """
        Initialize ActionExecutor.

        Args:
            model: Model instance for LLM access. If None, uses default Model().
            parallel: Enable parallel execution for same-priority steps (default: True).
            max_workers: Maximum number of parallel workers (default: 3).
        """
        self.model = model if model is not None else Model()
        self.parallel = parallel
        self.max_workers = max_workers

        # Initialize sub-agents
        self.document_type_detector = DocumentTypeDetector(model=self.model)
        self.powerpoint_summarizer = PowerPointSummarizer(model=self.model)
        self.word_summarizer = WordSummarizer(model=self.model)

        # Japanese descriptions for action types
        self._action_type_ja = {
            "summarize_pdf": "PDF要約",
            "generate_initial_overview": "概要生成",
            "integrate_summaries": "要約統合",
            "finalize": "最終化",
            "post_to_bluesky": "Bluesky投稿",
        }

        # Japanese category names
        self._category_ja = {
            "agenda": "議事次第",
            "minutes": "議事録",
            "executive_summary": "とりまとめ",
            "material": "資料",
            "reference": "参考資料",
            "announcement": "お知らせ",
        }

    def _get_step_description_ja(self, step: ActionStep) -> str:
        """
        Get Japanese description for an action step.

        Args:
            step: ActionStep to describe

        Returns:
            Japanese description string
        """
        action_ja = self._action_type_ja.get(step.action_type, step.action_type)

        if step.action_type == "summarize_pdf":
            doc_name = step.params.get("doc_name")
            category = step.params.get("category", "")
            category_ja = self._category_ja.get(category, "")

            if doc_name:
                display_name = doc_name[:30] + "..." if len(doc_name) > 30 else doc_name
            else:
                display_name = step.target.split("/")[-1] if "/" in step.target else step.target

            if category_ja:
                return f"{action_ja}: {display_name} ({category_ja})"
            else:
                return f"{action_ja}: {display_name}"
        else:
            return action_ja

    def execute_plan(self, state: ExecutionState) -> ExecutionState:
        """
        Execute all steps in the action plan.

        Args:
            state: ExecutionState with plan to execute

        Returns:
            Updated ExecutionState with results
        """
        plan = state["plan"]

        logger.info(f"Executing action plan with {len(plan.steps)} steps")
        logger.info(f"Plan reasoning: {plan.reasoning}")
        logger.info(f"Parallel mode: {self.parallel}")

        if self.parallel:
            return self._execute_plan_parallel(state)
        else:
            return self._execute_plan_sequential(state)

    def _execute_plan_sequential(self, state: ExecutionState) -> ExecutionState:
        """
        Execute all steps sequentially.

        Args:
            state: ExecutionState with plan to execute

        Returns:
            Updated ExecutionState with results
        """
        plan = state["plan"]
        current_index = state.get("current_step_index", 0)

        # Execute each step sequentially
        for i, step in enumerate(plan.steps):
            if i < current_index:
                # Already executed
                continue

            step_desc = self._get_step_description_ja(step)
            logger.info("")
            logger.info(f"[{i + 1}/{len(plan.steps)}] {step_desc}")

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

                logger.info("  ✅ 完了")

            except Exception as e:
                logger.error(f"  ❌ 失敗: {e}")
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

        logger.info("")
        logger.info("=" * 60)
        logger.info("📋 実行完了")
        logger.info("=" * 60)
        success_count = sum(1 for a in state["completed_actions"] if a.success)
        logger.info(f"  成功: {success_count}/{len(plan.steps)}件")
        if state["errors"]:
            logger.info(f"  失敗: {len(state['errors'])}件")
        logger.info("=" * 60)

        return state

    def _execute_plan_parallel(self, state: ExecutionState) -> ExecutionState:
        """
        Execute steps with parallel processing for same-priority steps.

        Groups steps by priority and executes same-priority steps in parallel.
        Only parallelizes 'summarize_pdf' action types for safety.

        Args:
            state: ExecutionState with plan to execute

        Returns:
            Updated ExecutionState with results
        """
        plan = state["plan"]

        # Group steps by priority
        priority_groups: dict[int, list[tuple[int, ActionStep]]] = defaultdict(list)
        for i, step in enumerate(plan.steps):
            priority_groups[step.priority].append((i, step))

        logger.info(f"Grouped into {len(priority_groups)} priority levels")

        step_count = 0
        total_steps = len(plan.steps)

        # Execute by priority order (lower number = higher priority)
        for priority in sorted(priority_groups.keys()):
            steps_with_indices = priority_groups[priority]

            # Separate parallelizable steps (summarize_pdf) from sequential steps
            parallel_steps = [
                (i, s) for i, s in steps_with_indices if s.action_type == "summarize_pdf"
            ]
            sequential_steps = [
                (i, s) for i, s in steps_with_indices if s.action_type != "summarize_pdf"
            ]

            # Execute parallel steps
            if len(parallel_steps) > 1:
                # Log each parallel step with Japanese description
                logger.info("")
                logger.info(f"並列実行開始: {len(parallel_steps)}件のPDF要約")
                for idx, step in parallel_steps:
                    step_desc = self._get_step_description_ja(step)
                    logger.info(f"  [{idx + 1}/{total_steps}] {step_desc}")

                results = self._execute_steps_parallel(parallel_steps, state)

                for (idx, step), result in zip(parallel_steps, results, strict=True):
                    step_count += 1
                    step_desc = self._get_step_description_ja(step)
                    if result.get("success", False):
                        logger.info(f"  ✅ [{idx + 1}] {step_desc} 完了")
                    else:
                        logger.error(
                            f"  ❌ [{idx + 1}] {step_desc} 失敗: {result.get('error', 'Unknown')}"
                        )

            elif len(parallel_steps) == 1:
                # Single step - run sequentially
                sequential_steps.extend(parallel_steps)

            # Execute sequential steps
            for idx, step in sequential_steps:
                step_count += 1
                step_desc = self._get_step_description_ja(step)
                logger.info("")
                logger.info(f"[{idx + 1}/{total_steps}] {step_desc}")

                try:
                    result = self._execute_step(step, state)

                    completed_action = CompletedAction(
                        step=step,
                        result=result,
                        tokens_used=result.get("tokens_used") if isinstance(result, dict) else None,
                        success=True,
                    )
                    state["completed_actions"].append(completed_action)
                    state["current_step_index"] = idx + 1

                    logger.info("  ✅ 完了")

                except Exception as e:
                    logger.error(f"  ❌ 失敗: {e}")
                    import traceback

                    traceback.print_exc()

                    completed_action = CompletedAction(
                        step=step,
                        result=None,
                        success=False,
                        error_message=str(e),
                    )
                    state["completed_actions"].append(completed_action)
                    state["errors"].append(f"Step {idx + 1} ({step.action_type}): {str(e)}")
                    state["current_step_index"] = idx + 1

        logger.info("")
        logger.info("=" * 60)
        logger.info("📋 実行完了")
        logger.info("=" * 60)
        success_count = sum(1 for a in state["completed_actions"] if a.success)
        logger.info(f"  成功: {success_count}/{total_steps}件")
        if state["errors"]:
            logger.info(f"  失敗: {len(state['errors'])}件")
        logger.info("=" * 60)

        return state

    def _execute_steps_parallel(
        self, steps_with_indices: list[tuple[int, ActionStep]], state: ExecutionState
    ) -> list[dict]:
        """
        Execute multiple steps in parallel using ThreadPoolExecutor.

        Args:
            steps_with_indices: List of (index, step) tuples to execute
            state: ExecutionState (shared, but only append to document_summaries)

        Returns:
            List of result dicts
        """

        def execute_single(idx_step: tuple[int, ActionStep]) -> dict:
            idx, step = idx_step
            try:
                # Only execute summarize_pdf in parallel
                if step.action_type == "summarize_pdf":
                    result = self._execute_summarize_pdf_isolated(step)
                    return {"success": True, "idx": idx, "step": step, "result": result}
                else:
                    return {
                        "success": False,
                        "idx": idx,
                        "step": step,
                        "error": "Not parallelizable",
                    }
            except Exception as e:
                return {"success": False, "idx": idx, "step": step, "error": str(e)}

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = list(executor.map(execute_single, steps_with_indices))
            results = futures

        # Process results and update state
        for r in results:
            step = r["step"]
            idx = r["idx"]

            if r["success"]:
                result = r["result"]
                doc_summary = result.get("doc_summary")
                if doc_summary:
                    state["document_summaries"].append(doc_summary)

                completed_action = CompletedAction(
                    step=step,
                    result=result,
                    tokens_used=result.get("tokens_used"),
                    success=True,
                )
                state["completed_actions"].append(completed_action)
            else:
                completed_action = CompletedAction(
                    step=step,
                    result=None,
                    success=False,
                    error_message=r.get("error"),
                )
                state["completed_actions"].append(completed_action)
                state["errors"].append(f"Step {idx + 1} ({step.action_type}): {r.get('error')}")

        return results

    def _execute_summarize_pdf_isolated(self, step: ActionStep) -> dict:
        """
        Execute PDF summarization in isolated context (for parallel execution).

        Unlike _execute_summarize_pdf, this returns the DocumentSummaryResult
        instead of modifying state directly.

        For agenda category documents, skips LLM and uses raw PDF text directly.

        Args:
            step: ActionStep to execute

        Returns:
            Result dict with doc_summary included
        """
        url = step.target
        category = step.params.get("category")

        logger.info(f"[Parallel] Loading PDF: {url.split('/')[-1]}")
        pdf_pages = load_pdf_as_text(url)
        logger.info(f"[Parallel] Loaded {len(pdf_pages)} pages from {url.split('/')[-1]}")

        # For agenda documents, skip LLM and use raw PDF text
        if category == "agenda":
            logger.info("[Parallel] Agenda document - using raw PDF text (no LLM)")
            summary = "\n\n".join(pdf_pages)
            title = url.split("/")[-1].replace(".pdf", "")
            document_type = "Agenda"
        else:
            # Detect document type
            detection_result = self.document_type_detector.invoke(
                {"pdf_pages": pdf_pages[:10], "url": url}
            )

            document_type = detection_result["document_type"]
            logger.info(f"[Parallel] Detected type: {document_type} for {url.split('/')[-1]}")

            # Select appropriate summarizer
            if document_type == "PowerPoint":
                summarizer_result = self.powerpoint_summarizer.invoke(
                    {"pdf_pages": pdf_pages, "url": url}
                )
            else:
                summarizer_result = self.word_summarizer.invoke(
                    {"pdf_pages": pdf_pages, "url": url}
                )

            summary = summarizer_result.get("summary", "")
            title = summarizer_result.get("title", url.split("/")[-1])

        doc_summary = DocumentSummaryResult(
            url=url,
            name=title,
            summary=summary,
            document_type=document_type,
            category=category,
        )

        logger.info(f"[Parallel] Completed: {title} ({len(summary)} chars)")

        return {
            "document_type": document_type,
            "summary_length": len(summary),
            "title": title,
            "category": category,
            "doc_summary": doc_summary,
        }

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
        elif step.action_type == "generate_initial_overview":
            return self._execute_generate_initial_overview(step, state)
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

        For agenda category documents, skips LLM and uses raw PDF text directly.

        Steps (for non-agenda):
        1. Load PDF
        2. Detect document type
        3. Invoke appropriate summarizer (PowerPoint or Word)
        4. Store result in state
        """
        url = step.target
        category = step.params.get("category")

        logger.info(f"Loading PDF from: {url}")
        pdf_pages = load_pdf_as_text(url)
        logger.info(f"Loaded {len(pdf_pages)} pages")

        # For agenda documents, skip LLM and use raw PDF text
        if category == "agenda":
            logger.info("Agenda document - using raw PDF text (no LLM)")
            summary = "\n\n".join(pdf_pages)
            title = url.split("/")[-1].replace(".pdf", "")
            document_type = "Agenda"
        else:
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
                logger.warning(
                    f"Unsupported type '{document_type}', falling back to WordSummarizer"
                )
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

        # Use category from Phase 1's discovered_documents (passed via ActionStep params)
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

    def _execute_generate_initial_overview(self, step: ActionStep, state: ExecutionState) -> dict:
        """
        Execute initial overview generation (Phase 2 Step 1).

        Creates overview from:
        1. Main content text (from HTML)
        2. Embedded agenda/minutes content (from HTML)
        3. Embedded minutes content (from HTML)
        4. Already-processed agenda/minutes PDF summaries (using Phase 1 category classifications)

        Args:
            step: ActionStep with params containing main_content, embedded_agenda, embedded_minutes
            state: ExecutionState with document_summaries from agenda/minutes PDFs

        Returns:
            Result dict with overview_length
        """
        main_content = step.params.get("main_content") or state.get("main_content", "")
        embedded_agenda = step.params.get("embedded_agenda") or state.get("embedded_agenda")
        embedded_minutes = step.params.get("embedded_minutes") or state.get("embedded_minutes")
        input_url = step.target

        # Get agenda/minutes summaries from already-processed PDFs
        # NOTE: Category filtering uses Phase 1 classifications (from HTMLProcessor's discovered_documents).
        # The category is passed through ActionStep params and stored in DocumentSummaryResult.
        document_summaries = state.get("document_summaries", [])
        meeting_docs = [doc for doc in document_summaries if doc.category in ["agenda", "minutes"]]

        logger.info("Generating initial overview:")
        logger.info(f"  - Main content: {len(main_content)} chars")
        logger.info(f"  - Embedded agenda: {'Yes' if embedded_agenda else 'No'}")
        logger.info(f"  - Embedded minutes: {'Yes' if embedded_minutes else 'No'}")
        logger.info(f"  - Agenda/minutes PDFs: {len(meeting_docs)}")

        # Build context for overview generation
        context_parts = []

        # 1. Main content (meeting page text)
        if main_content:
            context_parts.append(f"# 会議ページ本文\n\n{main_content[:8000]}")

        # 2. Embedded agenda from HTML
        if embedded_agenda:
            context_parts.append(f"\n\n# 議事次第（HTML内）\n\n{embedded_agenda}")

        # 3. Embedded minutes from HTML
        if embedded_minutes:
            context_parts.append(f"\n\n# 議事録（HTML内）\n\n{embedded_minutes}")

        # 4. Agenda/minutes PDF summaries
        for doc in meeting_docs:
            label = "議事次第" if doc.category == "agenda" else "議事録"
            context_parts.append(f"\n\n# {label}（PDF: {doc.name}）\n\n{doc.summary}")

        combined_context = "\n".join(context_parts)

        if not combined_context.strip():
            logger.warning("No content available for overview generation")
            state["initial_overview"] = "(内容なし)"
            return {"overview_length": 0}

        # Generate overview using LLM
        llm = self.model.llm()

        from langchain.prompts import PromptTemplate

        overview_prompt = PromptTemplate(
            input_variables=["content", "url"],
            template="""あなたは会議情報を要約する専門家です。以下の会議情報から概要を作成してください。

# 会議ページURL
{url}

# 会議情報
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

# 出力形式
概要文のみを出力してください（Markdown見出し不要、改行は適宜使用）

# 文量
500-1500文字程度

# 制約
- 推測や補完は行わない
- 提供された情報に記載されている内容のみを使用
- 会議の性格（定例会議、臨時会議、審議会等）を明記
""",
        )

        chain = overview_prompt | llm

        try:
            result = chain.invoke({"content": combined_context[:15000], "url": input_url})
            overview = result.content.strip()

            logger.info(f"Generated initial overview: {len(overview)} characters")

            # Store in state
            state["initial_overview"] = overview

            return {"overview_length": len(overview)}

        except Exception as e:
            logger.error(f"Error generating overview: {e}")
            import traceback

            traceback.print_exc()
            state["initial_overview"] = "(概要生成エラー)"
            return {"overview_length": 0, "error": str(e)}

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

        Combines initial_overview + document summaries into final summary.
        """
        # Use initial_overview from state (generated in Phase 2 Step 1)
        overview = state.get("initial_overview")
        document_summaries = state.get("document_summaries", [])

        logger.info("Integrating summaries:")
        logger.info(f"  - Initial overview: {'Yes' if overview else 'No'}")
        logger.info(f"  - Document summaries: {len(document_summaries)}")

        # Build integrated summary
        parts = []

        # 1. Overview (from Phase 2 Step 1)
        if overview:
            parts.append(overview)

        # 2. Document summaries (all processed documents)
        if document_summaries:
            parts.append("\n\n---\n\n## 関連資料")
            for doc_summary in document_summaries:
                parts.append(f"\n\n### {doc_summary.name}")
                if doc_summary.document_type:
                    parts.append(f"\n（{doc_summary.document_type}）")
                parts.append(f"\n\n{doc_summary.summary}")

        if parts:
            integrated_summary = "\n".join(parts)
        else:
            integrated_summary = "(要約なし)"

        state["final_summary"] = integrated_summary

        logger.info(f"Integrated summary: {len(integrated_summary)} characters")
        logger.info(f"  - Documents included: {len(document_summaries)}")

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
