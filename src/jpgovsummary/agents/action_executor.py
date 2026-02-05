"""
Action Executor for Plan-Action architecture.

This module executes ActionPlan steps by invoking appropriate sub-agents
and collecting results in ExecutionState.

Supports both sequential and parallel execution modes.
"""

import json
import os
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from .. import Model, logger
from ..state import (
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
            "generate_initial_summary": "概要生成",
            "integrate_summaries": "要約統合",
            "post_to_bluesky": "Bluesky投稿",
        }

        # Emojis for action types
        self._action_emoji = {
            "summarize_pdf": "📄",
            "generate_initial_summary": "📝",
            "integrate_summaries": "🔗",
            "post_to_bluesky": "🦋",
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
        Get Japanese description for an action step with emoji.

        Args:
            step: ActionStep to describe

        Returns:
            Japanese description string with emoji prefix
        """
        action_ja = self._action_type_ja.get(step.action_type, step.action_type)
        emoji = self._action_emoji.get(step.action_type, "▶️")

        if step.action_type == "summarize_pdf":
            doc_name = step.params.get("doc_name")
            category = step.params.get("category", "")
            category_ja = self._category_ja.get(category, "")

            if doc_name:
                display_name = doc_name[:7] + "..." if len(doc_name) > 10 else doc_name
            else:
                display_name = step.target.split("/")[-1] if "/" in step.target else step.target

            if category_ja:
                return f"{emoji} {action_ja}: {display_name} ({category_ja})"
            else:
                return f"{emoji} {action_ja}: {display_name}"
        else:
            return f"{emoji} {action_ja}"

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

        # Determine log prefix early: prefer doc_name, fallback to category_ja
        doc_name = step.params.get("doc_name", "") if step.params else ""
        category_ja = self._category_ja.get(category, "")
        if doc_name:
            log_prefix = doc_name[:7] + "..." if len(doc_name) > 10 else doc_name
        elif category_ja:
            log_prefix = category_ja
        else:
            log_prefix = "文書"

        # For agenda documents, skip LLM and use raw PDF text
        if category == "agenda":
            pdf_pages = load_pdf_as_text(url)
            summary = "\n\n".join(pdf_pages)
            title = url.split("/")[-1].replace(".pdf", "")
            document_type = "Agenda"
        else:
            pdf_pages = load_pdf_as_text(url)
            # Detect document type
            detection_result = self.document_type_detector.invoke(
                {"pdf_pages": pdf_pages[:10], "url": url, "display_name": log_prefix}
            )

            document_type = detection_result["document_type"]

            # Select appropriate summarizer (normalize to lowercase)
            doc_type_lower = document_type.lower()
            if doc_type_lower == "powerpoint":
                summarizer_result = self.powerpoint_summarizer.invoke(
                    {"pdf_pages": pdf_pages, "url": url, "display_name": log_prefix}
                )
            elif doc_type_lower == "word":
                summarizer_result = self.word_summarizer.invoke(
                    {"pdf_pages": pdf_pages, "url": url, "display_name": log_prefix}
                )
            else:
                # Fallback: Try Word summarizer for other types
                summarizer_result = self.word_summarizer.invoke(
                    {"pdf_pages": pdf_pages, "url": url, "display_name": log_prefix}
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

        logger.info(f"  [{log_prefix}] 要約完了 ({len(summary)}文字)")

        # Output generated summary
        logger.info(f"  [{log_prefix}] 要約：{summary.replace(chr(10), '\\n')}")

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
        elif step.action_type == "generate_initial_summary":
            return self._execute_generate_initial_summary(step, state)
        elif step.action_type == "integrate_summaries":
            return self._execute_integrate_summaries(step, state)
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

        # Determine log prefix early: prefer doc_name, fallback to category_ja
        doc_name = step.params.get("doc_name", "") if step.params else ""
        category_ja = self._category_ja.get(category, "")
        if doc_name:
            log_prefix = doc_name[:7] + "..." if len(doc_name) > 10 else doc_name
        elif category_ja:
            log_prefix = category_ja
        else:
            log_prefix = "文書"

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
                    "display_name": log_prefix,
                }
            )

            document_type = detection_result["document_type"]
            confidence_scores = detection_result["confidence_scores"]

            logger.info(f"Detected type: {document_type}")
            logger.info(f"Confidence scores: {confidence_scores}")

            # Select appropriate summarizer (normalize to lowercase)
            doc_type_lower = document_type.lower()
            if doc_type_lower == "powerpoint":
                logger.info("Using PowerPointSummarizer sub-agent")
                summarizer_result = self.powerpoint_summarizer.invoke(
                    {
                        "pdf_pages": pdf_pages,
                        "url": url,
                        "display_name": log_prefix,
                    }
                )
            elif doc_type_lower == "word":
                logger.info("Using WordSummarizer sub-agent")
                summarizer_result = self.word_summarizer.invoke(
                    {
                        "pdf_pages": pdf_pages,
                        "url": url,
                        "display_name": log_prefix,
                    }
                )
            else:
                # Fallback: Try Word summarizer for other types
                logger.info(
                    f"Type '{document_type}' detected, using WordSummarizer"
                )
                summarizer_result = self.word_summarizer.invoke(
                    {
                        "pdf_pages": pdf_pages,
                        "url": url,
                        "display_name": log_prefix,
                    }
                )

            # Extract summary
            summary = summarizer_result.get("summary", "")
            title = summarizer_result.get("title", url.split("/")[-1])

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

        logger.info(f"  [{log_prefix}] Generated summary: {len(summary)} characters")

        # Output generated summary
        logger.info(f"  [{log_prefix}] 要約：{summary.replace(chr(10), '\\n')}")

        return {
            "document_type": document_type,
            "summary_length": len(summary),
            "title": title,
            "category": category,
        }

    def _execute_generate_initial_summary(self, step: ActionStep, state: ExecutionState) -> dict:
        """
        Execute initial overview generation (Phase 2 Step 1).

        Generates BOTH:
        1. structured_overview (箇条書き形式 - bullet point format)
        2. initial_summary (文章形式 - prose format)

        From:
        - Main content text (from HTML)
        - Already-processed agenda/minutes PDF summaries

        Args:
            step: ActionStep with params containing main_content
            state: ExecutionState with document_summaries from agenda/minutes PDFs

        Returns:
            Result dict with overview_length
        """
        main_content = step.params.get("main_content") or state.get("main_content", "")
        input_url = step.target

        # Get agenda/minutes summaries from already-processed PDFs
        document_summaries = state.get("document_summaries", [])
        meeting_docs = [doc for doc in document_summaries if doc.category in ["agenda", "minutes"]]

        logger.info("Generating meeting overviews (Phase 2):")
        logger.info(f"  - Main content: {len(main_content)} chars")
        logger.info(f"  - Agenda/minutes PDFs: {len(meeting_docs)}")

        # Build context for overview generation
        context_parts = []

        # 1. Main content (meeting page text)
        if main_content:
            context_parts.append(f"# 会議ページ本文\n\n{main_content[:8000]}")

        # 2. Agenda/minutes PDF summaries
        for doc in meeting_docs:
            label = "議事次第" if doc.category == "agenda" else "議事録"
            context_parts.append(f"\n\n# {label}（PDF: {doc.name}）\n\n{doc.summary}")

        combined_context = "\n".join(context_parts)

        if not combined_context.strip():
            logger.warning("No content available for overview generation")
            state["structured_overview"] = ""
            state["initial_summary"] = ""
            state["has_meeting_info"] = False
            return {"overview_length": 0}

        # Generate both structured_overview and initial_summary using LLM
        llm = self.model.llm()

        from langchain.prompts import PromptTemplate

        overview_prompt = PromptTemplate(
            input_variables=["content", "url"],
            template="""あなたは会議情報から2種類の会議概要を作成する専門家です。

# 会議ページURL
{url}

# 会議情報
{content}

# タスク

以下の2つの概要を生成してください：

## 1. 構造化概要（箇条書き形式）

会議の基本情報を箇条書きで簡潔にまとめてください。

**出力形式例：**
```
- 会議名: [会議の正式名称と回数]
- 開催日時: [日時]
- 開催場所: [場所]
- 議題: [主要な議題を簡潔に]
- 決定事項: [重要な決定事項]
```

**重要な注意事項：**
- 箇条書き形式（-記号使用）
- 各項目は1-2行で簡潔に
- **情報が見つからない項目は記載しない（省略する）**
- 「不明」「記載なし」などの否定的な記述は不要

## 2. 議事要約（文章形式）

会議の内容を500-1500文字程度の文章形式で詳しく説明してください。

**厳守事項：**
- **箇条書き厳格禁止**：箇条書き記号（・、-、*、+、番号など）は絶対に使用しない
- すべて文章形式（段落形式）で記述
- 複数の事項は接続詞や句読点で自然に繋ぐ

**含めるべき内容（ある場合のみ）：**
1. 会議の目的と背景
2. 主要な議論内容と論点
3. 決定事項や合意内容
4. 今後の予定や次回会議
5. その他重要な情報

**重要な注意事項：**
- 提供された情報のみ使用（推測・創作禁止）
- **情報がない項目には一切言及しない**
- 「不明」「記載なし」「確認が必要」などの否定的表現は使用しない
- 書いてある内容だけを自然な文章で記述
- 文章形式で自然に読める内容
- 段落分けは適宜行ってよい
- 会議名と回数（「第X回○○会議」等）は必ず含めること

**除外すべき形式的情報：**
- 会議の進行説明（「議事次第に沿って」「事務局説明を受けた後」「意見交換を行い閉会した」等）
- 配布資料の名称リスト（「〇〇資料、△△資料が配布された」等）※資料の中身の要約は残す
- 資料の公開/非公開情報（「一部非公開扱い」「資料公開中」等）
- 「事務局説明では」「〇〇委員からは」等の発言者導入句

**表現の改善：**
- 「〜が確認された」「〜が報告された」→「〜である」「〜とされた」等、進行報告ではなく内容記述の表現を使用

# 出力形式

以下の形式で出力してください：

---STRUCTURED_OVERVIEW---
[構造化概要（箇条書き）をここに記述]
---INITIAL_SUMMARY---
[議事要約（文章形式）をここに記述]
---END---
""",
        )

        chain = overview_prompt | llm

        try:
            result = chain.invoke({"content": combined_context[:15000], "url": input_url})
            full_output = result.content.strip()

            # Parse the two sections
            import re
            structured_match = re.search(
                r'---STRUCTURED_OVERVIEW---\s*(.+?)\s*---INITIAL_SUMMARY---',
                full_output,
                re.DOTALL
            )
            summary_match = re.search(
                r'---INITIAL_SUMMARY---\s*(.+?)\s*---END---',
                full_output,
                re.DOTALL
            )

            if structured_match and summary_match:
                structured_overview = structured_match.group(1).strip()
                initial_summary = summary_match.group(1).strip()
            else:
                logger.warning("Failed to parse structured output, using fallback")
                structured_overview = ""
                initial_summary = full_output

            logger.info(f"Generated structured overview: {len(structured_overview)} characters")
            logger.info(f"Generated initial summary: {len(initial_summary)} characters")

            # Check if meaningful meeting information was found
            has_meeting_info = (
                len(structured_overview) > 50
                or len(initial_summary) > 200
            )

            # Store in state
            state["structured_overview"] = structured_overview
            state["initial_summary"] = initial_summary
            state["has_meeting_info"] = has_meeting_info

            # Output both overviews
            logger.info("")
            logger.info("=" * 60)
            logger.info("構造化概要（箇条書き）:")
            logger.info("=" * 60)
            logger.info(structured_overview)
            logger.info("")
            logger.info("=" * 60)
            logger.info("議事要約（文章形式）:")
            logger.info("=" * 60)
            logger.info(initial_summary)
            logger.info("=" * 60)

            return {
                "structured_overview_length": len(structured_overview),
                "initial_summary_length": len(initial_summary),
            }

        except Exception as e:
            logger.error(f"Error generating overviews: {e}")
            import traceback

            traceback.print_exc()
            state["structured_overview"] = "（生成エラー）"
            state["initial_summary"] = "（生成エラー）"
            state["has_meeting_info"] = False
            return {"overview_length": 0, "error": str(e)}

    def _execute_create_meeting_summary(self, step: ActionStep, state: ExecutionState) -> dict:
        """
        Execute meeting summary creation step.

        Combines:
        1. Structured meeting overview (from HTML main content)
        2. Agenda category document summaries
        3. Minutes category document summaries

        Into a consolidated meeting summary.
        """
        structured_overview = step.params.get("structured_overview")
        overview = step.params.get("overview")

        # Filter document summaries by category (agenda, minutes only)
        document_summaries = state.get("document_summaries", [])
        meeting_docs = [doc for doc in document_summaries if doc.category in ["agenda", "minutes"]]

        logger.info("Creating meeting summary:")
        logger.info(f"  - Structured overview: {'Yes' if structured_overview else 'No'}")
        logger.info(
            f"  - Agenda documents: {len([d for d in meeting_docs if d.category == 'agenda'])}"
        )
        logger.info(
            f"  - Minutes documents: {len([d for d in meeting_docs if d.category == 'minutes'])}"
        )

        # Build combined meeting content
        parts = []

        if structured_overview:
            parts.append(f"# 会議概要（HTMLより）\n\n{structured_overview}")

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

# 出力形式
議事要約のみを文章形式で出力してください（マークダウン見出しは不要、本文のみ）

**厳守事項：**
- **箇条書きや項目列挙の厳格な禁止**：
  - 箇条書き記号（・、-、*、+など）は絶対に使用しない
  - 番号付きリスト（1.、2.、①、②など）は絶対に使用しない
  - 改行による項目の列挙は行わない
  - すべての内容は文章形式（段落形式）で記述する
  - 複数の事項を述べる場合は「〜について、〜について、〜について」のように接続詞や句読点で自然に繋ぐ

# 含めるべき内容

以下の要素を文章中に自然に織り込んでください：
- 会議で議論された主要なトピック
- 各議題での重要な議論内容
- 決定事項・合意事項
- 今後のアクションアイテム（あれば）
- 次回会議の予定（あれば）

# 文量
500-1500文字程度

# 制約
- 推測や補完は行わない
- 提供されたコンテンツに記載されている内容のみを使用
- 「について：」などの空虚な表現は避ける
- 具体的な内容を含める
- 冗長な表現を避ける
- 重要な情報を優先
- 議事の流れを保持
- 会議名と回数（「第X回○○会議」等）は必ず含めること

# 除外すべき形式的情報
- 会議の進行説明（「議事次第に沿って」「事務局説明を受けた後」「意見交換を行い閉会した」等）
- 配布資料の名称リスト（「〇〇資料、△△資料が配布された」等）※資料の中身の要約は残す
- 資料の公開/非公開情報（「一部非公開扱い」「資料公開中」等）
- 「事務局説明では」「〇〇委員からは」等の発言者導入句

# 表現の改善
- 「〜が確認された」「〜が報告された」→「〜である」「〜とされた」等、進行報告ではなく内容記述の表現を使用
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
                "structured_overview": bool(structured_overview),
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
        Execute summary integration step (Phase 3).

        Creates integrated summary:
        - Integrated overview (LLM-generated from initial_summary + document summaries) + URL

        The integrated overview combines:
        - Phase 2 meeting overview (initial_summary)
        - All document summaries (from Phase 2 document processing)
        into a comprehensive meeting summary.

        Note: Individual document summaries are NOT included in final_summary.
        Only the integrated overview is stored in state["final_summary"].
        """
        initial_summary = state.get("initial_summary", "")
        document_summaries = state.get("document_summaries", [])
        input_url = state.get("input_url", "")

        logger.info("Integrating summaries (Phase 3 - Markdown output):")
        logger.info(f"  - Initial overview: {'Yes' if initial_summary else 'No'}")
        logger.info(f"  - Document summaries: {len(document_summaries)}")
        logger.info(f"  - Input URL: {input_url}")

        # Build integrated summary in markdown format
        parts = []

        # 1. Generate integrated overview from initial_summary + document summaries using LLM
        logger.info("Generating integrated overview from meeting summary and document summaries...")

        # Build context from initial_summary and document summaries
        context_parts = []

        # Add initial_summary (meeting overview)
        context_parts.append(f"# 会議概要\n\n{initial_summary}" if initial_summary else "")

        # Add document summaries
        context_parts.append("\n\n# 関連資料の要約")
        if document_summaries:
            for doc in document_summaries:
                category_label = self._category_ja.get(doc.category, doc.category)
                context_parts.append(
                    f"\n\n### {doc.name} ({category_label})\n{doc.summary}"
                )

        combined_context = "\n".join(context_parts)

        if combined_context.strip():
            # Use LLM to generate integrated overview
            llm = self.model.llm()
            from langchain.prompts import PromptTemplate

            integration_prompt = PromptTemplate(
                input_variables=["content"],
                template="""以下の会議概要と関連資料の要約から、この会議全体の統合要約を作成してください。

# 入力内容
{content}

# 最重要ルール：ハルシネーション禁止

**絶対に守るべきこと：**
- 入力内容に明示的に書かれている情報のみを使用すること
- 資料のタイトルから内容を推測・想像しないこと
- 「〜と考えられる」「〜が想定される」等の推測表現は禁止
- 資料が要約されていない場合、その資料の内容には一切言及しないこと

**情報が少ない場合の対応：**
- 議事次第のみで内容が不明な場合は、「議事次第によると〜が議題とされた」と事実のみ記述
- 事務局資料が「再掲」の場合は、「事務局資料は第X回資料の再掲である」と明記
- 要約された資料がない場合は、「公開された資料のうち、新規の内容を含むものはない」と正直に記述
- 情報が限定的であることを隠さず、ありのままに伝える

# 出力要件
- 2000字以内の文章形式で出力
- 会議概要と関連資料の重要な内容を統合
- 文章形式（箇条書き禁止）
- 「だ・である調」で統一
- 重複を避け、簡潔に
- 日時、開催場所、参加者は省略する（会議名と回数は含める）
- 1つの段落で記述する（段落分けしない）
- 情報が少ない場合は短くてよい（無理に長くしない）

# 文章の品質
- 提供された情報のみを使用し、推測や補完は絶対に行わない
- 1文は1つの主題に絞り、主語・述語の対応を明確にする
- より適切な日本語の文章に推敲する

# 除外すべき形式的情報
- 会議の進行説明（「議事次第に沿って」「事務局説明を受けた後」「意見交換を行い閉会した」等）
- 配布資料の名称リスト（「〇〇資料、△△資料が配布された」等）※資料の中身の要約は残す
- 資料の公開/非公開情報（「一部非公開扱い」「資料公開中」等）
- 「事務局説明では」「〇〇委員からは」等の発言者導入句

# 表現の改善
- 「〜が確認された」「〜が報告された」→「〜である」「〜とされた」等、進行報告ではなく内容記述の表現を使用
""",
            )

            chain = integration_prompt | llm

            try:
                result = chain.invoke({"content": combined_context[:20000]})
                integrated_overview = result.content.strip()

                # Add URL on a new line (single newline between overview and URL)
                if input_url and input_url not in integrated_overview:
                    final_content = f"{integrated_overview}\n{input_url}"
                else:
                    final_content = integrated_overview

                parts.append(final_content)

                logger.info(f"Generated integrated overview: {len(integrated_overview)} characters")

            except Exception as e:
                logger.error(f"Error generating integrated overview: {e}")
                import traceback
                traceback.print_exc()
                # Fallback: use initial_summary if available
                if initial_summary:
                    if input_url:
                        fallback_content = f"{initial_summary.strip()}\n{input_url}"
                    else:
                        fallback_content = initial_summary.strip()
                    parts.append(fallback_content)

        # final_summary contains only the integrated overview + URL
        if parts:
            final_summary = "\n".join(parts)
        else:
            final_summary = ""

        state["final_summary"] = final_summary

        logger.info(f"Final summary (integrated overview only): {len(final_summary)} characters")
        logger.info(f"  - Source documents: {len(document_summaries)}")

        return {
            "summary_length": len(final_summary),
            "document_count": len(document_summaries),
        }

    def _execute_post_to_bluesky(self, step: ActionStep, state: ExecutionState) -> dict:
        """
        Execute Bluesky posting step.

        Posts the finalized summary to Bluesky.
        """
        logger.info("🟢 Blueskyに投稿...")

        # Get final summary and URL
        final_summary = state.get("final_summary", "")
        url = step.target  # URL is stored in step.target by action_planner

        if not final_summary:
            logger.warning("⚠️ Bluesky投稿用の最終要約がありません")
            state["bluesky_post_content"] = None
            state["bluesky_post_response"] = None
            return {"posted": False, "reason": "No summary"}

        try:
            # Format post content
            post_content = self._format_bluesky_content(final_summary, url)

            # Post to Bluesky via ssky
            # v2 does not support interactive mode, so always auto-post
            post_result = self._post_to_bluesky_via_ssky(post_content)

            if post_result["success"]:
                logger.info("✅ Blueskyへの投稿に成功しました")
                if post_result.get("uri"):
                    logger.debug(f"URI: {post_result['uri']}")
                state["bluesky_post_content"] = post_content
                if post_result.get("result"):
                    state["bluesky_post_response"] = str(post_result["result"])
                return {
                    "posted": True,
                    "uri": post_result.get("uri"),
                    "content_length": len(post_content),
                }
            else:
                logger.error(f"❌ Bluesky投稿に失敗しました: {post_result['error']}")
                state["bluesky_post_content"] = post_content
                state["bluesky_post_response"] = f"Error: {post_result['error']}"
                return {
                    "posted": False,
                    "reason": post_result["error"],
                }

        except Exception as e:
            logger.error(f"❌ Bluesky投稿で想定しないエラーが発生しました: {type(e).__name__}: {e}")
            state["bluesky_post_content"] = None
            state["bluesky_post_response"] = f"Exception: {type(e).__name__}: {e}"
            return {
                "posted": False,
                "reason": f"Exception: {type(e).__name__}: {e}",
            }

    def _format_bluesky_content(self, summary: str, url: str) -> str:
        """
        Format content for Bluesky posting.

        Only appends URL if it's a web URL (http/https) and not already in summary.
        Local file paths are not appended.
        """
        # Check if URL is a web URL and not already in summary
        if url and (url.startswith("http://") or url.startswith("https://")):
            if url in summary:
                return summary
            return f"{summary}\n{url}"
        else:
            # Don't append local file paths
            return summary

    def _post_to_bluesky_via_ssky(self, content: str) -> dict:
        """
        Post to Bluesky by executing ssky command directly.

        Returns:
            dict with keys: success, content, result, uri, error
        """
        # Get SSKY_USER from environment
        ssky_user = os.getenv("SSKY_USER")
        if not ssky_user:
            error_msg = "SSKY_USER environment variable not set. Format: 'USER:PASSWORD'"
            logger.error(f"❌ {error_msg}")
            return {"success": False, "content": content, "result": None, "error": error_msg}

        try:
            # Execute ssky post command
            result = subprocess.run(
                ["ssky", "post", "--json", content],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                # Parse JSON response on success
                try:
                    response_data = json.loads(result.stdout)
                    uri = response_data.get("uri")
                    return {
                        "success": True,
                        "content": content,
                        "result": result.stdout,
                        "uri": uri,
                        "error": None,
                    }
                except json.JSONDecodeError:
                    # If JSON parsing fails but returncode is 0, treat as success
                    return {
                        "success": True,
                        "content": content,
                        "result": result.stdout,
                        "uri": None,
                        "error": None,
                    }
            else:
                error_msg = result.stderr or result.stdout or "Unknown error"
                logger.error(f"❌ sskyコマンドが失敗しました: {error_msg}")
                return {
                    "success": False,
                    "content": content,
                    "result": None,
                    "error": error_msg,
                }

        except subprocess.TimeoutExpired:
            error_msg = "sskyコマンドがタイムアウトしました (30秒)"
            logger.error(f"❌ {error_msg}")
            return {"success": False, "content": content, "result": None, "error": error_msg}
        except Exception as e:
            error_msg = f"sskyコマンド実行エラー: {e}"
            logger.error(f"❌ {error_msg}")
            return {"success": False, "content": content, "result": None, "error": error_msg}
