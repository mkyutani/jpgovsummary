"""
PowerPointSummarizer Sub-Agent

This sub-agent processes PowerPoint presentations in an isolated context,
reducing main workflow token consumption by 2000-3000 tokens per document.

Architecture for large presentations (up to 200 pages):
- Multi-stage pipeline with 20-page batch processing
- Stage 1: Extract title (first 3 pages)
- Stage 2: Score slides in 20-page batches (scales to 200+ pages)
- Stage 3: Select high-scoring slides (non-LLM logic)
- Stage 4: Generate summary from selected content

Modern prompt engineering techniques applied:
- Clear step-by-step instructions in each prompt
- Structured output using Pydantic for reliability
- Explicit reasoning requests for better quality
- Context-aware scoring criteria

This is an independent LangGraph StateGraph with its own PowerPointState,
enabling parallel execution and better error isolation.
"""

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from jpgovsummary import Model, logger
from jpgovsummary.state import PowerPointState


class SlideInfo(BaseModel):
    """Individual slide metadata with importance score."""

    page: int = Field(description="ページ番号")
    title: str = Field(description="スライドタイトル")
    score: int = Field(description="重要度スコア (1-5)", ge=1, le=5)
    reason: str = Field(description="スコア付け理由")


class SlideAnalysis(BaseModel):
    """Collection of analyzed slides (20-page batch result)."""

    slides: list[SlideInfo] = Field(description="スライド分析結果リスト")


class PowerPointSummarizer:
    """
    PowerPoint summarization sub-agent with batch processing.

    Designed for large presentations (up to 200 pages):
    - Processes slides in 20-page batches to manage context
    - Accumulates results across batches
    - Selects top-scoring slides for final summary

    Usage:
        summarizer = PowerPointSummarizer()
        result = summarizer.invoke({
            "pdf_pages": ["page1 text", "page2 text", ...],  # up to 200 pages
            "url": "https://example.go.jp/document.pdf"
        })
        summary = result["summary"]
    """

    # Batch size for slide analysis (optimized for token limits)
    PAGES_PER_BATCH = 20

    def __init__(self, model: Model | None = None):
        """
        Initialize PowerPointSummarizer.

        Args:
            model: Optional Model instance. If None, uses global Model().
        """
        self.model = model or Model()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph StateGraph for PowerPoint processing.

        Returns:
            StateGraph for compilation
        """
        graph = StateGraph(PowerPointState)

        # Add processing nodes
        graph.add_node("extract_title", self._extract_title)
        graph.add_node("score_slides", self._score_slides_batched)
        graph.add_node("select_content", self._select_content)
        graph.add_node("generate_summary", self._generate_summary)

        # Define edges (linear pipeline)
        graph.set_entry_point("extract_title")
        graph.add_edge("extract_title", "score_slides")
        graph.add_edge("score_slides", "select_content")
        graph.add_edge("select_content", "generate_summary")
        graph.add_edge("generate_summary", END)

        return graph

    def _extract_title(self, state: PowerPointState) -> PowerPointState:
        """
        Extract presentation title from first 3 pages.

        Uses modern prompt engineering:
        - Clear instructions with step-by-step guidance
        - Specific examples and constraints
        - Direct output format specification

        Args:
            state: Current PowerPointState

        Returns:
            Updated state with title field
        """
        llm = self.model.llm()
        pdf_pages = state["pdf_pages"]
        display_name = state.get("display_name", "")

        # Analyze first 3 pages for title extraction
        pages_to_analyze = min(3, len(pdf_pages))
        sample_texts = pdf_pages[:pages_to_analyze]

        # Prepare merged text
        if pages_to_analyze == 1:
            merged_text = f"ページ1:\n{sample_texts[0]}"
        else:
            merged_text = "\n\n".join(
                [f"ページ{i+1}:\n{text}" for i, text in enumerate(sample_texts)]
            )

        title_prompt = PromptTemplate(
            input_variables=["text"],
            template="""あなたはPowerPoint資料のタイトル抽出の専門家です。以下のテキストからプレゼンテーションの正式なタイトルを抽出してください。

# 入力テキスト
{text}

# 抽出手順
ステップ1: 最初のページに記載されているメインタイトルを特定してください
ステップ2: 副題がある場合は含めてください
ステップ3: 組織名、日付、ページ番号は除外してください
ステップ4: タイトルの文言は変更せず、原文のまま抽出してください

# 出力形式
タイトルのみを1行で出力してください。説明や前置きは不要です。

# 出力例
「令和6年度 国土強靱化関係予算概算要求の概要」
            """,
        )

        chain = title_prompt | llm
        result = chain.invoke({"text": merged_text})
        extracted_title = result.content.strip()

        log_prefix = f"[{display_name}] " if display_name else ""
        logger.info(f"{log_prefix}PowerPointタイトル抽出: 「{extracted_title.replace('\n', '\\n')}」")

        return {"title": extracted_title}

    def _score_slides_batched(self, state: PowerPointState) -> PowerPointState:
        """
        Score slides by importance in 20-page batches.

        Handles large presentations (up to 200 pages) by:
        - Processing in PAGES_PER_BATCH (20) page chunks
        - Accumulating results across batches
        - Graceful error handling for individual batches

        Args:
            state: Current PowerPointState with pdf_pages and title

        Returns:
            Updated state with scored_slides field
        """
        pdf_pages = state["pdf_pages"]
        total_pages = len(pdf_pages)
        all_slides = []
        display_name = state.get("display_name", "")
        log_prefix = f"[{display_name}] " if display_name else ""

        logger.info(
            f"{log_prefix}スライド分析開始: 総ページ数={total_pages}, バッチサイズ={self.PAGES_PER_BATCH}"
        )

        for start_page in range(0, total_pages, self.PAGES_PER_BATCH):
            end_page = min(start_page + self.PAGES_PER_BATCH - 1, total_pages - 1)
            try:
                logger.info(f"{log_prefix}バッチ処理中 (ページ{start_page+1}-{end_page+1}/{total_pages})")
                slide_analysis = self._analyze_slide_batch(pdf_pages, start_page, end_page)

                for slide in slide_analysis.slides:
                    logger.info(
                        f"{log_prefix}  ページ{slide.page}: {slide.title} → スコア: {slide.score} - {slide.reason}"
                    )
                all_slides.extend(slide_analysis.slides)

            except Exception as e:
                logger.warning(
                    f"{log_prefix}⚠️ バッチ分析失敗 (ページ{start_page+1}-{end_page+1}): {e}"
                )
                # Continue processing remaining batches even if one fails

        logger.info(f"{log_prefix}スライド分析完了: {len(all_slides)}枚のスライドを分析")

        # Convert Pydantic objects to dicts for state storage
        scored_slides = [
            {"page": s.page, "title": s.title, "score": s.score, "reason": s.reason}
            for s in all_slides
        ]

        return {"scored_slides": scored_slides}

    def _analyze_slide_batch(
        self, texts: list[str], start_page: int, end_page: int
    ) -> SlideAnalysis:
        """
        Analyze a batch of slides (up to 20 pages) with importance scoring.

        Modern prompt engineering applied:
        - Explicit step-by-step reasoning instructions
        - Clear scoring criteria with examples
        - Structured output for reliability
        - Retry logic for robustness

        Args:
            texts: Full PDF page texts
            start_page: Start page index (0-based)
            end_page: End page index (0-based, inclusive)

        Returns:
            SlideAnalysis with scored slides
        """
        llm = self.model.llm()
        parser = PydanticOutputParser(pydantic_object=SlideAnalysis)

        # Extract page range
        page_texts = texts[start_page : end_page + 1]
        content = "\n\n".join(
            [f"--- ページ {start_page + i + 1} ---\n{text}" for i, text in enumerate(page_texts)]
        )

        prompt = PromptTemplate(
            input_variables=["content", "format_instructions"],
            template="""あなたはPowerPoint資料の重要度評価の専門家です。以下のスライドを分析し、各ページのタイトルと重要度を評価してください。

# 入力スライド
{content}

# 分析手順
各ページについて、以下のステップで分析してください：

ステップ1: スライドのタイトルを抽出
- ページの最上部や冒頭に記載されている見出しを特定
- タイトルが不明確な場合は、ページの主題を簡潔に表現

ステップ2: スライドの内容タイプを判定
- アジェンダ/目次: 全体構成を示すページ
- 論点/検討事項: 議論すべき重要なポイント
- まとめ/結論: 資料の要点や結論
- 概要/方針: 基本方針や全体像
- 詳細説明: 個別施策や事例の説明
- 表紙/その他: タイトルページや補足資料

ステップ3: 重要度スコアを付与 (1-5点)
以下の基準に従ってスコアリングしてください：

**5点（最重要）:**
- アジェンダ、目次
- 主な論点、検討事項、今回の論点
- まとめ、結論、骨子

**4点（高優先）:**
- 概要、基本方針、ポイント、要点
- 重要課題、今後の方針、スケジュール
- 資料タイトルに直接関連する内容

**3点（中優先）:**
- 振り返り、背景、課題
- 分析結果、戦略
- 個別施策の概要説明

**2点（低優先）:**
- 詳細説明、補足資料
- 事例紹介、参考資料
- 個別取組の詳細
- セクション見出し（タイトルのみで内容がない仕切りスライド）

**1点（最低優先）:**
- 表紙、タイトルページ
- 事務連絡、その他

ステップ4: スコア付け理由を記述
- 判定したスライドタイプと重要度の根拠を簡潔に説明

# 出力フォーマット
以下のJSON形式で出力してください：

{format_instructions}

# 重要な注意事項
- 全ページを必ず分析してください
- スコアは内容の重要性に基づいて厳格に判定してください
- 同じスコアが複数ページに付与されても問題ありません
- 理由は具体的かつ簡潔に記述してください
            """,
        )

        chain = prompt | llm | parser

        # Retry logic for JSON parsing failures
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.info(f"スライド分析リトライ ({attempt+1}/{max_retries})")

                result = chain.invoke(
                    {
                        "content": content,
                        "format_instructions": parser.get_format_instructions(),
                    }
                )
                return result

            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"❌ スライド分析失敗: {e}")
                    return SlideAnalysis(slides=[])
                continue

        return SlideAnalysis(slides=[])

    def _select_content(self, state: PowerPointState) -> PowerPointState:
        """
        Select high-scoring slides for summarization (non-LLM logic).

        Selection strategy:
        1. Select all slides with score >= 4
        2. Sort by page number for coherent reading

        This is a deterministic, non-LLM step for efficiency.

        Args:
            state: Current PowerPointState with scored_slides

        Returns:
            Updated state with selected_content field
        """
        pdf_pages = state["pdf_pages"]
        scored_slides_dicts = state.get("scored_slides", [])
        display_name = state.get("display_name", "")
        log_prefix = f"[{display_name}] " if display_name else ""

        # Convert dicts back to SlideInfo objects
        scored_slides = [
            SlideInfo(
                page=s["page"], title=s["title"], score=s["score"], reason=s.get("reason", "")
            )
            for s in scored_slides_dicts
        ]

        total_pages = len(pdf_pages)

        if not scored_slides:
            # Fallback: use all pages if scoring failed
            merged_content = "\n\n".join(
                [f"--- ページ {i+1} ---\n{text}" for i, text in enumerate(pdf_pages)]
            )
            logger.info(f"{log_prefix}⚠️ スライド分析結果なし - 全ページ使用")
            return {"selected_content": merged_content}

        # Select all slides with score >= 4
        selected_slides = [slide for slide in scored_slides if slide.score >= 4]

        if not selected_slides:
            # Fallback: use top-scoring slides if none scored >= 4
            max_score = max(slide.score for slide in scored_slides)
            selected_slides = [slide for slide in scored_slides if slide.score == max_score]
            logger.info(f"{log_prefix}スコア4以上なし - 最高スコア{max_score}点のスライドを使用")

        # Sort by page number for coherent reading
        selected_slides = sorted(selected_slides, key=lambda x: x.page)

        logger.info(
            f"{log_prefix}選択スライド: {','.join([f'p{slide.page}({slide.score}点)' for slide in selected_slides])} "
            f"({len(selected_slides)}/{total_pages}枚)"
        )

        # Extract selected slide texts
        selected_texts = []
        for slide in selected_slides:
            page_idx = slide.page - 1  # Convert 1-based to 0-based
            if 0 <= page_idx < len(pdf_pages):
                selected_texts.append(
                    f"--- ページ {slide.page} ({slide.title}, スコア: {slide.score}) ---\n{pdf_pages[page_idx]}"
                )

        merged_content = "\n\n".join(selected_texts)

        return {"selected_content": merged_content}

    def _generate_summary(self, state: PowerPointState) -> PowerPointState:
        """
        Generate summary from selected slide content.

        Modern prompt engineering:
        - Clear role definition
        - Step-by-step summarization instructions
        - Explicit constraints and output format
        - Content-adaptive structure

        Args:
            state: Current PowerPointState with title and selected_content

        Returns:
            Updated state with summary field
        """
        llm = self.model.llm()
        title = state.get("title", "")
        selected_content = state.get("selected_content", "")
        display_name = state.get("display_name", "")
        log_prefix = f"[{display_name}] " if display_name else ""

        summary_prompt = PromptTemplate(
            input_variables=["title", "content"],
            template="""あなたはPowerPoint資料の要約の専門家です。

# 最重要ルール
- <input>タグ内に明示的に書かれている情報のみを使用すること
- 推測・創作・補完は一切行わないこと
- 「〜と考えられる」「〜が想定される」「〜が見込まれる」等の推測表現は絶対に使用しないこと
- 情報が少ない場合は短くてよい（無理に長くしない、同じ内容を繰り返さない）

# 要約作成手順
1. 資料の目的と性質を理解する（政策検討、事業報告、説明資料、計画資料など）
2. 該当する項目を含める：目的・概要、背景・課題、検討事項・論点、実績・成果、制度・仕組み、計画・施策の詳細、数値・指標（目標値、予算額、スケジュール等）、結論・方向性
3. スライドの内容を詳しくまとめ、重要な数値・固有名詞・専門用語は正確に記載する

# 出力形式
要約内容のみを出力してください。見出しや項目ラベルは含めないでください。
改行は適宜使用して読みやすくしてください。

# 文量の目安
- 最小: 500文字程度
- 推奨: 1000-3000文字（資料の内容に応じて調整）
- 最大: 5000文字以内

# 制約事項
- 該当しない項目は無理に含めない
- 重要な数値や固有名詞は省略しない
- 資料タイトル（回数含む）は必ず含めること
- 「〜の要約である」等の冗長な導入句は使用しない
- 「〜が確認された」→「〜する方針である」「〜とされている」等、進行報告ではなく内容記述の表現を使用

# 重要：以下の<input>タグ内のみが要約対象です。上記の指示やルールは要約に含めないでください。

<input>
資料タイトル: 「{title}」

選択された重要スライド:
{content}
</input>
            """,
        )

        chain = summary_prompt | llm
        result = chain.invoke({"title": title, "content": selected_content})

        summary = result.content.strip()
        logger.info(f"{log_prefix}PowerPoint要約生成完了 ({len(summary)}文字)")

        return {"summary": summary}

    def invoke(self, input_data: dict) -> dict:
        """
        Execute PowerPoint summarization pipeline.

        Handles large presentations (up to 200 pages) through:
        - Batched slide analysis (20 pages per batch)
        - Intelligent slide selection
        - Focused summarization on key content

        Args:
            input_data: Input dict with keys:
                - pdf_pages: list[str] - Text content of PDF pages (up to 200)
                - url: str - Source URL for reference

        Returns:
            Result dict with keys:
                - summary: str - Generated summary
                - title: str - Extracted title
                - url: str - Source URL (pass-through)

        Example:
            >>> summarizer = PowerPointSummarizer()
            >>> result = summarizer.invoke({
            ...     "pdf_pages": ["page1", "page2", ..., "page200"],
            ...     "url": "https://example.go.jp/large-presentation.pdf"
            ... })
            >>> print(result["summary"])
        """
        compiled_graph = self.graph.compile()
        final_state = compiled_graph.invoke(input_data)

        return {
            "summary": final_state.get("summary", ""),
            "title": final_state.get("title", ""),
            "url": input_data.get("url", ""),
        }
