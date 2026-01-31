"""
WordSummarizer sub-agent for Plan-Action architecture (v2).

This version extracts important sections from TOC with page numbers,
then reads only those pages to minimize token consumption.
"""


from langchain.chains.summarize import load_summarize_chain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from .. import Model, logger
from ..state_v2 import WordState


class ImportantSection(BaseModel):
    """Important section identified from TOC."""

    section_title: str = Field(description="セクションのタイトル")
    page_number: int | None = Field(description="ページ番号（特定できない場合はNone）")
    score: int = Field(description="重要度スコア（1-5点）", ge=1, le=5)
    reason: str = Field(description="このセクションが重要な理由とスコアの根拠")


class ImportantSections(BaseModel):
    """List of important sections."""

    sections: list[ImportantSection] = Field(description="重要なセクションのリスト")


class WordSummarizer:
    """
    Word document summarization sub-agent (v2 - selective page reading).

    Extracts TOC, identifies important sections with page numbers,
    then reads only those pages to create summary.
    """

    # Maximum pages to analyze
    MAX_PAGES_FOR_TITLE = 5
    MAX_PAGES_FOR_TOC = 15  # Increased to capture more TOC
    MAX_PAGES_TO_READ = 20  # Maximum pages to read for summary

    def __init__(self, model: Model | None = None):
        """
        Initialize WordSummarizer sub-agent.

        Args:
            model: Model instance for LLM access. If None, uses default Model().
        """
        self.model = model if model is not None else Model()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build the StateGraph for Word document summarization.

        Returns:
            Compiled StateGraph for Word processing workflow
        """
        graph = StateGraph(WordState)

        # Four-stage pipeline
        graph.add_node("extract_title", self._extract_title)
        graph.add_node("extract_toc_with_pages", self._extract_toc_with_pages)
        graph.add_node("select_important_pages", self._select_important_pages)
        graph.add_node("generate_summary", self._generate_summary)

        # Linear flow
        graph.set_entry_point("extract_title")
        graph.add_edge("extract_title", "extract_toc_with_pages")
        graph.add_edge("extract_toc_with_pages", "select_important_pages")
        graph.add_edge("select_important_pages", "generate_summary")
        graph.add_edge("generate_summary", END)

        return graph

    def _extract_title(self, state: WordState) -> WordState:
        """
        Extract document title from first few pages.

        Args:
            state: Current state with pdf_pages

        Returns:
            Updated state with title field
        """
        llm = self.model.llm()
        pdf_pages = state["pdf_pages"]

        # Analyze first N pages
        title_pages = min(self.MAX_PAGES_FOR_TITLE, len(pdf_pages))
        merged_text = "\n\n".join(
            [f"--- ページ {i+1} ---\n{text}" for i, text in enumerate(pdf_pages[:title_pages])]
        )

        logger.info(f"先頭{title_pages}ページからタイトルを抽出します")

        title_prompt = PromptTemplate(
            input_variables=["text"],
            template="""あなたは文書分析の専門家です。以下のWord文書の最初のページからタイトルを抽出してください。

# 抽出対象テキスト
{text}

# 出力
タイトルのみを出力してください（説明不要）。
副題・組織名・発行日は除外してください。
            """,
        )

        chain = title_prompt | llm
        result = chain.invoke({"text": merged_text})
        extracted_title = result.content.strip()

        logger.info(f"抽出されたタイトル: {extracted_title}")

        return {"title": extracted_title}

    def _extract_toc_with_pages(self, state: WordState) -> WordState:
        """
        Extract table of contents WITH page numbers.

        Args:
            state: Current state with pdf_pages

        Returns:
            Updated state with table_of_contents field
        """
        llm = self.model.llm()
        pdf_pages = state["pdf_pages"]

        # Analyze first N pages
        toc_pages = min(self.MAX_PAGES_FOR_TOC, len(pdf_pages))
        merged_text = "\n\n".join(
            [f"--- ページ {i+1} ---\n{text}" for i, text in enumerate(pdf_pages[:toc_pages])]
        )

        logger.info(f"先頭{toc_pages}ページから目次（ページ番号付き）を抽出します")

        toc_prompt = PromptTemplate(
            input_variables=["text"],
            template="""あなたは文書分析の専門家です。以下のテキストから目次を抽出してください。

# 抽出対象テキスト
{text}

# 抽出手順

ステップ1: 目次セクションを特定する
- 「目次」「Contents」「もくじ」などの見出しを探す
- 章・節・項の階層的なリストを探す

ステップ2: 目次を抽出する（**ページ番号を保持**）
- セクションタイトルとページ番号を抽出
- 階層構造を保持
- ページ番号が記載されている場合は必ず含める

# 出力形式

**目次が見つかった場合:**
```
1. はじめに ...... 1
2. 背景 ...... 3
  2.1 現状の課題 ...... 3
  2.2 これまでの取組 ...... 5
3. 基本方針 ...... 8
  3.1 目標 ...... 8
  3.2 主な施策 ...... 10
4. 具体的な取組 ...... 15
  4.1 施策A ...... 15
  4.2 施策B ...... 18
5. まとめ ...... 25
```

**目次が見つからない場合:**
```
目次なし
```

# 重要
- ページ番号を**必ず含める**（ドット、スペース、ハイフンなどで区切られている）
- ページ番号がない場合は「...... (ページ番号なし)」と記載
- 階層構造を保持
            """,
        )

        chain = toc_prompt | llm
        result = chain.invoke({"text": merged_text})

        extracted_toc = result.content.strip()

        if extracted_toc and "目次なし" not in extracted_toc:
            logger.info("目次を抽出しました")
            logger.info(f"目次内容:\n{extracted_toc[:500]}...")
        else:
            logger.info("目次が見つかりませんでした")

        return {"table_of_contents": [{"type": "toc", "content": extracted_toc}]}

    def _select_important_pages(self, state: WordState) -> WordState:
        """
        Select important pages based on TOC analysis.

        Args:
            state: Current state with table_of_contents

        Returns:
            Updated state with selected pages info
        """
        llm = self.model.llm()
        table_of_contents = state.get("table_of_contents", [])
        pdf_pages = state["pdf_pages"]
        total_pages = len(pdf_pages)

        # Check if TOC exists
        has_toc = False
        toc_content = ""
        if table_of_contents:
            for toc_item in table_of_contents:
                if toc_item.get("type") == "toc":
                    toc_content = toc_item.get("content", "")
                    if toc_content and "目次なし" not in toc_content:
                        has_toc = True
                    break

        if not has_toc:
            logger.info("目次がないため、全ページから要約を作成します")
            # Mark that full text will be used (handled in generate_summary)
            return {}

        logger.info("目次から重要なセクションとページ番号を特定します")

        parser = PydanticOutputParser(pydantic_object=ImportantSections)

        select_prompt = PromptTemplate(
            input_variables=["toc", "total_pages", "format_instructions"],
            template="""あなたは文書分析の専門家です。目次から要約に必要な重要セクションをスコアリングして特定してください。

# 目次
{toc}

# 総ページ数
{total_pages}ページ

# スコアリング基準（1-5点）

**5点（最優先 - 要約に必須）:**
- まとめ・結論・提言（文書の結論）
- 基本方針・目標・コンセプト（文書の方向性）
- 今後の方向性・スケジュール（文書の展望）

**4点（高優先 - 要約の質を高める）:**
- 背景・経緯・課題（文書の動機）
- 主要施策・取組内容の概要（具体的な内容）
- 実績・評価・成果（文書の成果）
- 重要ポイント・留意事項（重要な注意点）

**3点（中優先 - 補足的に重要）:**
- 現状分析・データ（背景の補強）
- 個別施策の要点（具体例）

**2点（低優先 - あれば良い）:**
- 個別施策の詳細
- 用語解説・定義

**1点（不要 - 要約には含めない）:**
- 参考資料・付録
- 謝辞・問い合わせ先
- 目次自体・表紙

# 抽出手順

ステップ1: 目次を分析する
- 各セクションのタイトルから内容を推測
- 文書タイプ（報告書、計画書、ガイドライン等）を判断

ステップ2: 全セクションをスコアリングする
- 上記の基準に基づいて1-5点で評価
- スコアの理由を明確に記述
- ページ番号を抽出（記載がある場合）

ステップ3: 高スコアセクションを選択する
- スコア4-5点のセクションを優先
- 合計ページ数が15ページ以内になるよう調整
- 最低3セクション、最大10セクションを選択

# 出力フォーマット
各セクションについて以下を記述：
- section_title: セクションのタイトル
- page_number: ページ番号（目次に記載がある場合、なければNone）
- score: 重要度スコア（1-5点）
- reason: スコアの理由（なぜこのスコアなのか、要約にどう貢献するか）

{format_instructions}

# 制約
- 全セクションをスコアリング（スコア1-2のものも含む）
- 合計15ページ以内に抑える
- ページ番号が特定できない場合はNoneを指定
- スコア4-5点を優先的に選択
            """,
        )

        chain = select_prompt | llm | parser

        try:
            result = chain.invoke({
                "toc": toc_content,
                "total_pages": total_pages,
                "format_instructions": parser.get_format_instructions()
            })

            important_sections = result.sections

            # Sort by score (descending)
            sorted_sections = sorted(important_sections, key=lambda x: x.score, reverse=True)
            logger.info(f"{len(sorted_sections)}個のセクションをスコアリングしました")

            # Filter by score (4-5 points)
            high_priority_sections = [s for s in sorted_sections if s.score >= 4]

            if not high_priority_sections:
                logger.warning("スコア4点以上のセクションがありません。スコア3点以上を使用します。")
                high_priority_sections = [s for s in sorted_sections if s.score >= 3]

            logger.info(f"高優先度セクション: {len(high_priority_sections)}個（スコア4-5点）")

            # Extract page numbers
            pages_to_read = set()
            for section in high_priority_sections:
                if section.page_number is not None:
                    # Add page and surrounding pages (±1 page)
                    page_num = section.page_number
                    for p in range(max(1, page_num - 1), min(total_pages + 1, page_num + 2)):
                        pages_to_read.add(p)
                    logger.info(
                        f"  - [{section.score}点] {section.section_title} (ページ{section.page_number}): {section.reason}"
                    )
                else:
                    logger.info(
                        f"  - [{section.score}点] {section.section_title} (ページ不明): {section.reason}"
                    )

            # Limit to MAX_PAGES_TO_READ
            if len(pages_to_read) > self.MAX_PAGES_TO_READ:
                logger.warning(
                    f"選択ページ数が{len(pages_to_read)}ページで上限を超えています。"
                    f"最初の{self.MAX_PAGES_TO_READ}ページのみ使用します。"
                )
                pages_to_read = set(sorted(pages_to_read)[:self.MAX_PAGES_TO_READ])

            if pages_to_read:
                sorted_pages = sorted(pages_to_read)
                logger.info(f"読み込むページ: {sorted_pages} (合計{len(sorted_pages)}ページ)")

                # Store selected pages and sections info
                return {
                    "table_of_contents": [
                        {
                            "type": "selected_pages",
                            "page_numbers": sorted_pages,
                            "sections": [
                                {
                                    "title": s.section_title,
                                    "page": s.page_number,
                                    "score": s.score,
                                    "reason": s.reason
                                }
                                for s in high_priority_sections
                            ]
                        }
                    ]
                }
            else:
                logger.warning("ページ番号が特定できませんでした。目次のみで要約します。")
                return {}

        except Exception as e:
            logger.error(f"重要セクション特定中にエラー: {e}")
            logger.info("フォールバック：目次のみで要約します")
            return {}

    def _generate_summary(self, state: WordState) -> WordState:
        """
        Generate summary from selected pages + TOC.

        Args:
            state: Current state with title, table_of_contents

        Returns:
            Updated state with summary field
        """
        llm = self.model.llm()
        title = state.get("title", "")
        table_of_contents = state.get("table_of_contents", [])
        pdf_pages = state["pdf_pages"]

        # Check if we have selected pages
        has_selected_pages = False
        selected_pages = []
        toc_content = ""

        for toc_item in table_of_contents:
            if toc_item.get("type") == "selected_pages":
                has_selected_pages = True
                page_numbers = toc_item.get("page_numbers", [])
                # Extract text from selected pages (convert 1-based to 0-based)
                selected_pages = [
                    (page_num, pdf_pages[page_num - 1])
                    for page_num in page_numbers
                    if 0 < page_num <= len(pdf_pages)
                ]
                logger.info(f"選択された{len(selected_pages)}ページから要約を作成します")
            elif toc_item.get("type") == "toc":
                toc_content = toc_item.get("content", "")

        if has_selected_pages and selected_pages:
            # TOC + selected pages summarization
            selected_text = "\n\n".join([
                f"--- ページ {page_num} ---\n{text}"
                for page_num, text in selected_pages
            ])

            summary_prompt = PromptTemplate(
                input_variables=["title", "toc", "content"],
                template="""あなたはWord文書の要約専門家です。以下のタイトル、目次、重要ページから詳細な要約を作成してください。

# 文書タイトル
「{title}」

# 目次
{toc}

# 重要ページの内容
{content}

# 要約作成手順

ステップ1: 全体構造を把握する
- 目次から文書の全体像を理解
- 重要ページの内容から具体的な詳細を把握

ステップ2: 要約を構成する
以下の項目を含める（該当する場合）：
- 文書の目的・背景
- 主要な課題・論点
- 基本方針・目標
- 具体的な施策・取組
- 実績・評価（該当する場合）
- 結論・提言・今後の方向性

ステップ3: 詳細な要約を作成する
- 重要ページの具体的な内容を反映
- 数値・指標・固有名詞は正確に記載
- 目次の構造を活用して論理的に整理

# 出力形式
要約文のみを出力（Markdown不要、改行は適宜使用）

# 文量の目安
- 最小: 500文字程度
- 推奨: 1000-3000文字（内容の豊富さに応じて調整）
- 最大: 5000文字以内

# 制約事項
- 提供されたページの内容のみを使用
- 推測や補完は行わない
- 重要な数値・固有名詞は省略しない
- 詳細かつ分かりやすく
                """,
            )

            chain = summary_prompt | llm
            result = chain.invoke({
                "title": title,
                "toc": toc_content if "目次なし" not in toc_content else "（目次なし）",
                "content": selected_text
            })
            summary = result.content.strip()

        elif toc_content and "目次なし" not in toc_content:
            # TOC-only summarization
            logger.info("目次のみから要約を作成します")

            toc_summary_prompt = PromptTemplate(
                input_variables=["title", "toc"],
                template="""あなたはWord文書の要約専門家です。以下のタイトルと目次から要約を作成してください。

# 文書タイトル
「{title}」

# 目次
{toc}

# 要約作成
目次の構造から推察される文書の内容を要約してください。
- 文書の目的・位置づけ
- 主要な検討項目・論点
- 文書の特徴・性格

# 文量: 300-1500文字程度
# 制約: 目次にない具体的内容は推測しない
                """,
            )

            chain = toc_summary_prompt | llm
            result = chain.invoke({"title": title, "toc": toc_content})
            summary = result.content.strip()

        else:
            # Full-text summarization (fallback)
            logger.info("目次も選択ページもないため、全文から要約を作成します")

            documents = [
                Document(page_content=text, metadata={"page": i + 1})
                for i, text in enumerate(pdf_pages)
            ]

            summary_prompt_template = """以下のWord文書の詳細な要約を作成してください。

文書タイトル: {title}

文書内容:
{{text}}

# 要約（500-5000文字、詳細かつ分かりやすく）
            """

            summary_prompt = PromptTemplate(
                template=summary_prompt_template.format(title=title), input_variables=["text"]
            )

            chain = load_summarize_chain(llm, chain_type="stuff", prompt=summary_prompt)
            result = chain.invoke({"input_documents": documents})
            summary = result["output_text"].strip()

        logger.info(f"Word要約生成完了 ({len(summary)}文字)")

        return {"summary": summary}

    def invoke(self, input_data: dict) -> dict:
        """
        Execute Word document summarization.

        Args:
            input_data: Dict with keys:
                - pdf_pages: list[str] - PDF page texts
                - url: str - Source URL (optional)

        Returns:
            Dict with keys:
                - title: str - Extracted title
                - summary: str - Generated summary
                - table_of_contents: list[dict] | None - Selected pages info
        """
        compiled = self.graph.compile()
        result = compiled.invoke(input_data)
        return result
