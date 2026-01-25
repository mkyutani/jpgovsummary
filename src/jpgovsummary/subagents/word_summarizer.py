"""
WordSummarizer sub-agent for Plan-Action architecture.

This sub-agent handles Word document summarization with isolated context,
using table of contents (TOC) based summarization when available.
"""

from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from .. import Model, logger
from ..state_v2 import WordState


class WordSummarizer:
    """
    Word document summarization sub-agent.

    Extracts title and table of contents (TOC) from Word documents,
    then generates structured summaries based on TOC or full text.
    """

    # Maximum pages to analyze for title and TOC extraction
    MAX_PAGES_FOR_TITLE = 5
    MAX_PAGES_FOR_TOC = 10

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

        # Three-stage pipeline
        graph.add_node("extract_title", self._extract_title)
        graph.add_node("extract_toc", self._extract_toc)
        graph.add_node("generate_summary", self._generate_summary)

        # Linear flow
        graph.set_entry_point("extract_title")
        graph.add_edge("extract_title", "extract_toc")
        graph.add_edge("extract_toc", "generate_summary")
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
            input_variables=["text", "pages"],
            template="""あなたは文書分析の専門家です。以下のWord文書の最初の{pages}ページからタイトルを抽出してください。

# 抽出対象テキスト
{text}

# タイトル抽出手順

ステップ1: 表紙・冒頭ページを確認する
- 文書の主タイトルを探す
- 副題がある場合は主タイトルのみを選択
- 組織名・部署名・日付は除外

ステップ2: タイトルを特定する
- 最も大きな見出し、または明確に「タイトル」として記載されているもの
- 章タイトル（「第1章」など）ではなく、文書全体のタイトル
- 複数ページにわたる場合は、最初に出現するものを採用

# 出力形式
タイトルのみを出力してください（説明や前置きは不要）

# 制約事項
- 主タイトルのみ抽出
- 副題・サブタイトルは除外
- 組織名・発行日は除外
- 文書全体を代表するタイトル
            """,
        )

        chain = title_prompt | llm
        result = chain.invoke({"text": merged_text, "pages": title_pages})
        extracted_title = result.content.strip()

        logger.info(f"抽出されたタイトル: {extracted_title}")

        return {"title": extracted_title}

    def _extract_toc(self, state: WordState) -> WordState:
        """
        Extract table of contents from document.

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

        logger.info(f"先頭{toc_pages}ページから目次を抽出します")

        toc_prompt = PromptTemplate(
            input_variables=["text", "pages"],
            template="""あなたは文書分析の専門家です。以下のWord文書の最初の{pages}ページから目次を抽出してください。

# 抽出対象テキスト
{text}

# 目次抽出手順

ステップ1: 目次セクションを特定する
以下の特徴を持つ部分を探す：
- 「目次」「Contents」「もくじ」などの見出し
- 章・節・項の階層的なリスト
- ページ番号が記載されている項目リスト
- 文書の構成を示す構造的なリスト

ステップ2: 目次を抽出する
- 階層構造を保持（インデントや番号を維持）
- ページ番号は除外
- 章・節・項のタイトルのみ抽出

# 抽出対象外（目次ではないもの）
- 単なる箇条書き（施策や項目の列挙）
- 図表リスト（「図1.1」「表2.3」など）
- 参考文献リスト
- 配付資料一覧

# 出力形式

**目次が見つかった場合:**
```
第1章 ○○○
  1.1 ○○○
  1.2 ○○○
第2章 ○○○
  2.1 ○○○
```

**目次が見つからない場合:**
```
目次なし
```

# 制約事項
- 目次の階層構造を保持
- ページ番号は除外
- 簡潔で読みやすい形式
- 本文の章立てではなく、明確な「目次」セクションのみを抽出
            """,
        )

        chain = toc_prompt | llm
        result = chain.invoke({"text": merged_text, "pages": toc_pages})

        extracted_toc = result.content.strip()

        if extracted_toc and extracted_toc != "目次なし":
            logger.info("目次を抽出しました")
            logger.info(f"目次内容（抜粋）: {extracted_toc[:200]}...")
        else:
            logger.info("目次が見つかりませんでした")

        return {"table_of_contents": [{"type": "toc", "content": extracted_toc}]}

    def _generate_summary(self, state: WordState) -> WordState:
        """
        Generate summary from TOC or full text.

        Args:
            state: Current state with title and table_of_contents

        Returns:
            Updated state with summary field
        """
        llm = self.model.llm()
        title = state.get("title", "")
        table_of_contents = state.get("table_of_contents", [])
        pdf_pages = state["pdf_pages"]

        # Check if TOC exists
        has_toc = False
        toc_content = ""
        if table_of_contents:
            for toc_item in table_of_contents:
                if toc_item.get("type") == "toc":
                    toc_content = toc_item.get("content", "")
                    if toc_content and toc_content != "目次なし":
                        has_toc = True
                    break

        if has_toc:
            # TOC-based summarization
            logger.info("目次から要約を作成します")

            toc_summary_prompt = PromptTemplate(
                input_variables=["title", "toc"],
                template="""あなたはWord文書の要約専門家です。以下のタイトルと目次から、文書の詳細な要約を作成してください。

# 文書タイトル
「{title}」

# 目次
{toc}

# 要約作成手順

ステップ1: 文書の全体像を把握する
- 目次の構造から文書のタイプを判断（報告書、計画書、ガイドライン、議事録など）
- 章・節の構成から論理的な流れを理解
- 主要なトピックやテーマを特定

ステップ2: 要約内容を構成する
文書のタイプに応じて、以下から適切な項目を選択：

**基本項目（必須）:**
- 文書の目的・位置づけ
- 対象読者や適用範囲

**内容項目（該当するもののみ）:**
- 背景・経緯（なぜこの文書が作成されたか）
- 主要な検討事項・論点（議事録や検討資料の場合）
- 主要施策・取組内容（計画書や報告書の場合）
- 制度・仕組みの要点（ガイドラインや説明資料の場合）
- 実績・評価（実績報告の場合）

**結論項目（該当するもののみ）:**
- 結論・提言（検討資料の場合）
- 今後の予定・方向性（計画書の場合）
- 重要なポイント・留意事項（ガイドラインの場合）

ステップ3: 詳細な要約を作成する
- 目次の各章・節から推察される内容を整理
- 文書の構造と論理的な流れを重視
- 具体的な内容は推測せず、構造に基づいた要約を記述

# 出力形式
要約文のみを出力（Markdown不要、改行は適宜使用）

# 文量の目安
- 最小: 300文字程度
- 推奨: 500-1500文字（目次の詳細度に応じて調整）
- 最大: 3000文字以内

# 制約事項
- 目次にない具体的内容は推測しない
- 文書の構造と論理的な流れを重視
- 簡潔かつ分かりやすく
- 「検討」「分析」「提案」等の文書の性格を明示
                """,
            )

            chain = toc_summary_prompt | llm
            result = chain.invoke({"title": title, "toc": toc_content})
            summary = result.content.strip()

        else:
            # Full-text summarization (traditional approach)
            logger.info("目次が見つからないため、全文から要約を作成します")

            # Convert pages to documents
            documents = [
                Document(page_content=text, metadata={"page": i + 1})
                for i, text in enumerate(pdf_pages)
            ]

            # Use LangChain's summarize chain
            summary_prompt_template = """以下のWord文書の詳細な要約を作成してください。

文書タイトル: {title}

文書内容:
{{text}}

# 要約作成手順

ステップ1: 文書全体を確認する
- 文書のタイプと目的を把握
- 主要なトピックとセクションを特定
- 重要な数値・指標・固有名詞を確認

ステップ2: 詳細な要約を構成する
以下から適切な項目を選択：
- 文書の目的・背景
- 主要な内容・論点
- 重要な数値・指標・KPI
- 結論・提言・今後の方向性

# 文量の目安
- 最小: 500文字程度
- 推奨: 1000-3000文字（文書の内容に応じて調整）
- 最大: 5000文字以内

# 制約事項
- 詳細かつ分かりやすい要約
- 重要な数値・固有名詞は省略しない
- 文書に記載された内容のみを使用
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
                - table_of_contents: list[dict] | None - Extracted TOC structure
        """
        compiled = self.graph.compile()
        result = compiled.invoke(input_data)
        return result
