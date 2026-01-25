"""
HTMLProcessor sub-agent for Plan-Action architecture.

This sub-agent handles HTML loading, markdown conversion, main content extraction,
and related document discovery with isolated context.
"""

import urllib.parse

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langgraph.graph import END, StateGraph

from .. import CandidateReportList, Model, logger
from ..state_v2 import HTMLProcessorState
from ..tools import load_html_as_markdown


class HTMLProcessor:
    """
    HTML processing sub-agent.

    Loads HTML pages, converts to markdown, extracts main content
    (removing headers/footers/navigation), and discovers related documents.
    """

    def __init__(self, model: Model | None = None):
        """
        Initialize HTMLProcessor sub-agent.

        Args:
            model: Model instance for LLM access. If None, uses default Model().
        """
        self.model = model if model is not None else Model()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build the StateGraph for HTML processing.

        Returns:
            Compiled StateGraph for HTML workflow
        """
        graph = StateGraph(HTMLProcessorState)

        # Three-stage pipeline
        graph.add_node("load_html", self._load_html)
        graph.add_node("extract_main_content", self._extract_main_content)
        graph.add_node("discover_documents", self._discover_documents)

        # Linear flow
        graph.set_entry_point("load_html")
        graph.add_edge("load_html", "extract_main_content")
        graph.add_edge("extract_main_content", "discover_documents")
        graph.add_edge("discover_documents", END)

        return graph

    def _load_html(self, state: HTMLProcessorState) -> HTMLProcessorState:
        """
        Load HTML page and convert to markdown.

        Args:
            state: Current state with url

        Returns:
            Updated state with markdown field
        """
        url = state["url"]

        logger.info(f"HTMLをロード中: {url}")

        try:
            markdown_content = load_html_as_markdown(url)
            logger.info(f"HTML→Markdown変換完了 ({len(markdown_content)}文字)")

            return {"markdown": markdown_content}

        except Exception as e:
            logger.error(f"HTML読み込みエラー: {e}")
            return {"markdown": None}

    def _extract_main_content(self, state: HTMLProcessorState) -> HTMLProcessorState:
        """
        Extract main content from markdown (remove headers, footers, navigation).

        Args:
            state: Current state with markdown

        Returns:
            Updated state with main_content field
        """
        llm = self.model.llm()
        markdown = state.get("markdown")
        url = state["url"]

        if not markdown:
            logger.error("Markdownが空のため、メインコンテンツを抽出できません")
            return {"main_content": None}

        logger.info("メインコンテンツを抽出中...")

        system_prompt = SystemMessagePromptTemplate.from_template(
            """あなたはマークダウンからメインコンテンツを抽出する専門家です。

# 役割
Webページのマークダウンを分析し、ヘッダー・フッター・ナビゲーションを除去して、
本質的な内容（会議情報、報告書、お知らせなど）のみを抽出してください。

# 抽出手順

ステップ1: ページ構造を分析する
- ヘッダー（上部ナビゲーション、ロゴ、検索ボックス）を特定
- フッター（著作権、プライバシーポリシー、お問い合わせ）を特定
- サイドバー（関連リンク、メニュー）を特定
- メインコンテンツ（会議情報、報告書、資料リスト）を特定

ステップ2: 不要な部分を除去する
以下を削除：
- ヘッダー、フッター、ナビゲーション
- パンくずリスト
- 広告、バナー、通知
- サイトマップ、プライバシーポリシーへのリンク
- 「ページの先頭へ」などのナビゲーション要素

ステップ3: 重要な部分を保持する
以下を保持：
- 会議・報告書・とりまとめの概要
- 議題・議事録・決定事項
- 資料リスト（配付資料、参考資料）
- 日時・場所・出席者などの会議情報
- お知らせ・募集の本文

# 出力形式
抽出したメインコンテンツをマークダウン形式で出力してください。
構造（見出し、リスト、表、リンク）は保持してください。

# エラー処理
メインコンテンツが抽出できない場合は、必ず「[HTML_PARSING_ERROR]」と出力してください。
            """
        )

        assistant_prompt = AIMessagePromptTemplate.from_template(
            """マークダウンからメインコンテンツを抽出してください。

# 制約事項
- ヘッダー、フッター、ナビゲーションは除去
- メインコンテンツの構造は保持
- リンク（メインコンテンツ内）は保持
- 抽出できない場合は「[HTML_PARSING_ERROR]」
            """
        )

        prompt = ChatPromptTemplate.from_messages(
            [system_prompt, assistant_prompt, MessagesPlaceholder(variable_name="messages")]
        )

        chain = prompt | llm

        # Create messages
        messages = [
            HumanMessage(content=f'会議のURLは"{url}"です。'),
            HumanMessage(content=f"マークダウンは以下の通りです：\n\n{markdown}"),
        ]

        result = chain.invoke({"messages": messages})

        # Check for HTML parsing error
        if "[HTML_PARSING_ERROR]" in result.content:
            logger.warning("⚠️ HTMLパースエラーが検出されました。lxmlで自動修正を試みます...")

            try:
                # Retry with lxml normalization
                normalized_markdown = load_html_as_markdown(url)
                logger.info("🔧 HTMLを正規化して再変換しました")

                # Retry extraction
                retry_messages = [
                    HumanMessage(content=f'会議のURLは"{url}"です。'),
                    HumanMessage(content=f"マークダウンは以下の通りです：\n\n{normalized_markdown}"),
                ]

                retry_result = chain.invoke({"messages": retry_messages})

                if "[HTML_PARSING_ERROR]" not in retry_result.content:
                    logger.info("✅ HTML正規化後にメインコンテンツの抽出に成功しました")
                    result = retry_result
                else:
                    logger.error("❌ HTML正規化後もメインコンテンツの抽出に失敗しました")
                    return {"main_content": None}

            except Exception as e:
                logger.error(f"❌ HTML自動修正中にエラーが発生しました: {e}")
                return {"main_content": None}

        main_content = result.content.strip()
        logger.info(f"メインコンテンツ抽出完了 ({len(main_content)}文字)")

        return {"main_content": main_content}

    def _discover_documents(self, state: HTMLProcessorState) -> HTMLProcessorState:
        """
        Discover related document URLs from main content.

        Args:
            state: Current state with main_content

        Returns:
            Updated state with discovered_documents field
        """
        llm = self.model.llm()
        main_content = state.get("main_content")
        url = state["url"]

        if not main_content:
            logger.error("メインコンテンツが空のため、関連資料を発見できません")
            return {"discovered_documents": []}

        logger.info("関連資料を発見中...")

        parser = JsonOutputParser(pydantic_object=CandidateReportList)

        system_prompt = SystemMessagePromptTemplate.from_template(
            """あなたはマークダウンから関連資料のリンクを抽出する専門家です。

# 役割
会議ページのメインコンテンツから、要約対象となる関連資料（PDF、Word文書など）を
正確に特定し、不要なリンク（ナビゲーション、外部サイトなど）を除外してください。

# 判定手順

ステップ1: すべてのリンクを抽出する
- マークダウン内のすべてのリンクを漏れなく抽出
- リンク先URLとリンクテキストを取得

ステップ2: 各リンクを判定する
以下の5つの基準で順番に確認：

**基準1: 関連資料か？**
✅ 以下は関連資料：
- 会議の議事録、報告書、配付資料
- とりまとめの本文・概要
- 構成員一覧、目次、索引
- 案内・お知らせ・募集の本文

❌ 以下は関連資料ではない：
- プライバシーポリシー、サイトマップ
- YouTube、動画ファイル（mp4、avi）
- NDL Warp（国立国会図書館）
- 一般的な案内・お知らせ

**基準2: 会議資料・補足資料か？**
- 会議で使用された資料
- 参考資料、追加資料

**基準3: ナビゲーション要素ではないか？**
- ヘッダー、フッター、メニュー
- パンくずリスト
- サイドバーのリンク

**基準4: 相対パスの処理**
- 相対パスは絶対URLに変換
- ベースURL: {url}

ステップ3: 出力
すべてのリンクについて以下を記述：
- URL（絶対パス）
- リンクテキスト
- 判定結果（true/false）
- 判断理由（具体的に）

# 制約事項
- すべてのリンクを漏れなく出力
- 判定理由は具体的に記述
- 不確かな場合は厳密に判断
            """
        )

        assistant_prompt = AIMessagePromptTemplate.from_template(
            """以下のメインコンテンツから関連資料のリンクを抽出してください。

# メインコンテンツ
{main_content}

# 処理手順
1. すべてのリンクを抽出
2. 5つの基準で判定（関連資料、会議資料、ナビゲーション、相対パス）
3. 判定結果と理由を出力

# 出力フォーマット
{format_instructions}
            """
        )

        prompt = ChatPromptTemplate.from_messages([system_prompt, assistant_prompt])

        chain = prompt | llm | parser

        try:
            result = chain.invoke({
                "url": url,
                "main_content": main_content,
                "format_instructions": parser.get_format_instructions()
            })

            # Extract document URLs (only those marked as related)
            discovered_urls = []
            if hasattr(result, "reports"):
                for report in result.reports:
                    if report.is_related_document:
                        # Convert relative URLs to absolute
                        absolute_url = urllib.parse.urljoin(url, report.url)
                        discovered_urls.append(absolute_url)

            logger.info(f"関連資料を{len(discovered_urls)}件発見しました")

            return {"discovered_documents": discovered_urls}

        except Exception as e:
            logger.error(f"関連資料発見中にエラー: {e}")
            return {"discovered_documents": []}

    def invoke(self, input_data: dict) -> dict:
        """
        Execute HTML processing.

        Args:
            input_data: Dict with keys:
                - url: str - HTML page URL

        Returns:
            Dict with keys:
                - markdown: str | None - Converted markdown
                - main_content: str | None - Extracted main content
                - discovered_documents: list[str] - URLs of related documents
        """
        compiled = self.graph.compile()
        result = compiled.invoke(input_data)
        return result
