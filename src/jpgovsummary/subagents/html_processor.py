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

from .. import Model, logger
from ..state_v2 import DiscoveredDocument, DiscoveredDocumentList, HTMLProcessorState
from ..tools import load_html_as_markdown
from .meeting_summary_extractor import MeetingSummaryExtractor


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
        self.meeting_summary_extractor = MeetingSummaryExtractor(model=self.model)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build the StateGraph for HTML processing.

        Returns:
            Compiled StateGraph for HTML workflow
        """
        graph = StateGraph(HTMLProcessorState)

        # Three-stage pipeline (no meeting summary extraction in Phase 1)
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
                    HumanMessage(
                        content=f"マークダウンは以下の通りです：\n\n{normalized_markdown}"
                    ),
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
        logger.info("-" * 64)
        logger.info(f"メインコンテンツ： {main_content.replace(chr(10), '\\n')}")
        logger.info("-" * 64)

        return {"main_content": main_content}

    def _extract_meeting_summary(self, state: HTMLProcessorState) -> HTMLProcessorState:
        """
        Extract structured meeting summary from main content.

        Args:
            state: Current state with main_content

        Returns:
            Updated state with structured_overview and has_meeting_info flag
        """
        main_content = state.get("main_content")

        if not main_content:
            logger.info("メインコンテンツが空のため、会議概要抽出をスキップします")
            return {
                "structured_overview": None,
                "has_meeting_info": False,
            }

        logger.info("構造化会議概要の抽出を開始...")

        try:
            extraction_result = self.meeting_summary_extractor.invoke(
                {"main_content": main_content}
            )

            structured_overview = extraction_result.get("structured_overview")
            has_meeting_info = extraction_result.get("has_meeting_info", False)

            if has_meeting_info:
                logger.info("✅ 構造化会議概要の抽出が完了しました")
            else:
                logger.info("会議情報が見つかりませんでした")

            return {
                "structured_overview": structured_overview,
                "has_meeting_info": has_meeting_info,
            }

        except Exception as e:
            logger.error(f"会議概要抽出中にエラー: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return {
                "structured_overview": None,
                "has_meeting_info": False,
            }

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

        parser = JsonOutputParser(pydantic_object=DiscoveredDocumentList)

        system_prompt = SystemMessagePromptTemplate.from_template(
            """あなたはマークダウンから関連資料のリンクを抽出する専門家です。

# 役割
会議ページのメインコンテンツから、要約対象となる関連資料(PDFのみ)を正確に特定てください。

# 判定手順

ステップ1: すべてのPDFリンクを抽出する
- マークダウン内のすべてのリンクを漏れなく抽出
- リンク先URLとリンクテキストを取得

ステップ2: リンクテキストのクリーニング
- リンクテキストから以下を除去して、クリーンな文書名のみを抽出：
  - ファイル形式表記（【PDF形式】、【Excel形式】、【Word形式】など）
  - ファイルサイズ表記（［56KB］、［1.2MB］、［200KB］など）
  - その他の装飾記号や括弧内の形式情報
- 例：「【PDF形式】［56KB］ (議事次第)」→「議事次第」
- 例：「資料1【PDF形式】」→「資料1」
- 例：「【PDF形式】［1.2MB］ 委員名簿」→「委員名簿」

ステップ3: 各リンクのカテゴリを判定する

- 以下のいずれかに分類する
    - `agenda`: 議事次第
    - `minutes`: 議事録、議事要旨
    - `executive_summary`: とりまとめ、概要、Executive Summary
    - `material`: 資料X、資料X-X
    - `reference`: 参考資料
    - `participants`: 委員名簿、出席者名簿
    - `seating`: 座席表
    - `personal_material`: 個人名・団体名を含む資料
    - `announcement`: プレスリリース、ニュース、報道発表
    - `other`: その他

ステップ4: 相対パスを絶対パスに変換
- 相対パスは絶対URLに変換
- ベースURL: {url}

ステップ5: 出力
すべてのリンクについて以下を記述：
- URL（絶対パス）
- クリーンな文書名（形式情報を除去したもの）
- カテゴリ
- 判断理由（具体的に）

# 制約事項
- すべてのリンクを漏れなく出力
- 文書名は必ずクリーニングすること（形式情報を含めない）
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
2. リンクテキストをクリーニング（ファイル形式・サイズ表記を除去）
3. 各リンクが関連資料か判定し、カテゴリを決定
4. 相対パスを絶対パスに変換
5. 全リンクについて、URL、クリーンな文書名、カテゴリを出力

# 重要な注意事項
- 文書名には「【PDF形式】」「［56KB］」などの形式情報を含めないこと
- 括弧内の実際の文書名のみを抽出すること

# 出力フォーマット
{format_instructions}
            """
        )

        prompt = ChatPromptTemplate.from_messages([system_prompt, assistant_prompt])

        chain = prompt | llm | parser

        try:
            result = chain.invoke(
                {
                    "url": url,
                    "main_content": main_content,
                    "format_instructions": parser.get_format_instructions(),
                }
            )

            # Extract discovered documents and convert relative URLs to absolute
            discovered_documents = []
            if "documents" in result:
                for doc_dict in result["documents"]:
                    # Convert relative URLs to absolute
                    absolute_url = urllib.parse.urljoin(url, doc_dict["url"])
                    discovered_documents.append(
                        DiscoveredDocument(
                            url=absolute_url, name=doc_dict["name"], category=doc_dict["category"]
                        )
                    )

            logger.info(f"関連資料を{len(discovered_documents)}件発見しました")
            for doc in discovered_documents:
                logger.info(f"  - [{doc.category}] {doc.name}: {doc.url}")

            return {"discovered_documents": discovered_documents}

        except Exception as e:
            logger.error(f"関連資料発見中にエラー: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return {"discovered_documents": []}

    def invoke(self, input_data: dict) -> dict:
        """
        Execute HTML processing (Phase 1 - no LLM summary generation).

        Args:
            input_data: Dict with keys:
                - url: str - HTML page URL

        Returns:
            Dict with keys:
                - markdown: str | None - Converted markdown
                - main_content: str | None - Extracted main content
                - discovered_documents: list[DiscoveredDocument] - Discovered related documents
        """
        compiled = self.graph.compile()
        result = compiled.invoke(input_data)
        return result
