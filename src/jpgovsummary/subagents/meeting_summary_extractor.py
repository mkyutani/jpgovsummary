"""
MeetingSummaryExtractor sub-agent for Plan-Action architecture.

This sub-agent extracts embedded agenda and minutes content from HTML main content.
"""

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from .. import Model, logger


class MeetingSummaryExtraction(BaseModel):
    """Extracted meeting summary components."""

    agenda_content: str | None = Field(
        default=None, description="Extracted agenda section (議事次第)"
    )
    minutes_content: str | None = Field(
        default=None, description="Extracted minutes section (議事録/議事要旨)"
    )
    remaining_content: str = Field(description="Remaining content after extraction")
    has_embedded_agenda: bool = Field(default=False, description="True if agenda section was found")
    has_embedded_minutes: bool = Field(
        default=False, description="True if minutes section was found"
    )


class MeetingSummaryExtractorState(TypedDict):
    """State for MeetingSummaryExtractor sub-agent."""

    # Input
    main_content: str

    # Output
    agenda_content: str | None
    minutes_content: str | None
    remaining_content: str | None
    has_embedded_agenda: bool
    has_embedded_minutes: bool


class MeetingSummaryExtractor:
    """
    Meeting summary extraction sub-agent.

    Analyzes HTML main content to extract embedded agenda and minutes sections.
    """

    def __init__(self, model: Model | None = None):
        """
        Initialize MeetingSummaryExtractor sub-agent.

        Args:
            model: Model instance for LLM access. If None, uses default Model().
        """
        self.model = model if model is not None else Model()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build the StateGraph for meeting summary extraction.

        Returns:
            Compiled StateGraph for extraction workflow
        """
        graph = StateGraph(MeetingSummaryExtractorState)

        # Single-stage extraction
        graph.add_node("extract", self._extract_meeting_content)

        # Linear flow
        graph.set_entry_point("extract")
        graph.add_edge("extract", END)

        return graph

    def _extract_meeting_content(
        self, state: MeetingSummaryExtractorState
    ) -> MeetingSummaryExtractorState:
        """
        Extract agenda and minutes content from main content.

        Args:
            state: Current state with main_content

        Returns:
            Updated state with extracted components
        """
        llm = self.model.llm()
        main_content = state.get("main_content")

        if not main_content:
            logger.error("メインコンテンツが空のため、議事要約を抽出できません")
            return {
                "agenda_content": None,
                "minutes_content": None,
                "remaining_content": main_content or "",
                "has_embedded_agenda": False,
                "has_embedded_minutes": False,
            }

        logger.info("メインコンテンツから議事次第・議事録を抽出中...")

        parser = JsonOutputParser(pydantic_object=MeetingSummaryExtraction)

        system_prompt = SystemMessagePromptTemplate.from_template(
            """あなたは会議ページのコンテンツから議事次第と議事録を抽出する専門家です。

# 役割
HTMLメインコンテンツを分析し、以下のセクションを検出・抽出してください：
1. 議事次第（議事日程、議題、アジェンダ）
2. 議事録（議事要旨、会議録、議事概要）

# 抽出手順

ステップ1: コンテンツを分析する
- 見出しやキーワードから議事次第セクションを特定
  - キーワード例: 「議事次第」「議事日程」「議題」「アジェンダ」
- 見出しやキーワードから議事録セクションを特定
  - キーワード例: 「議事録」「議事要旨」「会議録」「議事概要」

ステップ2: 各セクションを抽出する
- 議事次第セクション:
  - 見出しからセクション終了まで（次の見出しまたはコンテンツ終了）
  - 会議の議題リスト、スケジュール、議事項目を含む
  - 議題の説明や背景情報も含める

- 議事録セクション:
  - 見出しからセクション終了まで（次の見出しまたはコンテンツ終了）
  - 発言内容、議論の要約、決定事項を含む
  - 質疑応答の記録も含める

ステップ3: 残りのコンテンツを保持する
- 抽出されなかった部分を remaining_content に格納
- 会議概要、開催情報など、議事次第・議事録以外のコンテンツ

# 検出条件

議事次第として検出する条件:
- 明確な「議事次第」「議事日程」「議題」などの見出しがある
- 議題リスト（箇条書きまたは番号付き）がある
- 会議の流れや議論項目が構造化されている

議事録として検出する条件:
- 明確な「議事録」「議事要旨」「会議録」などの見出しがある
- 実際の発言や議論内容が記録されている
- 決定事項や合意内容が記述されている

# 検出しないケース

以下は議事次第・議事録として検出しない:
- 単なるリンクリスト（「議事次第はこちら」というリンクのみ）
- 資料リスト（「配付資料1: ○○.pdf」のような列挙のみ）
- 開催案内（日時・場所・出席者のみ）
- 参加者名簿
- 座席表

# 出力形式
{format_instructions}

# 重要な注意事項
- 見出しも含めて抽出（マークダウン形式を保持）
- セクションが見つからない場合はnullを返す
- 必ずremaining_contentに残りのコンテンツを格納
- has_embedded_*フラグは該当セクションの有無を示す
            """
        )

        assistant_prompt = AIMessagePromptTemplate.from_template(
            """以下のメインコンテンツから議事次第と議事録を抽出してください。

# メインコンテンツ
{main_content}

# 処理手順
1. 議事次第セクションを検出・抽出
2. 議事録セクションを検出・抽出
3. 残りのコンテンツを保持
4. フラグを設定

# 出力フォーマット
{format_instructions}
            """
        )

        prompt = ChatPromptTemplate.from_messages([system_prompt, assistant_prompt])

        chain = prompt | llm | parser

        try:
            result = chain.invoke(
                {
                    "main_content": main_content,
                    "format_instructions": parser.get_format_instructions(),
                }
            )

            agenda_content = result.get("agenda_content")
            minutes_content = result.get("minutes_content")
            remaining_content = result.get("remaining_content", main_content)
            has_embedded_agenda = result.get("has_embedded_agenda", False)
            has_embedded_minutes = result.get("has_embedded_minutes", False)

            if has_embedded_agenda:
                logger.info(
                    f"✅ 議事次第セクションを抽出しました ({len(agenda_content or '')}文字)"
                )
            else:
                logger.info("議事次第セクションは見つかりませんでした")

            if has_embedded_minutes:
                logger.info(f"✅ 議事録セクションを抽出しました ({len(minutes_content or '')}文字)")
            else:
                logger.info("議事録セクションは見つかりませんでした")

            return {
                "agenda_content": agenda_content,
                "minutes_content": minutes_content,
                "remaining_content": remaining_content,
                "has_embedded_agenda": has_embedded_agenda,
                "has_embedded_minutes": has_embedded_minutes,
            }

        except Exception as e:
            logger.error(f"議事要約抽出中にエラー: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return {
                "agenda_content": None,
                "minutes_content": None,
                "remaining_content": main_content,
                "has_embedded_agenda": False,
                "has_embedded_minutes": False,
            }

    def invoke(self, input_data: dict) -> dict:
        """
        Execute meeting summary extraction.

        Args:
            input_data: Dict with keys:
                - main_content: str - HTML main content

        Returns:
            Dict with keys:
                - agenda_content: str | None - Extracted agenda section
                - minutes_content: str | None - Extracted minutes section
                - remaining_content: str - Remaining content
                - has_embedded_agenda: bool - Flag if agenda found
                - has_embedded_minutes: bool - Flag if minutes found
        """
        compiled = self.graph.compile()
        result = compiled.invoke(input_data)
        return result
