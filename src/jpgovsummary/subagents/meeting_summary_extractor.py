"""
MeetingSummaryExtractor sub-agent for Plan-Action architecture.

This sub-agent extracts embedded agenda and minutes content from HTML main content.
"""

from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from .. import Model, logger


class MeetingSummaryExtractorState(TypedDict):
    """State for MeetingSummaryExtractor sub-agent."""

    # Input
    main_content: str

    # Output
    structured_summary: str | None
    has_meeting_info: bool


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
        Extract structured meeting summary from main content.

        Args:
            state: Current state with main_content

        Returns:
            Updated state with structured summary
        """
        llm = self.model.llm()
        main_content = state.get("main_content")

        if not main_content:
            logger.error("メインコンテンツが空のため、会議概要を抽出できません")
            return {
                "structured_summary": None,
                "has_meeting_info": False,
            }

        logger.info("メインコンテンツから構造化会議概要を抽出中...")

        system_prompt = SystemMessagePromptTemplate.from_template(
            """あなたは会議ページのコンテンツから構造化された会議概要を抽出する専門家です。

# 役割
HTMLメインコンテンツを分析し、以下の構造化されたMarkdown形式で会議概要を作成してください。

# 出力形式

以下のMarkdown構造で出力してください：

```markdown
## 会議概要

### プロフィール
- 会議名、回数: [会議名と第X回を記載。見つからない場合は「不明」]
- 日時、場所: [開催日時と場所を記載。見つからない場合は「不明」]
- 議題一覧: [議題を箇条書きで記載。見つからない場合は「記載なし」]
- 配布資料一覧: [配布資料名を箇条書きで記載。見つからない場合は「記載なし」]
- 出席者一覧: [出席者名を箇条書きまたは人数のみで記載。見つからない場合は「記載なし」]

### 議事
- 会議の目的、議題: [会議の目的や主要議題を記載。見つからない場合は「記載なし」]
- 主要な議論内容: [主な議論や発言内容を記載。見つからない場合は「記載なし」]
- 決定事項の要約: [決定事項や合意内容を記載。見つからない場合は「記載なし」]
- 次回予定: [次回開催予定を記載。見つからない場合は「記載なし」]
```

# 抽出手順

ステップ1: プロフィール情報を抽出
- 会議名と回数を特定（例: 「第3回 デジタル社会推進会議」）
- 開催日時と場所を特定
- 議題リストを抽出
- 配布資料リストを抽出（資料名のみ、PDFリンクは含めない）
- 出席者情報を抽出（名簿がある場合は名前、ない場合は人数のみ）

ステップ2: 議事情報を抽出
- 会議の目的や主要議題を特定
- 議論内容や主要な発言を抽出
- 決定事項や合意内容を特定
- 次回予定を確認

ステップ3: 構造化されたMarkdownを生成
- 上記の形式に従って整形
- 情報が見つからない項目は「不明」または「記載なし」と明記
- 箇条書きは適切にインデントする

# 重要な注意事項
- メインコンテンツに記載されている情報のみを使用（推測・創作はしない）
- 情報が見つからない場合は正直に「不明」「記載なし」と記載
- ファイルサイズ、ソフトウェア案内などの技術情報は除外
- 見出し構造（##、###）を必ず守る
- 単なるリンクリストページの場合でも、利用可能な情報は可能な限り抽出
            """
        )

        assistant_prompt = AIMessagePromptTemplate.from_template(
            """以下のメインコンテンツから構造化された会議概要を作成してください。

# メインコンテンツ
{main_content}

# 指示
上記のMarkdown形式に従って、会議概要を作成してください。
コンテンツ内に該当情報がない項目は「不明」または「記載なし」と明記してください。
            """
        )

        prompt = ChatPromptTemplate.from_messages([system_prompt, assistant_prompt])

        chain = prompt | llm

        try:
            result = chain.invoke({"main_content": main_content})

            structured_summary = result.content.strip()

            # Check if meaningful meeting information was found
            has_meeting_info = (
                "不明" not in structured_summary or
                "記載なし" not in structured_summary or
                len(structured_summary) > 200
            )

            logger.info(f"✅ 構造化会議概要を抽出しました ({len(structured_summary)}文字)")
            logger.info("-" * 64)
            logger.info(f"構造化会議概要:\n{structured_summary}")
            logger.info("-" * 64)

            return {
                "structured_summary": structured_summary,
                "has_meeting_info": has_meeting_info,
            }

        except Exception as e:
            logger.error(f"会議概要抽出中にエラー: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return {
                "structured_summary": None,
                "has_meeting_info": False,
            }

    def invoke(self, input_data: dict) -> dict:
        """
        Execute meeting summary extraction.

        Args:
            input_data: Dict with keys:
                - main_content: str - HTML main content

        Returns:
            Dict with keys:
                - structured_summary: str | None - Structured meeting summary in markdown
                - has_meeting_info: bool - Flag if meaningful meeting info was found
        """
        compiled = self.graph.compile()
        result = compiled.invoke(input_data)
        return result
