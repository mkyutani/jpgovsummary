"""
DocumentTypeDetector sub-agent for Plan-Action architecture.

This sub-agent handles document type classification with isolated context,
reducing token consumption by avoiding main workflow state pollution.
"""

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from .. import Model, logger
from ..state import DocumentTypeDetectorState


class CategoryAnalysis(BaseModel):
    """Analysis result for a single document category."""

    score: int = Field(description="重要度スコア（1-5点）", ge=1, le=5)
    reason: str = Field(description="スコアの理由")
    evidence: str = Field(description="根拠テキスト例")


class DocumentTypeAnalysis(BaseModel):
    """Complete analysis of document type across all categories."""

    word: CategoryAnalysis = Field(description="Word文書の分析")
    powerpoint: CategoryAnalysis = Field(description="PowerPoint文書の分析")
    other: CategoryAnalysis = Field(description="その他の分析")
    conclusion: str = Field(description="最も可能性が高いと判断される形式")


class DocumentTypeDetector:
    """
    Document type classification sub-agent.

    Analyzes PDF pages to determine document type (PowerPoint, Word, Agenda, etc.)
    with isolated context to avoid token overhead from main workflow state.
    """

    # Maximum pages to analyze for type detection
    MAX_PAGES_TO_ANALYZE = 10

    # Category mapping
    CATEGORY_MAPPING = {
        "word": "word",
        "powerpoint": "powerpoint",
        "other": "other",
    }

    def __init__(self, model: Model | None = None):
        """
        Initialize DocumentTypeDetector sub-agent.

        Args:
            model: Model instance for LLM access. If None, uses default Model().
        """
        self.model = model if model is not None else Model()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build the StateGraph for document type detection.

        Returns:
            Compiled StateGraph for type detection workflow
        """
        graph = StateGraph(DocumentTypeDetectorState)

        # Single node: detect type with one LLM call
        graph.add_node("detect_type", self._detect_type)

        # Linear flow
        graph.set_entry_point("detect_type")
        graph.add_edge("detect_type", END)

        return graph

    def _detect_type(self, state: DocumentTypeDetectorState) -> DocumentTypeDetectorState:
        """
        Detect document type using LLM analysis.

        Args:
            state: Current state with pdf_pages

        Returns:
            Updated state with document_type, confidence_scores, detection_detail
        """
        llm = self.model.llm()
        pdf_pages = state["pdf_pages"]
        display_name = state.get("display_name", "")
        log_prefix = f"[{display_name}] " if display_name else ""

        # Analyze first N pages
        pages_to_analyze = min(self.MAX_PAGES_TO_ANALYZE, len(pdf_pages))
        sample_texts = pdf_pages[:pages_to_analyze]

        logger.info(f"{log_prefix}先頭{len(sample_texts)}ページを分析して文書タイプを判定します")

        # Prepare text for analysis
        if pages_to_analyze == 1:
            merged_text = f"ページ1:\n{sample_texts[0]}"
        else:
            merged_text = "\n\n".join(
                [f"ページ{i+1}:\n{text}" for i, text in enumerate(sample_texts)]
            )

        # Create parser
        parser = PydanticOutputParser(pydantic_object=DocumentTypeAnalysis)

        # Modern prompt with step-by-step instructions
        detection_prompt = PromptTemplate(
            input_variables=["text", "total_pages", "pages_count", "format_instructions"],
            template="""あなたは政府文書の分類専門家です。PDFから抽出されたテキストを分析し、文書タイプを7つのカテゴリーから判定してください。

# 役割
政府のPDF文書を正確に分類する専門家として、文書の構造的・内容的特徴を分析し、最適なカテゴリーを判定してください。

# 判定カテゴリー（3分類）

**1. Word文書 (word)**
文章的特徴：
- 完全な文章構造（主語・述語が明確）
- 段落構成で論理的展開
- 詳細な説明・背景・経緯を含む
- 議事録、報告書、ガイドライン、仕様書、とりまとめ、議事次第など

**2. PowerPoint資料 (powerpoint)**
構造的特徴：
- 各ページに明確なスライドタイトル
- 箇条書き中心（●・○マーク多用）
- 体言止め表現が多い（「〜について」「〜の検討」）
- 1ページ完結型の構成
- プレゼン資料、説明資料、概要資料

**3. その他 (other)**
- 上記2分類に該当しない文書
- 参加者一覧、ニュース・お知らせ、調査・アンケート結果など

# 判定手順

ステップ1: 文書の基本構造を確認する
- ページタイトルの有無と形式
- 箇条書きと文章の割合
- 表形式データの有無と割合

ステップ2: 各カテゴリーの特徴を評価する
以下の3カテゴリーについて、それぞれ1-5点でスコアリング：
- 5点: 該当カテゴリーの特徴が明確に多数確認できる
- 4点: 該当カテゴリーの特徴がいくつか確認できる
- 3点: 該当カテゴリーの特徴が部分的に見られる
- 2点: 該当カテゴリーの特徴がわずかに見られる
- 1点: 該当カテゴリーの特徴がほとんど見られない

各カテゴリーについて：
- スコア（1-5の数値）
- 理由（そのスコアをつけた具体的理由）
- 根拠テキスト例（文書から実際に引用）

ステップ3: 最終判定を行う
- 最高スコアのカテゴリーを選択
- PowerPoint vs Word の場合は以下を重視：
  * 箇条書きの割合（70%以上→PowerPoint）
  * 完全文の連続（3行以上→Word）
  * ページタイトルの明確さ（明確→PowerPoint）
  * 体言止めの頻度（多い→PowerPoint）

# 分析対象
総ページ数: {total_pages}ページ
分析対象: 最初の{pages_count}ページ

PDFテキスト:
{text}

# 出力形式
以下のJSON形式で出力してください：

{format_instructions}

# 重要な制約
- 必ず全3カテゴリーにスコア・理由・根拠を記載
- 複数カテゴリーが同じ高スコアにならないよう注意
- 根拠テキスト例は文書から実際に引用
- 結論（conclusion）で最も可能性が高いカテゴリー名を明記（word/powerpoint/other のいずれか）
            """,
        )

        # Invoke LLM
        chain = detection_prompt | llm | parser
        result = chain.invoke(
            {
                "text": merged_text,
                "total_pages": len(pdf_pages),
                "pages_count": pages_to_analyze,
                "format_instructions": parser.get_format_instructions(),
            }
        )

        # Extract scores and reasoning
        scores = {
            "word": result.word.score,
            "powerpoint": result.powerpoint.score,
            "other": result.other.score,
        }

        reasoning = {
            "word": result.word.reason,
            "powerpoint": result.powerpoint.reason,
            "other": result.other.reason,
        }

        evidence = {
            "word": result.word.evidence,
            "powerpoint": result.powerpoint.evidence,
            "other": result.other.evidence,
        }

        # Determine document type from scores
        doc_type = None
        doc_reason = ""
        selected_evidence = ""

        if scores:
            # Find highest scoring category
            max_score = max(scores.values())
            max_categories = [cat for cat, score in scores.items() if score == max_score]

            if max_categories:
                top_category = max_categories[0]

                # Map category name to doc_type
                for doc_type_key, category_name in self.CATEGORY_MAPPING.items():
                    if category_name in top_category:
                        doc_type = doc_type_key
                        doc_reason = reasoning.get(category_name, "不明")
                        selected_evidence = evidence.get(category_name, "不明")
                        break

        # Fallback: use conclusion text
        if doc_type is None:
            conclusion_lower = result.conclusion.lower()
            for doc_type_key in self.CATEGORY_MAPPING.keys():
                if doc_type_key in conclusion_lower:
                    doc_type = doc_type_key
                    doc_reason = "結論から判定"
                    selected_evidence = "なし"
                    break

        # Final fallback
        if doc_type is None:
            doc_type = "other"
            doc_reason = "判定不能"
            selected_evidence = "なし"

        # Log results
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        logger.info(
            f"{log_prefix}この文書を{doc_type}と推定しました({', '.join([f'{cat}:{score}' for cat, score in sorted_scores])})"
        )
        logger.info(f"{log_prefix}推定理由: {doc_reason.replace('\n', '\\n')}")
        logger.info(f"{log_prefix}根拠: {selected_evidence.replace('\n', '\\n')}")

        # Prepare confidence scores (normalized to 0-1)
        max_possible_score = 5.0
        confidence_scores = {
            category.lower(): score / max_possible_score for category, score in scores.items()
        }

        # Prepare detection detail
        detection_detail = f"""判定結果: {doc_type}
理由: {doc_reason}
根拠: {selected_evidence}

スコア詳細:
{chr(10).join([f'  {cat}: {score}/5 - {reasoning[cat][:50]}...' for cat, score in sorted_scores[:3]])}

総ページ数: {len(pdf_pages)}ページ
分析対象: 先頭{pages_to_analyze}ページ
"""

        return {
            "document_type": doc_type,
            "confidence_scores": confidence_scores,
            "detection_detail": detection_detail,
        }

    def invoke(self, input_data: dict) -> dict:
        """
        Execute document type detection.

        Args:
            input_data: Dict with keys:
                - pdf_pages: list[str] - PDF page texts
                - url: str - Source URL (optional)

        Returns:
            Dict with keys:
                - document_type: str - Detected type (word/powerpoint/agenda/etc)
                - confidence_scores: dict[str, float] - Confidence for each type
                - detection_detail: str - Detailed explanation
        """
        compiled = self.graph.compile()
        result = compiled.invoke(input_data)
        return result
