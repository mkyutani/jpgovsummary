"""
V1 workflow for jpgovsummary using LangGraph state machine.

This is the legacy workflow that uses a monolithic State-based approach
with message histories. Use --use-v1 flag to enable this workflow.
"""

import sys

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .. import Config, Model
from ..tools import load_html_as_markdown
from . import Report, State, TargetReportList
from .agents import (
    bluesky_poster,
    document_summarizer,
    main_content_extractor,
    overview_generator,
    report_enumerator,
    report_selector,
    summary_finalizer,
    summary_integrator,
)


def should_continue_from_main_content_extractor(state: State) -> str:
    """
    main_content_extractorの後の条件分岐
    メインコンテンツの抽出に失敗した場合はエラー終了
    """
    main_content = state.get("main_content", "")

    # HTMLパースエラーが検出された場合は終了
    if "[HTML_PARSING_ERROR]" in main_content:
        return END

    # 正常な場合は overview_generator へ
    return "overview_generator"


def should_process_additional_files(state: State) -> str:
    """
    overview_generatorの後の条件分岐
    議事録が検出された場合やoverview-onlyモードの場合は追加ファイル処理をスキップ
    """
    # overview-onlyモードまたは議事録検出時は直接summary_finalizerへ
    if state.get("overview_only", False) or state.get("meeting_minutes_detected", False):
        return "summary_finalizer"

    # 通常の場合は report_enumerator へ
    return "report_enumerator"


def should_continue_target_reports(state: State) -> str | bool:
    """
    target_reports関連の条件分岐
    """
    # target_reportsが存在しない場合もsummary_integratorへ遷移
    # overviewをfinal_summaryとして使用
    if "target_reports" not in state:
        return "summary_integrator"

    # target_report_indexがなければ0で初期化
    if "target_report_index" not in state:
        state["target_report_index"] = 0
        return "document_summarizer"

    # まだ要約が完了していない資料がある場合は再度document_summarizerへ
    if state["target_report_index"] < len(state["target_reports"]):
        return "document_summarizer"

    # すべての資料の要約が完了した場合
    # target_report_summariesがある場合のみsummary_integratorへ
    state.get("target_report_summaries", [])

    # すべての資料の要約が完了した場合はsummary_integratorへ遷移
    # summary_integratorで有効な要約がない場合のハンドリングを行う
    return "summary_integrator"


def should_post_to_bluesky(state: State) -> str:
    """
    summary_finalizerの後にBluesky投稿を行うかどうかを判定

    以下の場合は投稿をスキップ:
    - skip_bluesky_postingフラグが立っている
    - final_review_summaryまたはfinal_summaryにエラーメッセージが含まれている
    """
    # skip_bluesky_postingフラグのチェック
    if state.get("skip_bluesky_posting", False):
        return END

    # 最終要約を取得
    final_summary = state.get("final_review_summary") or state.get("final_summary", "")

    # エラーメッセージが含まれている場合は投稿しない
    error_messages = [
        "文書の要約がないため要約を統合できませんでした",
        "要約の統合中にエラーが発生しました"
    ]
    if any(error_msg in final_summary for error_msg in error_messages):
        return END

    # 正常な要約がある場合のみBluesky投稿へ
    return "bluesky_poster"


def run_v1(
    url: str,
    page_type: str,
    model: Model | None = None,
    batch: bool = False,
    skip_bluesky_posting: bool = False,
    overview_only: bool = False,
) -> dict:
    """
    Run the v1 jpgovwatcher workflow.

    Args:
        url: URL or file path to process
        page_type: Type of page ("html" or "pdf")
        model: Model instance (optional, uses default if not provided)
        batch: Run in batch mode without human interaction
        skip_bluesky_posting: Skip Bluesky posting step
        overview_only: Generate overview only without processing additional documents

    Returns:
        dict with keys:
            - success: bool
            - summary: str (final summary if successful)
            - url: str
            - error: str (error message if failed)
    """
    # Initialize the default model if not provided
    if model is None:
        Model()

    config = Config(1).get()
    graph = StateGraph(State)

    # Add agent nodes
    graph.add_node("main_content_extractor", main_content_extractor)
    graph.add_node("overview_generator", overview_generator)
    graph.add_node("report_enumerator", report_enumerator)
    graph.add_node("report_selector", report_selector)
    graph.add_node("document_summarizer", document_summarizer)
    graph.add_node("summary_integrator", summary_integrator)
    graph.add_node("summary_finalizer", summary_finalizer)
    graph.add_node("bluesky_poster", bluesky_poster)

    # Define graph edges based on page type
    if page_type == "html":
        # Load HTML content directly
        try:
            markdown = load_html_as_markdown(url)
            initial_message = {
                "messages": [
                    HumanMessage(content=f'会議のURLは"{url}"です。'),
                    HumanMessage(content=f"マークダウンは以下の通りです：\n\n{markdown}"),
                ],
                "url": url,
                "batch": batch,
                "skip_bluesky_posting": skip_bluesky_posting,
                "overview_only": overview_only,
                "is_meeting_page": True,  # HTMLページは会議ページとして初期化
            }
        except Exception as e:
            return {
                "success": False,
                "summary": "",
                "url": url,
                "error": f"Error loading HTML content: {e}",
            }

        graph.add_edge(START, "main_content_extractor")

        # main_content_extractorの後の条件分岐を追加
        graph.add_conditional_edges(
            "main_content_extractor",
            should_continue_from_main_content_extractor,
            {
                "overview_generator": "overview_generator",
                END: END,
            },
        )

        # overview_generatorの後の処理（条件分岐で制御）
        graph.add_conditional_edges(
            "overview_generator",
            should_process_additional_files,
            {
                "summary_finalizer": "summary_finalizer",
                "report_enumerator": "report_enumerator",
            },
        )

        graph.add_edge("report_enumerator", "report_selector")

        # report_selectorの後の条件分岐を追加
        graph.add_conditional_edges(
            "report_selector",
            should_continue_target_reports,
            {
                "document_summarizer": "document_summarizer",
                "summary_integrator": "summary_integrator",
                END: END,
            },
        )

        # document_summarizerの後の条件分岐を追加
        graph.add_conditional_edges(
            "document_summarizer",
            should_continue_target_reports,
            {
                "document_summarizer": "document_summarizer",
                "summary_integrator": "summary_integrator",
                END: END,
            },
        )

        # summary_integratorの後は常にsummary_finalizerへ
        graph.add_edge("summary_integrator", "summary_finalizer")

        # summary_finalizerの後の処理（条件分岐でBluesky投稿の有無を判定）
        graph.add_conditional_edges(
            "summary_finalizer",
            should_post_to_bluesky,
            {
                "bluesky_poster": "bluesky_poster",
                END: END,
            },
        )
        graph.add_edge("bluesky_poster", END)
    else:  # pdf
        # PDFファイルの場合は直接document_summarizerで処理
        initial_message = {
            "messages": [HumanMessage(content=f'PDFファイルのURLは"{url}"です。')],
            "url": url,
            "target_reports": TargetReportList(
                reports=[
                    Report(url=url, name="", reason="直接指定されたPDFファイル")
                ]
            ),
            "target_report_index": 0,
            "overview": "",  # summary_integratorで使用
            "batch": batch,
            "skip_bluesky_posting": skip_bluesky_posting,
            "is_meeting_page": False,  # PDF単体は会議ページではない
        }

        # PDFフロー：START -> document_summarizer -> summary_integrator -> summary_finalizer -> bluesky_poster -> END
        graph.add_edge(START, "document_summarizer")
        graph.add_edge("document_summarizer", "summary_integrator")
        graph.add_edge("summary_integrator", "summary_finalizer")

        # summary_finalizerの後の処理（条件分岐でBluesky投稿の有無を判定）
        graph.add_conditional_edges(
            "summary_finalizer",
            should_post_to_bluesky,
            {
                "bluesky_poster": "bluesky_poster",
                END: END,
            },
        )
        graph.add_edge("bluesky_poster", END)

    memory = MemorySaver()
    graph = graph.compile(checkpointer=memory)

    for _event in graph.stream(initial_message, config):
        pass

    # Get the final state and output the meeting title
    final_state = graph.get_state(config)
    final_review_summary = final_state.values.get("final_review_summary", "")
    final_summary = final_state.values.get("final_summary", "")
    overview = final_state.values.get("overview", "")
    main_content = final_state.values.get("main_content", "")

    # HTMLパースエラーでの終了をチェック
    if "[HTML_PARSING_ERROR]" in main_content:
        return {
            "success": False,
            "summary": "",
            "url": url,
            "error": "Failed to extract main content from HTML",
        }

    # Use the reviewed summary if available, otherwise fall back to original logic
    if final_review_summary:
        # Human-reviewed summaryがある場合（最優先）
        print(f"{final_review_summary}\n{url}")
        return {
            "success": True,
            "summary": final_review_summary,
            "url": url,
            "error": "",
        }
    elif final_summary:
        # final_summaryがある場合
        print(f"{final_summary}\n{url}")
        return {
            "success": True,
            "summary": final_summary,
            "url": url,
            "error": "",
        }
    elif overview:
        # final_summaryが空でoverviewがある場合
        print(f"{overview}\n{url}")
        return {
            "success": True,
            "summary": overview,
            "url": url,
            "error": "",
        }
    else:
        # 両方とも空の場合
        print("No summary created", file=sys.stderr)
        print(url)
        return {
            "success": False,
            "summary": "",
            "url": url,
            "error": "No summary created",
        }
