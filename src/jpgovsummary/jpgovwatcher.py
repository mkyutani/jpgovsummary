import argparse
import signal
import sys

import requests
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from . import Config, Model, Report, State, TargetReportList, logger
from .agents import (
    document_summarizer,
    summary_finalizer,
    main_content_extractor,
    overview_generator,
    report_enumerator,
    report_selector,
    summary_integrator,
    bluesky_poster,
)
from .tools import load_html_as_markdown
from .utils import is_local_file, get_local_file_path, validate_local_file


def get_page_type(url: str) -> str:
    """
    Determine the page type based on Content-Type header for URLs or file extension for local files.

    Args:
        url (str): URL or local file path to check the page type

    Returns:
        str: Page type ("html", "text", "pdf", "application", "unknown")
    """
    # Check if it's a local file
    if is_local_file(url):
        file_path = get_local_file_path(url)
        
        try:
            validate_local_file(file_path)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return "unknown"
        
        # Determine type by file extension
        file_path_lower = file_path.lower()
        if file_path_lower.endswith(".pdf"):
            return "pdf"
        elif file_path_lower.endswith((".html", ".htm")):
            return "html"
        elif file_path_lower.endswith(".txt"):
            return "text"
        else:
            return "unknown"
    
    # Handle remote URLs (existing logic)
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }
        response = requests.head(url, headers=headers, allow_redirects=True)
        content_type = response.headers.get("Content-Type", "").lower()

        if "application/pdf" in content_type:
            return "pdf"
        elif content_type.startswith("application/"):
            return "application"
        elif "text/html" in content_type:
            return "html"
        elif content_type.startswith("text/"):
            return "text"
        else:
            return "unknown"
    except Exception as e:
        print(f"Error checking page type: {e}", file=sys.stderr)
        return "unknown"


def setup() -> None:
    signal.signal(signal.SIGINT, lambda num, frame: sys.exit(1))
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)


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
    target_report_summaries = state.get("target_report_summaries", [])

    # すべての資料の要約が完了した場合はsummary_integratorへ遷移
    # summary_integratorで有効な要約がない場合のハンドリングを行う
    return "summary_integrator"


def main() -> int:
    setup()

    parser = argparse.ArgumentParser(description="RAG-based web browsing agent")
    parser.add_argument("url", nargs="?", type=str, help="URL of the meeting or local file path (PDF/HTML)")
    parser.add_argument(
        "--graph", nargs=1, type=str, default=None, help="Output file path for the graph"
    )
    parser.add_argument("--model", type=str, default=None, help="OpenAI model to use")
    parser.add_argument(
        "--skip-human-review", action="store_true", 
        help="Skip human review step for automated workflows"
    )
    parser.add_argument(
        "--skip-bluesky-posting", action="store_true",
        help="Skip Bluesky posting step"
    )
    parser.add_argument(
        "--overview-only", action="store_true",
        help="Generate overview only without processing additional documents"
    )

    args = parser.parse_args()

    if args.url is None:
        print("No meeting URL or file path provided", file=sys.stderr)
        return 1

    # Check page type
    page_type = get_page_type(args.url)
    if page_type not in ["html", "pdf"]:
        print(f"Unsupported page type: {page_type}", file=sys.stderr)
        return 1

    # Initialize the default model
    if args.model:
        Model(args.model)
    else:
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
            markdown = load_html_as_markdown(args.url)
            initial_message = {
                "messages": [
                    HumanMessage(content=f'会議のURLは"{args.url}"です。'),
                    HumanMessage(content=f"マークダウンは以下の通りです：\n\n{markdown}"),
                ],
                "url": args.url,
                "skip_human_review": args.skip_human_review,
                "skip_bluesky_posting": args.skip_bluesky_posting,
                "overview_only": args.overview_only,
            }
        except Exception as e:
            print(f"Error loading HTML content: {e}", file=sys.stderr)
            return 1

        graph.add_edge(START, "main_content_extractor")
        graph.add_edge("main_content_extractor", "overview_generator")
        
        # overview_generatorの後の処理（overview-onlyモードで分岐）
        if args.overview_only:
            # overview-onlyの場合は直接summary_finalizerへ
            graph.add_edge("overview_generator", "summary_finalizer")
        else:
            # 通常のフローは report_enumerator へ
            graph.add_edge("overview_generator", "report_enumerator")
        
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
        
        # summary_finalizerの後の処理（Bluesky投稿の有無で分岐）
        if args.skip_bluesky_posting:
            # Bluesky投稿をスキップする場合は直接終了
            graph.add_edge("summary_finalizer", END)
        else:
            # Bluesky投稿を実行する場合
            graph.add_edge("summary_finalizer", "bluesky_poster")
            graph.add_edge("bluesky_poster", END)
    else:  # pdf
        # PDFファイルの場合は直接document_summarizerで処理
        initial_message = {
            "messages": [HumanMessage(content=f'PDFファイルのURLは"{args.url}"です。')],
            "url": args.url,
            "target_reports": TargetReportList(
                reports=[
                    Report(url=args.url, name="", reason="直接指定されたPDFファイル")
                ]
            ),
            "target_report_index": 0,
            "overview": "",  # summary_integratorで使用
            "skip_human_review": args.skip_human_review,
            "skip_bluesky_posting": args.skip_bluesky_posting,
        }

        # PDFフロー：START -> document_summarizer -> summary_integrator -> summary_finalizer -> bluesky_poster -> END
        graph.add_edge(START, "document_summarizer")
        graph.add_edge("document_summarizer", "summary_integrator")
        graph.add_edge("summary_integrator", "summary_finalizer")
        
        # summary_finalizerの後の処理（Bluesky投稿の有無で分岐）
        if args.skip_bluesky_posting:
            # Bluesky投稿をスキップする場合は直接終了
            graph.add_edge("summary_finalizer", END)
        else:
            # Bluesky投稿を実行する場合
            graph.add_edge("summary_finalizer", "bluesky_poster")
            graph.add_edge("bluesky_poster", END)

    memory = MemorySaver()
    graph = graph.compile(checkpointer=memory)

    if args.graph:
        graph.get_graph().draw_png(output_file_path=args.graph[0])
        return 0

    for _event in graph.stream(initial_message, config):
        pass

    # Get the final state and output the meeting title
    final_state = graph.get_state(config)
    final_review_summary = final_state.values.get("final_review_summary", "")
    final_summary = final_state.values.get("final_summary", "")
    overview = final_state.values.get("overview", "")
    url = final_state.values.get("url", "URL")

    # Use the reviewed summary if available, otherwise fall back to original logic
    if final_review_summary:
        # Human-reviewed summaryがある場合（最優先）
        print(f"{final_review_summary}\n{url}")
    elif final_summary:
        # final_summaryがある場合
        print(f"{final_summary}\n{url}")
    elif overview:
        # final_summaryが空でoverviewがある場合
        print(f"{overview}\n{url}")
    else:
        # 両方とも空の場合
        print("No summary created", file=sys.stderr)
        print(url)

    return 0


if __name__ == "__main__":
    sys.exit(main())
