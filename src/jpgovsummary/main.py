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
    main_content_extractor,
    overview_generator,
    report_enumerator,
    report_selector,
    summary_integrator,
)
from .tools import load_html_as_markdown


def get_page_type(url: str) -> str:
    """
    Determine the page type based on Content-Type header.

    Args:
        url (str): URL to check the page type

    Returns:
        str: Page type ("html", "text", "pdf", "application", "unknown")
    """
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


def should_continue(state: State) -> str | bool:
    """
    次のステップを決定する条件分岐
    """
    # summary_integratorの結果をチェック（final_summaryが存在する場合）
    if "final_summary" in state and state["final_summary"]:
        final_summary = state["final_summary"]
        url = state.get("url", "")

        # 最終出力全体（final_summary + "\n" + url）が300文字以上の場合のみ処理
        if len(f"{final_summary}\n{url}") >= 300:
            # 再実行回数を取得（初回は0）
            summary_retry_count = state.get("summary_retry_count", 0)
            max_retries = 3  # 最大再実行回数

            # まだ再実行回数の上限に達していない場合は再実行
            if summary_retry_count < max_retries:
                # 再実行回数をインクリメント
                state["summary_retry_count"] = summary_retry_count + 1

                # ログ出力
                logger.warning("Retrying summary_integrator: final_summary exceeds 300 characters")

                # より短い要約を求めるメッセージを追加
                retry_message = HumanMessage(
                    content=f"前回の要約が{len(final_summary)}文字で300文字以上になっています。299文字以下でより簡潔な要約を作成してください。\n\n前回の要約: {final_summary}\n\nURL: {url}"
                )

                # 既存のメッセージに追加
                current_messages = state.get("messages", [])
                state["messages"] = current_messages + [retry_message]

                return "summary_integrator"
            else:
                # 再実行上限に達した場合
                logger.warning(
                    "summary_integrator retry limit exceeded: final_summary still exceeds 300 characters"
                )

        # 299文字以下、または再実行上限に達した場合は終了
        return END

    # document_summarizerの結果を確認
    if "target_reports" in state:
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

        # 有効な要約（contentが存在する）が1つ以上あるかチェック
        valid_summaries = [s for s in target_report_summaries if s.get("content", "")]

        if valid_summaries:
            return "summary_integrator"
        else:
            # 要約がない場合は、overviewを最終要約として設定
            overview = state.get("overview", "")
            url = state.get("url", "")
            message = HumanMessage(content=f"{overview}\n{url}")
            state["messages"] = [message]
            state["final_summary"] = overview
            return END

    return END


def main() -> int:
    setup()

    parser = argparse.ArgumentParser(description="RAG-based web browsing agent")
    parser.add_argument("url", nargs="?", type=str, help="URL of the meeting")
    parser.add_argument(
        "--graph", nargs=1, type=str, default=None, help="Output file path for the graph"
    )
    parser.add_argument("--model", type=str, default=None, help="OpenAI model to use")

    args = parser.parse_args()

    if args.url is None:
        print("No meeting URL provided", file=sys.stderr)
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
            }
        except Exception as e:
            print(f"Error loading HTML content: {e}", file=sys.stderr)
            return 1

        graph.add_edge(START, "main_content_extractor")
        graph.add_edge("main_content_extractor", "overview_generator")
        graph.add_edge("overview_generator", "report_enumerator")
        graph.add_edge("report_enumerator", "report_selector")

        # report_selectorの後の条件分岐を追加
        graph.add_conditional_edges(
            "report_selector",
            should_continue,
            {
                "document_summarizer": "document_summarizer",
                "summary_integrator": "summary_integrator",
                END: END,
            },
        )

        # document_summarizerの後の条件分岐を追加
        graph.add_conditional_edges(
            "document_summarizer",
            should_continue,
            {
                "document_summarizer": "document_summarizer",
                "summary_integrator": "summary_integrator",
                END: END,
            },
        )

        # summary_integratorの後の条件分岐を追加
        graph.add_conditional_edges(
            "summary_integrator",
            should_continue,
            {"summary_integrator": "summary_integrator", END: END},
        )
    else:  # pdf
        # PDFファイルの場合は直接document_summarizerで処理
        initial_message = {
            "messages": [HumanMessage(content=f'PDFファイルのURLは"{args.url}"です。')],
            "url": args.url,
            "target_reports": TargetReportList(
                reports=[
                    Report(url=args.url, name="PDFファイル", reason="直接指定されたPDFファイル")
                ]
            ),
            "target_report_index": 0,
            "overview": "",  # summary_integratorで使用
        }

        # PDFフロー：START -> document_summarizer -> summary_integrator -> END
        graph.add_edge(START, "document_summarizer")

        # document_summarizerの後の条件分岐を追加
        graph.add_conditional_edges(
            "document_summarizer",
            should_continue,
            {
                "document_summarizer": "document_summarizer",
                "summary_integrator": "summary_integrator",
                END: END,
            },
        )

        # summary_integratorの後の条件分岐を追加
        graph.add_conditional_edges(
            "summary_integrator",
            should_continue,
            {"summary_integrator": "summary_integrator", END: END},
        )

    memory = MemorySaver()
    graph = graph.compile(checkpointer=memory)

    if args.graph:
        graph.get_graph().draw_png(output_file_path=args.graph[0])
        return 0

    for _event in graph.stream(initial_message, config):
        pass

    # Get the final state and output the meeting title
    final_state = graph.get_state(config)
    final_summary = final_state.values.get("final_summary")
    url = final_state.values.get("url")

    if final_summary and url:
        print(f"{final_summary}\n{url}")
    else:
        # 従来の出力をフォールバックとして使用
        overview = final_state.values.get("overview")
        if overview and url:
            print(f"{overview}\n{url}")
        else:
            print("No summary found", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
