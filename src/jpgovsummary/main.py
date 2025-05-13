import argparse
import os
import signal
import sys

from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from . import Config, Model, route_tools, State
from .agents import (
    meeting_page_type_selector,
    overview_generator,
    report_enumerator,
    summary_writer,
    main_content_extractor
)
from .tools import (
    html_loader,
    pdf_loader
)

def setup() -> None:
    signal.signal(signal.SIGINT, lambda num, frame: sys.exit(1))
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

def main() -> int:
    setup()

    parser = argparse.ArgumentParser(description="RAG-based web browsing agent")
    parser.add_argument("url", nargs='?', type=str, help="URL of the meeting")
    parser.add_argument("--graph", nargs=1, type=str, default=None, help="Output file path for the graph")
    parser.add_argument("--log", action="store_true", help="Print graph logs")
    parser.add_argument("--model", type=str, default=None, help="OpenAI model to use")

    args = parser.parse_args()

    # Initialize the default model
    if args.model:
        Model(args.model)
    else:
        Model()

    config = Config(1).get()
    graph = StateGraph(State)

    # Add agent nodes
    graph.add_node("meeting_page_type_selector", meeting_page_type_selector)
    graph.add_node("overview_generator", overview_generator)
    graph.add_node("summary_writer", summary_writer)
    graph.add_node("report_enumerator", report_enumerator)
    graph.add_node("main_content_extractor", main_content_extractor)
    # Add tool nodes
    graph.add_node("html_loader", ToolNode(tools=[html_loader]))
    graph.add_node("pdf_loader", ToolNode(tools=[pdf_loader]))

    # Define graph edges
    graph.add_edge(START, "meeting_page_type_selector")
    graph.add_conditional_edges("meeting_page_type_selector", route_tools, {"html_loader": "html_loader", "pdf_loader": "pdf_loader"})
    graph.add_edge("html_loader", "overview_generator")
    graph.add_edge("overview_generator", "main_content_extractor")
    graph.add_edge("main_content_extractor", "summary_writer")
    graph.add_edge("summary_writer", "report_enumerator")
    graph.add_edge("report_enumerator", END)
    graph.add_edge("pdf_loader", END)

    memory = MemorySaver()
    graph = graph.compile(checkpointer=memory)

    if args.graph:
        graph.get_graph().draw_png(output_file_path=args.graph[0])
        return 0

    if args.url is None:
        print("No meeting URL provided", file=sys.stderr)
        return 1

    human_message = f"会議のURLは\"{args.url}\"です。"

    initial_message = {
        "messages": [HumanMessage(content=human_message)]
    }
    for event in graph.stream(initial_message, config):
        if args.log:
            for value in event.values():
                if isinstance(value["messages"], list):
                    last_message = value["messages"][-1]
                else:
                    last_message = value["messages"]
                last_message.pretty_print()

    if args.log:
        print("-" * 80, file=sys.stderr)

    # Get the final state and output the meeting title
    final_state = graph.get_state(config)
    summary = final_state.values.get("summary")

    if summary:
        print(summary)
    else:
        print("No summary found", file=sys.stderr)

    return 0

if __name__ == "__main__":
    sys.exit(main())