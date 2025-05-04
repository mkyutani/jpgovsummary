import argparse
import signal
import sys

from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from . import Config, is_uuid, route_tools, State
from .agents import (
    base_url_generator,
    meeting_page_reader,
    summary_writer
)
from .tools import (
    html_loader,
    meeting_url_collector,
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
    parser.add_argument("id", nargs='?', type=str, help="UUID or URL of the meeting")
    parser.add_argument("--graph", nargs=1, type=str, default=None, help="Output file path for the graph")
    parser.add_argument("--log", action="store_true", help="Print graph logs")

    args = parser.parse_args()

    graph = StateGraph(State)

    # Add agent nodes
    graph.add_node("base_url_generator", base_url_generator)
    graph.add_node("meeting_page_reader", meeting_page_reader)
    graph.add_node("summary_writer", summary_writer)

    # Add tool nodes
    graph.add_node("meeting_url_collector", ToolNode(tools=[meeting_url_collector]))
    graph.add_node("html_loader", ToolNode(tools=[html_loader]))
    graph.add_node("pdf_loader", ToolNode(tools=[pdf_loader]))

    # Define graph edges
    graph.add_edge(START, "base_url_generator")
    graph.add_conditional_edges("base_url_generator", route_tools, {"meeting_url_collector": "meeting_url_collector", "skip": "meeting_page_reader"})
    graph.add_edge("meeting_url_collector", "meeting_page_reader")
    graph.add_conditional_edges("meeting_page_reader", route_tools, {"html_loader": "html_loader", "pdf_loader": "pdf_loader"})
    graph.add_edge("html_loader", "summary_writer")
    graph.add_edge("pdf_loader", END)

    memory = MemorySaver()
    graph = graph.compile(checkpointer=memory)
    config = Config(1).get()

    if args.graph:
        graph.get_graph().draw_png(output_file_path=args.graph[0])
        return 0

    if args.id is None:
        print("No meeting ID provided", file=sys.stderr)
        return 1

    if is_uuid(args.id):
        human_message = f"会議のUUIDは\"{args.id}\"です。"
    else:
        human_message = f"会議のURLは\"{args.id}\"です。"

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
        print(graph.get_state(config), file=sys.stderr)

    print("-" * 80, file=sys.stderr)
    print(graph.get_state(config).values["messages"][-1].content)

    return 0

if __name__ == "__main__":
    sys.exit(main())