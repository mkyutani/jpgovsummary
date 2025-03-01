import argparse
import io
import json
import signal
import sys

from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from .config import Config
from .meeting_information_collector import MeetingInformationCollector
from .meeting_page_reader import MeetingPageReader
from .property_formatter import PropertyFormatter
from .researcher import Researcher
from .state import State

def setup() -> None:
    signal.signal(signal.SIGINT, lambda num, frame: sys.exit(1))
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)

def route_tools(state: State) -> str:
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get('messages', []):
        ai_message = messages[-1]
    else:
        raise ValueError(f'No messages found in input state to tool_edge: {state}')

    if hasattr(ai_message, 'tool_calls') and len(ai_message.tool_calls) > 0:
        return 'tools'

    return 'next'

def main() -> int:
    setup()

    parser = argparse.ArgumentParser(description='RAG-based web browsing agent')
    parser.add_argument('uuid', nargs=1, type=str, help='UUID of the meeting')
    parser.add_argument('--output-graph', nargs=1, type=str, default=None, help='Output file path for the graph')

    args = parser.parse_args()
    uuid = args.uuid[0]

    graph = StateGraph(State)

    graph.add_node('researcher', Researcher(uuid).node)
    graph.add_node('meeting_information_collector', ToolNode(tools=[MeetingInformationCollector.tool]))
#    graph.add_node('meeting_information_collector', ToolNode(tools=[meeting_information_collector]))
    graph.add_node('property_formatter', PropertyFormatter().node)
    graph.add_node('meeting_page_reader', MeetingPageReader().node)

    graph.add_edge(START, 'researcher')
    graph.add_conditional_edges('researcher', route_tools, {'tools': 'meeting_information_collector', 'next': 'property_formatter'})
    graph.add_edge('meeting_information_collector', 'researcher')
    graph.add_edge('property_formatter', 'meeting_page_reader')
    graph.add_edge('meeting_page_reader', END)

    memory = MemorySaver()
    graph = graph.compile(checkpointer=memory)
    config = Config(uuid).get()

    events = graph.stream(
        {'messages': [HumanMessage(content=f'会議の番号は{uuid}です。概要を説明してください。')]},
        config
    )
    for event in events:
        for value in event.values():
            last_message = value['messages'][-1]
            last_message.pretty_print()

    if args.output_graph:
        graph.get_graph().draw_png(output_file_path=args.output_graph[0])

    return 0

if __name__ == '__main__':
    sys.exit(main())