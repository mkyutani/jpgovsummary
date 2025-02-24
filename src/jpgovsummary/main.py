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
from .meeting_information_collector import meeting_information_collector
from .meeting_page_reader import MeetingPageReader
from .property_writer import PropertyWriter
from .reporter import Reporter
from .state import State

def setup():
    signal.signal(signal.SIGINT, lambda num, frame: sys.exit(1))
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=False)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=False)

def route_tools(
    state: State,
):
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

    args = parser.parse_args()
    uuid = args.uuid[0]

    graph = StateGraph(State)

    graph.add_node('reporter', Reporter(uuid).node)
    graph.add_node('meeting_information_collector', ToolNode(tools=[meeting_information_collector]))
    graph.add_node('property_writer', PropertyWriter().node)
    graph.add_node('meeting_page_reader', MeetingPageReader().node)

    graph.add_edge(START, 'reporter')
    graph.add_conditional_edges('reporter', route_tools, {'tools': 'meeting_information_collector', 'next': 'property_writer'})
    graph.add_edge('meeting_information_collector', 'property_writer')
    graph.add_edge('property_writer', 'meeting_page_reader')
    graph.add_edge('meeting_page_reader', END)

    memory = MemorySaver()
    graph = graph.compile(checkpointer=memory)
    config = Config(uuid).get()

    history = []
    properties = {}
    events = graph.stream(
        {'messages': [HumanMessage(content=f'会議の番号は{uuid}です。概要を説明してください。')]},
        config
    )
    for event in events:
        for value in event.values():
            history.append(value['messages'][-1].content)
            if type(value['messages'][-1]) is ToolMessage:
                props = json.loads(value['messages'][-1].content)
                properties.update(props)

    print(f'''{history[-1]}\n{properties['footer']}''')

    return 0

if __name__ == '__main__':
    sys.exit(main())