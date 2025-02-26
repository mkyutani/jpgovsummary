import operator
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list[str], add_messages] = []
    title: str = None
    number: str = None
    date: str = None
    url: str = None