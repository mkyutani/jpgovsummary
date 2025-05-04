from typing import Annotated, Optional
from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages

class State(TypedDict):
    meeting_title: Optional[str]
    messages: Annotated[list[AnyMessage], add_messages]
