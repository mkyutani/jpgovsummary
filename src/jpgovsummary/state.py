from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langgraph.graph.message import AnyMessage, add_messages

class Report(BaseModel):
    url: str = Field(description="The URL of the report document")
    name: str = Field(description="The name of the report document")

class ReportList(BaseModel):
    reports: List[Report] = Field(description="List of report documents")

class State(TypedDict):
    meeting_title: Optional[str]
    messages: Annotated[list[AnyMessage], add_messages]
    reports: Optional[ReportList]
    summary: Optional[str]
