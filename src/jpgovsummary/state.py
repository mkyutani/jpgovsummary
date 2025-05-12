from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langgraph.graph.message import AnyMessage, add_messages

class Report(BaseModel):
    url: str = Field(description="The URL of the report document")
    name: str = Field(description="The name of the report document")
    is_document: bool = Field(description="Whether this URL points to a relevant document")
    reason: str = Field(description="Reason for the document judgment")

class ReportList(BaseModel):
    reports: List[Report] = Field(description="List of report documents")

class State(TypedDict):
    main_content: Optional[str] = Field(description="The main content of the meeting extracted from markdown")
    markdown: Optional[str] = Field(description="The markdown content of the meeting")
    meeting_title: Optional[str] = Field(description="The title of the meeting")
    messages: Annotated[list[AnyMessage], add_messages] = Field(description="The messages of the conversation")
    reports: Optional[ReportList]
#    reports: Optional[str] = Field(description="The reports of the meeting")
    summary: Optional[str] = Field(description="The summary of the meeting")
