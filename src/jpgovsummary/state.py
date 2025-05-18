from typing import Annotated, List, Optional, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langgraph.graph.message import AnyMessage, add_messages

class ScoredReport(BaseModel):
    url: str = Field(description="The URL of the report document")
    name: str = Field(description="The text/name of the report document")
    score: int = Field(description="The importance score of the report (1-5)", ge=1, le=5)
    reason: str = Field(description="The reason why this report was selected")

class CandidateReport(BaseModel):
    url: str = Field(description="The URL of the report document")
    name: str = Field(description="The text/name of the report document")
    is_document: bool = Field(description="Whether this URL points to a relevant document")
    reason: str = Field(description="Reason for the document judgment")

class ScoredReportList(BaseModel):
    reports: List[ScoredReport] = Field(description="List of selected reports to be used for summarization")

class CandidateReportList(BaseModel):
    reports: List[CandidateReport] = Field(description="List of candidate reports to be selected")

class State(TypedDict):
    """
    State for the application.
    """
    main_content: Optional[str] = Field(description="The main content of the meeting extracted from markdown")
    markdown: Optional[str] = Field(description="The markdown content of the meeting")
    overview: Optional[str] = Field(description="The overview of the meeting, including the title and the URL")
    messages: Annotated[list[AnyMessage], add_messages] = Field(description="The messages of the conversation")
    candidate_reports: Optional[CandidateReportList] = Field(description="The candidate reports to be selected")
    scored_reports: Optional[ScoredReportList] = Field(description="The selected reports to be used for summarization")
    target_reports: Optional[ScoredReportList] = Field(description="The highest scored reports to be summarized")
    overview_summary: Optional[str] = Field(description="The overview summary of the meeting")
    target_report_summaries: Optional[List[str]] = Field(description="The summaries of the target reports")
    target_report_index: Optional[int] = Field(description="The current index for document summarization", default=0)
