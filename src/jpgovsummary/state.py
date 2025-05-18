from typing import Annotated, List, Optional, Any, Dict, Iterator, TypeVar, Generic
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langgraph.graph.message import AnyMessage, add_messages

T = TypeVar('T')

class Report(BaseModel):
    url: str = Field(description="The URL of the report document")
    name: str = Field(description="The text/name of the report document")
    reason: str = Field(description="The reason for the report")

class ReportList(BaseModel, Generic[T]):
    reports: List[T] = Field(description="List of reports")

    def __len__(self) -> int:
        return len(self.reports)

    def __iter__(self) -> Iterator[T]:
        return iter(self.reports)

    def __getitem__(self, index: int) -> T:
        return self.reports[index]

    def __str__(self) -> str:
        return self.model_dump_json(indent=2)

class ScoredReport(Report):
    score: int = Field(description="The importance score of the report (1-5)", ge=1, le=5)

class CandidateReport(Report):
    is_document: bool = Field(description="Whether this URL points to a relevant document")

class ScoredReportList(ReportList[ScoredReport]):
    reports: List[ScoredReport] = Field(description="List of selected reports to be used for summarization")

class CandidateReportList(ReportList[CandidateReport]):
    reports: List[CandidateReport] = Field(description="List of candidate reports to be selected")

class TargetReportList(ReportList[Report]):
    reports: List[Report] = Field(description="List of target reports to be summarized")

class State(TypedDict):
    """
    State for the application.
    """
    main_content: Optional[str] = Field(description="The main content of the meeting extracted from markdown")
    markdown: Optional[str] = Field(description="The markdown content of the meeting")
    messages: Annotated[list[AnyMessage], add_messages] = Field(description="The messages of the conversation")
    candidate_reports: Optional[CandidateReportList] = Field(description="The candidate reports to be selected")
    scored_reports: Optional[ScoredReportList] = Field(description="The selected reports to be used for summarization")
    target_reports: Optional[TargetReportList] = Field(description="The highest scored reports to be summarized")
    overview: Optional[str] = Field(description="The overview of the meeting")
    target_report_summaries: Optional[List[str]] = Field(description="The summaries of the target reports")
    target_report_index: Optional[int] = Field(description="The current index for document summarization", default=0)
    url: Optional[str] = Field(description="The URL of the meeting")