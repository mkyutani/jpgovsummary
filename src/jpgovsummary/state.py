from collections.abc import Iterator
from typing import Annotated, Generic, TypeVar

from langgraph.graph.message import AnyMessage, add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

T = TypeVar("T")


class Report(BaseModel):
    url: str = Field(description="The URL of the report document")
    name: str = Field(description="The text/name of the report document")
    reason: str = Field(description="The reason for the report")


class Summary(BaseModel):
    url: str = Field(description="The URL of the summarized document")
    name: str = Field(description="The name of the summarized document")
    content: str = Field(description="The summary content of the document")
    document_type: str | None = Field(description="The detected document type (word/pptx/other)", default=None)
    detection_detail: dict | None = Field(description="The document type detection details", default=None)


class ReportList(BaseModel, Generic[T]):
    reports: list[T] = Field(description="List of reports")

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
    reports: list[ScoredReport] = Field(
        description="List of selected reports to be used for summarization"
    )


class CandidateReportList(ReportList[CandidateReport]):
    reports: list[CandidateReport] = Field(description="List of candidate reports to be selected")


class TargetReportList(ReportList[Report]):
    reports: list[Report] = Field(description="List of target reports to be summarized")


class State(TypedDict):
    """
    State for the application.
    """

    main_content: str | None = Field(
        description="The main content of the meeting extracted from markdown"
    )
    markdown: str | None = Field(description="The markdown content of the meeting")
    messages: Annotated[list[AnyMessage], add_messages] = Field(
        description="The messages of the conversation"
    )
    candidate_reports: CandidateReportList | None = Field(
        description="The candidate reports to be selected"
    )
    scored_reports: ScoredReportList | None = Field(
        description="The selected reports to be used for summarization"
    )
    target_reports: TargetReportList | None = Field(
        description="The highest scored reports to be summarized"
    )
    overview: str | None = Field(description="The overview of the meeting")
    target_report_summaries: list[Summary] | None = Field(
        description="The summaries of the target reports"
    )
    target_report_index: int | None = Field(
        description="The current index for document summarization", default=0
    )
    url: str | None = Field(description="The URL of the meeting")
    final_summary: str | None = Field(
        description="The final integrated summary from all sources", default=None
    )
    summary_retry_count: int | None = Field(
        description="The retry count for summary_integrator when final_summary exceeds character limit",
        default=0,
    )
    # Human review fields
    review_session: dict | None = Field(
        description="Human review session data including Q&A history and improvements",
        default=None,
    )
    review_approved: bool | None = Field(
        description="Whether the human reviewer approved the final summary",
        default=None,
    )
    review_completed: bool | None = Field(
        description="Whether the human review process has been completed",
        default=False,
    )
    final_review_summary: str | None = Field(
        description="The final summary after human review and improvements",
        default=None,
    )
    batch: bool | None = Field(
        description="Whether to run in batch mode without human interaction",
        default=False,
    )
    # Bluesky posting fields
    skip_bluesky_posting: bool | None = Field(
        description="Whether to skip the Bluesky posting step",
        default=False,
    )
    bluesky_post_requested: bool | None = Field(
        description="Whether user requested Bluesky posting",
        default=None,
    )
    bluesky_post_completed: bool | None = Field(
        description="Whether Bluesky posting was completed",
        default=False,
    )
    bluesky_post_content: str | None = Field(
        description="Content posted to Bluesky",
        default=None,
    )
    bluesky_post_response: str | None = Field(
        description="Response from Bluesky posting (JSON string)",
        default=None,
    )
    overview_only: bool | None = Field(
        description="Whether to generate overview only without processing additional documents",
        default=False,
    )
    meeting_minutes_detected: bool | None = Field(
        description="Whether meeting minutes were detected in the HTML content",
        default=False,
    )
    is_meeting_page: bool | None = Field(
        description="Whether the content is from a meeting page (HTML) or individual document (PDF)",
        default=False,
    )
