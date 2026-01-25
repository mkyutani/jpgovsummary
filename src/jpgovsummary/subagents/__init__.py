"""
Sub-agents for isolated document processing.

This package contains specialized sub-agents that process specific
document types in isolated contexts, reducing main workflow token consumption.

Sub-agents:
- PowerPointSummarizer: Processes PowerPoint presentations (2000-3000 token reduction)
- DocumentTypeDetector: Detects document type from PDF content (1000-1500 token reduction)
- WordSummarizer: Processes Word documents
- HTMLProcessor: Handles HTML loading and content extraction

Each sub-agent is implemented as an independent LangGraph StateGraph
with its own isolated state, enabling parallel execution and better
error isolation.
"""

from jpgovsummary.subagents.document_type_detector import DocumentTypeDetector
from jpgovsummary.subagents.html_processor import HTMLProcessor
from jpgovsummary.subagents.powerpoint_summarizer import PowerPointSummarizer
from jpgovsummary.subagents.word_summarizer import WordSummarizer

__all__ = ["DocumentTypeDetector", "HTMLProcessor", "PowerPointSummarizer", "WordSummarizer"]
