#!/usr/bin/env python3
"""
Test script for PowerPointSummarizer sub-agent.

This script allows testing the PowerPointSummarizer in isolation,
useful for Phase 1 validation (token reduction measurement and quality check).

Usage:
    python tests/test_ppt_summarizer.py <PDF_FILE_PATH>

    # Or with poetry:
    poetry run python tests/test_ppt_summarizer.py <PDF_FILE_PATH>

Example:
    python tests/test_ppt_summarizer.py /path/to/presentation.pdf
    poetry run python tests/test_ppt_summarizer.py https://example.go.jp/doc.pdf
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jpgovsummary import Model, logger
from jpgovsummary.jpgovwatcher import setup
from jpgovsummary.subagents.powerpoint_summarizer import PowerPointSummarizer
from jpgovsummary.tools.pdf_loader import load_pdf_as_text


def main():
    """Test PowerPointSummarizer with a PDF file or URL."""
    setup()

    parser = argparse.ArgumentParser(
        description="Test PowerPointSummarizer sub-agent with a PowerPoint PDF"
    )
    parser.add_argument(
        "pdf_path", type=str, help="Path or URL to PowerPoint PDF file"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="LLM model to use (e.g., gpt-4o)"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Check if it's a URL or file path
    is_url = args.pdf_path.startswith("http://") or args.pdf_path.startswith("https://")

    if not is_url:
        # Validate file exists
        pdf_path = Path(args.pdf_path)
        if not pdf_path.exists():
            print(f"Error: File not found: {pdf_path}", file=sys.stderr)
            return 1

        if not pdf_path.suffix.lower() == ".pdf":
            print(f"Error: File must be a PDF: {pdf_path}", file=sys.stderr)
            return 1

        display_name = pdf_path.name
        source = str(pdf_path)
    else:
        display_name = args.pdf_path.split("/")[-1]
        source = args.pdf_path

    # Initialize model
    if args.model:
        model = Model(args.model)
    else:
        model = Model()

    print("=" * 80, file=sys.stderr)
    print(f"Testing PowerPointSummarizer with: {display_name}", file=sys.stderr)
    print(f"Source: {source}", file=sys.stderr)
    print(f"Model: {model.model}", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(file=sys.stderr)

    try:
        # Load PDF
        logger.info(f"Loading PDF: {source}")
        pdf_pages = load_pdf_as_text(source)
        logger.info(f"Loaded {len(pdf_pages)} pages")

        # Initialize summarizer
        summarizer = PowerPointSummarizer(model=model)

        # Run summarization
        logger.info("Starting PowerPoint summarization...")
        result = summarizer.invoke(
            {
                "pdf_pages": pdf_pages,
                "url": source,
            }
        )

        # Display results
        print(file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print("RESULTS", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print(file=sys.stderr)

        print(f"Title: {result['title']}", file=sys.stderr)
        print(f"Summary length: {len(result['summary'])} characters", file=sys.stderr)
        print(file=sys.stderr)

        print("Summary:", file=sys.stderr)
        print("-" * 80, file=sys.stderr)
        print(result["summary"], file=sys.stderr)
        print("-" * 80, file=sys.stderr)

        # Output in jpgovsummary format (stdout)
        print()
        print(result["summary"])
        print(result["url"])

        return 0

    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        import traceback

        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
