#!/usr/bin/env python3
"""
Test script for HTMLProcessor sub-agent.

This script allows testing the HTMLProcessor in isolation,
useful for Phase 2 validation (HTML processing and document discovery).

Usage:
    python tests/test_html_processor.py <HTML_URL>

    # Or with poetry:
    poetry run python tests/test_html_processor.py <HTML_URL>

Example:
    python tests/test_html_processor.py https://www.kantei.go.jp/jp/singi/example/
    poetry run python tests/test_html_processor.py https://example.go.jp/meeting/
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jpgovsummary import Model, logger
from jpgovsummary.jpgovwatcher import setup
from jpgovsummary.subagents.html_processor import HTMLProcessor


def main():
    """Test HTMLProcessor with an HTML page URL."""
    setup()

    parser = argparse.ArgumentParser(
        description="Test HTMLProcessor sub-agent with an HTML meeting page"
    )
    parser.add_argument("url", type=str, help="URL of HTML meeting page")
    parser.add_argument(
        "--model", type=str, default=None, help="LLM model to use (e.g., gpt-4o)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Validate URL
    if not (args.url.startswith("http://") or args.url.startswith("https://")):
        print("Error: URL must start with http:// or https://", file=sys.stderr)
        return 1

    # Initialize model
    if args.model:
        model = Model(args.model)
    else:
        model = Model()

    print("=" * 80, file=sys.stderr)
    print(f"Testing HTMLProcessor with: {args.url}", file=sys.stderr)
    print(f"Model: {model.model}", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(file=sys.stderr)

    try:
        # Initialize processor
        processor = HTMLProcessor(model=model)

        # Run HTML processing
        logger.info("Starting HTML processing...")
        result = processor.invoke(
            {
                "url": args.url,
            }
        )

        # Display results
        print(file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print("RESULTS", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print(file=sys.stderr)

        # Markdown
        markdown = result.get("markdown")
        if markdown:
            print(f"Markdown length: {len(markdown)} characters", file=sys.stderr)
        else:
            print("Markdown: (failed to load)", file=sys.stderr)

        # Main content
        main_content = result.get("main_content")
        if main_content:
            print(f"Main content length: {len(main_content)} characters", file=sys.stderr)
            print(file=sys.stderr)
            print("Main Content (first 1000 chars):", file=sys.stderr)
            print("-" * 80, file=sys.stderr)
            print(main_content[:1000], file=sys.stderr)
            if len(main_content) > 1000:
                print("...", file=sys.stderr)
            print("-" * 80, file=sys.stderr)
        else:
            print("Main content: (failed to extract)", file=sys.stderr)

        # Discovered documents
        discovered_docs = result.get("discovered_documents", [])
        print(file=sys.stderr)
        print(f"Discovered documents: {len(discovered_docs)} files", file=sys.stderr)
        if discovered_docs:
            print("-" * 80, file=sys.stderr)
            for i, doc in enumerate(discovered_docs, 1):
                print(f"{i}. [{doc.category}] {doc.name}", file=sys.stderr)
                print(f"   URL: {doc.url}", file=sys.stderr)
            print("-" * 80, file=sys.stderr)

        # Output in simple format (stdout)
        print()
        print(f"Main content extracted: {bool(main_content)}")
        print(f"Discovered documents: {len(discovered_docs)}")
        if discovered_docs:
            print("\nDocument Details:")
            for doc in discovered_docs:
                print(f"  - [{doc.category}] {doc.name}")
                print(f"    {doc.url}")

        return 0

    except Exception as e:
        logger.error(f"Error during HTML processing: {e}")
        import traceback

        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
