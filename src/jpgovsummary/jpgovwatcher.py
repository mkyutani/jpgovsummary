import argparse
import signal
import sys

import requests

from . import Model
from .logger import set_batch_mode
from .utils import get_local_file_path, is_local_file, validate_local_file


def get_page_type(url: str) -> str:
    """
    Determine the page type based on Content-Type header for URLs or file extension for local files.

    Args:
        url (str): URL or local file path to check the page type

    Returns:
        str: Page type ("html", "text", "pdf", "application", "unknown")
    """
    # Check if it's a local file
    if is_local_file(url):
        file_path = get_local_file_path(url)

        try:
            validate_local_file(file_path)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return "unknown"

        # Determine type by file extension
        file_path_lower = file_path.lower()
        if file_path_lower.endswith(".pdf"):
            return "pdf"
        elif file_path_lower.endswith((".html", ".htm")):
            return "html"
        elif file_path_lower.endswith(".txt"):
            return "text"
        else:
            return "unknown"

    # Handle remote URLs (existing logic)
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }
        response = requests.head(url, headers=headers, allow_redirects=True)
        content_type = response.headers.get("Content-Type", "").lower()

        if "application/pdf" in content_type:
            return "pdf"
        elif content_type.startswith("application/"):
            return "application"
        elif "text/html" in content_type:
            return "html"
        elif content_type.startswith("text/"):
            return "text"
        else:
            return "unknown"
    except Exception as e:
        print(f"Error checking page type: {e}", file=sys.stderr)
        return "unknown"


def setup() -> None:
    signal.signal(signal.SIGINT, lambda num, frame: sys.exit(1))
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)


def main() -> int:
    setup()

    parser = argparse.ArgumentParser(description="RAG-based web browsing agent")
    parser.add_argument("url", nargs="?", type=str, help="URL of the meeting or local file path (PDF/HTML)")
    parser.add_argument("--model", type=str, default=None, help="OpenAI model to use")
    parser.add_argument(
        "--batch", action="store_true",
        help="Run in batch mode without human interaction"
    )
    parser.add_argument(
        "--skip-bluesky-posting", action="store_true",
        help="Skip Bluesky posting step"
    )
    parser.add_argument(
        "--overview-only", action="store_true",
        help="Generate overview only without processing additional documents"
    )
    parser.add_argument(
        "--use-v1", action="store_true",
        help="Use v1 legacy architecture instead of v2 Plan-Action architecture"
    )

    args = parser.parse_args()

    # Set logger mode based on batch option
    set_batch_mode(args.batch)

    if args.url is None:
        print("No meeting URL or file path provided", file=sys.stderr)
        return 1

    # Strip whitespace and control characters from URL/file path
    args.url = args.url.strip()

    # Initialize model
    if args.model:
        model = Model(args.model)
    else:
        model = Model()

    # Check page type (needed for both v1 and v2)
    page_type = get_page_type(args.url)
    if page_type not in ["html", "pdf"]:
        print(f"Unsupported page type: {page_type}", file=sys.stderr)
        return 1

    # Use v1 legacy architecture if --use-v1 flag is set
    if args.use_v1:
        from .v1.workflow import run_v1

        result = run_v1(
            url=args.url,
            page_type=page_type,
            model=model,
            batch=args.batch,
            skip_bluesky_posting=args.skip_bluesky_posting,
            overview_only=args.overview_only,
        )

        if result["success"]:
            return 0
        else:
            print(f"Error: {result.get('error')}", file=sys.stderr)
            return 1

    # Use v2 Plan-Action architecture by default
    from .workflow import run_v2

    result = run_v2(
        url=args.url,
        model=model,
        batch=args.batch,
        skip_bluesky_posting=args.skip_bluesky_posting,
        overview_only=args.overview_only,
    )

    if result["success"]:
        return 0
    else:
        print(f"Error: {result.get('error')}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
