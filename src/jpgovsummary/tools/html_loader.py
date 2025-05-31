import requests
from langchain_core.tools import tool
from markitdown import MarkItDown

from .. import logger


def load_html_as_markdown(url: str) -> str:
    """
    Load HTML page into markdown string.

    Args:
        url (str): URL of the page with ending .html

    Returns:
        str: markdown of the page
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers, timeout=30, verify=True)
    response.raise_for_status()
    markdown = MarkItDown().convert(response)
    return markdown.text_content


@tool
def html_loader(html_url: str) -> str:
    """
    ## HTML Loader

    Load HTML page into markdown string.

    Args:
        html_url (str): URL of the page with ending .html

    Returns:
        str: markdown of the page
    """
    logger.info("html_loader")

    markdown = load_html_as_markdown(html_url)
    logger.info(f"length: {len(markdown)}")

    return markdown
