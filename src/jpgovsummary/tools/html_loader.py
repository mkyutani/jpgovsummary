import requests
from langchain_core.tools import tool
from markitdown import MarkItDown

from .. import logger
from ..utils import is_local_file, get_local_file_path, validate_local_file


def load_html_as_markdown(url: str) -> str:
    """
    Load HTML page into markdown string from URL or local file.

    Args:
        url (str): URL of the page with ending .html or local file path

    Returns:
        str: markdown of the page
    """
    if is_local_file(url):
        # Handle local file
        file_path = get_local_file_path(url)
        validate_local_file(file_path)
        
        # Read local HTML file and convert to markdown using MarkItDown
        markdown = MarkItDown().convert(file_path)
        return markdown.text_content
    else:
        # Handle remote URL (existing logic)
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
    logger.info("🌐 HTMLを読み込み...")

    markdown = load_html_as_markdown(html_url)
    logger.info(f"📊 読み込み結果: {len(markdown)}文字")

    return markdown
