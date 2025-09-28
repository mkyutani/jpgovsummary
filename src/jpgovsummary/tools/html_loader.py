import requests
from io import BytesIO
from langchain_core.tools import tool
from lxml import etree, html
from markitdown import MarkItDown

from .. import logger
from ..utils import is_local_file, get_local_file_path, validate_local_file


def _normalize_and_convert_html(html_content: str | bytes) -> str:
    """
    Normalize HTML with lxml and convert to markdown using MarkItDown.
    
    Args:
        html_content: HTML content as string or bytes
        
    Returns:
        str: markdown content
    """
    # lxmlでHTMLを正規化
    doc = html.fromstring(html_content)
    normalized_html = etree.tostring(doc, encoding='unicode', method='html')
    
    # 正規化されたHTMLをMarkItDownで変換
    html_bytes = normalized_html.encode('utf-8')
    html_stream = BytesIO(html_bytes)
    markdown = MarkItDown().convert(html_stream, file_extension='.html')
    
    return markdown.text_content


def load_html_as_markdown(url: str) -> str:
    """
    Load HTML page into markdown string from URL or local file.
    
    This function handles HTML normalization using lxml before converting
    to markdown, which can help with malformed HTML.

    Args:
        url (str): URL of the page with ending .html or local file path

    Returns:
        str: markdown of the page
    """
    if is_local_file(url):
        # Handle local file
        file_path = get_local_file_path(url)
        validate_local_file(file_path)
        logger.info(f"🟢 {file_path} (HTML)を読み込みます")

        # Read local HTML file content
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        return _normalize_and_convert_html(html_content)
    else:
        # Handle remote URL
        logger.info(f"🟢 {url} (HTML)を読み込みます")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=30, verify=True)
        response.raise_for_status()
        
        return _normalize_and_convert_html(response.content)
