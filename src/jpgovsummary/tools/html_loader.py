import os

from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_core.tools import tool

from .. import log

def load(url: str) -> str:
    loader = WebBaseLoader(
        url,
        requests_kwargs={
            "headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        }
    )
    docs = loader.load()
    return docs[0].page_content

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

    log("html_loader")

    markdown = load(html_url)
    return markdown