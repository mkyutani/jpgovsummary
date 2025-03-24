import os

from langchain_community.document_loaders.firecrawl import FireCrawlLoader
from langchain_core.tools import tool

from .. import log

firecrawl_api_key = os.environ.get('FIRECRAWL_API_KEY')

def load(url: str) -> str:
    loader = FireCrawlLoader(
        api_key=firecrawl_api_key,
        url=url,
        mode="scrape"
    )

    pages = []
    for doc in loader.lazy_load():
        pages.append(doc)

    markdown = '\n'.join([page.page_content for page in pages])

    return markdown

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