import os
import requests
from urllib.parse import urljoin

from langchain_core.tools import tool

from .. import log

@tool
def pdf_loader(pdf_url: str) -> str:
    """
    ## PDF Loader

    Load PDF document into markdown string.

    Args:
        pdf_url (str): URL of the page with ending .pdf

    Returns:
        str: markdown of the page
    """

    log("html_loader_by_firecrawl")

    return "Not implemented yet"