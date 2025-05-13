import os
import requests
from urllib.parse import urljoin

from langchain_core.tools import tool

from .. import logger

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

    logger.info("pdf_loader")

    return "Not implemented yet"