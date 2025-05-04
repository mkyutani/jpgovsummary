import os
import re
from typing import Any

from langchain_openai import ChatOpenAI
from openai import OpenAI

def is_uuid(id: str) -> bool:
    return re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", id) is not None

_openai_client = None

def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client

def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini")