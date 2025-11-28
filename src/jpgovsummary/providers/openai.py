import os

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLMプロバイダー"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.validate_config()

    def validate_config(self) -> None:
        """OpenAI固有の設定を検証する"""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

    def get_llm(self) -> BaseChatModel:
        """ChatOpenAIインスタンスを返す"""
        api_key = os.environ.get("OPENAI_API_KEY")
        return ChatOpenAI(api_key=api_key, model=self.model_name)
