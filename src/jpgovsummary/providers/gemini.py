import os

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from .base import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLMプロバイダー"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.validate_config()

    def validate_config(self) -> None:
        """Gemini固有の設定を検証する"""
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

    def get_llm(self) -> BaseChatModel:
        """ChatGoogleGenerativeAIインスタンスを返す"""
        api_key = os.environ.get("GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(api_key=api_key, model=self.model_name)
