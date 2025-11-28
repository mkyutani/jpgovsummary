import os

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from .base import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    """Ollama LLMプロバイダー"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.validate_config()

    def validate_config(self) -> None:
        """Ollama固有の設定を検証する"""
        # Ollamaはローカル実行のためAPI keyは不要だが、接続先URLは確認する
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        if not base_url:
            raise ValueError("OLLAMA_BASE_URL environment variable not set")

    def get_llm(self) -> BaseChatModel:
        """ChatOllamaインスタンスを返す"""
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(base_url=base_url, model=self.model_name)
