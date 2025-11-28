import os

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

from .base import BaseLLMProvider


class AnthropicProvider(BaseLLMProvider):
    """Anthropic LLMプロバイダー"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.validate_config()

    def validate_config(self) -> None:
        """Anthropic固有の設定を検証する"""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    def get_llm(self) -> BaseChatModel:
        """ChatAnthropicインスタンスを返す"""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        return ChatAnthropic(
            api_key=api_key,
            model=self.model_name,
            max_tokens=8192  # 長い出力に対応
        )
