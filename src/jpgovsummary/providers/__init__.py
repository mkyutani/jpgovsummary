import os

from .anthropic import AnthropicProvider
from .base import BaseLLMProvider
from .gemini import GeminiProvider
from .ollama import OllamaProvider
from .openai import OpenAIProvider

__all__ = [
    "BaseLLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "OllamaProvider",
    "get_provider",
]


def get_provider(model_name: str | None = None) -> BaseLLMProvider:
    """
    環境変数に基づいて適切なLLMプロバイダーを返す

    Args:
        model_name: 使用するモデル名（省略時は環境変数から取得）

    Returns:
        BaseLLMProvider: 選択されたプロバイダーのインスタンス

    Raises:
        ValueError: プロバイダーが不正または必要な環境変数が設定されていない場合
    """
    provider_name = os.environ.get("LLM_PROVIDER", "openai").lower()

    if provider_name == "openai":
        if model_name is None:
            model_name = os.environ.get("OPENAI_MODEL_NAME")
            if not model_name:
                raise ValueError("OPENAI_MODEL_NAME environment variable not set")
        return OpenAIProvider(model_name)

    elif provider_name == "anthropic":
        if model_name is None:
            model_name = os.environ.get("ANTHROPIC_MODEL_NAME")
            if not model_name:
                raise ValueError("ANTHROPIC_MODEL_NAME environment variable not set")
        return AnthropicProvider(model_name)

    elif provider_name == "gemini":
        if model_name is None:
            model_name = os.environ.get("GEMINI_MODEL_NAME")
            if not model_name:
                raise ValueError("GEMINI_MODEL_NAME environment variable not set")
        return GeminiProvider(model_name)

    elif provider_name == "ollama":
        if model_name is None:
            model_name = os.environ.get("OLLAMA_MODEL_NAME")
            if not model_name:
                raise ValueError("OLLAMA_MODEL_NAME environment variable not set")
        return OllamaProvider(model_name)

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider_name}. "
            f"Supported providers: openai, anthropic, gemini, ollama"
        )
